#!/usr/bin/env bash
# Cenários da Tabela III com várias sementes: para cada semente e cada cenário, treina e avalia
# o melhor checkpoint no val e no treino (gap). Mesmo protocolo dos runs originais.
#
# Roda DENTRO do container de fine-tuning — use `make docker-seeds` dentro de um tmux.
# Acompanhe com `make monitor` (scripts/training/monitor.py), que lê o progress.tsv daqui.
#
# Parâmetros (variáveis de ambiente):
#   SEEDS="1 2 3 4 5 6 7 8 9"     sementes (cada uma roda todos os cenários)
#   SCENARIOS="A-RAW B C-RAW D"   A-RAW/B = do zero (fase única); C-RAW/D = transfer learning (3 fases)
#   SCENARIO_SEEDS="A-RAW:584961187 D:726553664"
#                                 alternativa a SEEDS × SCENARIOS: um par cenário:semente por vez
#                                 (ex.: repetir os runs originais com a semente registrada no log)
#   EPOCHS_SCRATCH=150            épocas de A-RAW e B
#   F1=45 F2=60 F3=45             épocas das fases do transfer learning
#   BATCH=64  DELTA_PCK=0.05      VAL_INTERVAL= (vazio = o do config)
#   RUN_NAME=<data>_seeds         pasta da execução em runs/
#   KEEP_EPOCH_CKPTS=0            1 = mantém epoch_*.pth (por padrão só os best_*.pth ficam)
#   RESUME=0                      1 = retoma RUN_NAME existente: pula etapas já concluídas e acrescenta
#                                 ao progress.tsv (etapa de treino incompleta é refeita do zero)
#
# Antes de cada etapa a GPU é conferida; sem GPU a execução inteira é interrompida (ABORTED=gpu no
# params.env, linha "abort" no progress.tsv) em vez de seguir para as próximas sementes.
#   REFERENCE=docs/experiments/reference_run1.json
#                                 métricas de referência; o monitor mostra a comparação com elas

set -uo pipefail

SEEDS=${SEEDS:-"1 2 3 4 5 6 7 8 9"}
SCENARIOS=${SCENARIOS:-"A-RAW B C-RAW D"}
SCENARIO_SEEDS=${SCENARIO_SEEDS:-}
EPOCHS_SCRATCH=${EPOCHS_SCRATCH:-150}
F1=${F1:-45}; F2=${F2:-60}; F3=${F3:-45}
BATCH=${BATCH:-64}
DELTA_PCK=${DELTA_PCK:-0.05}
VAL_INTERVAL=${VAL_INTERVAL:-}
KEEP_EPOCH_CKPTS=${KEEP_EPOCH_CKPTS:-0}
RESUME=${RESUME:-0}
RUN_NAME=${RUN_NAME:-$(date +%Y%m%d_%H%M%S)_seeds}
RUN="runs/${RUN_NAME}"

# Plano: lista de pares semente:cenário, na ordem de execução.
PLAN=()
if [[ -n "$SCENARIO_SEEDS" ]]; then
  for pair in $SCENARIO_SEEDS; do PLAN+=("${pair#*:}:${pair%%:*}"); done
else
  for seed in $SEEDS; do for scen in $SCENARIOS; do PLAN+=("${seed}:${scen}"); done; done
fi

mkdir -p "$RUN"
# O container roda como root: devolve a posse dos arquivos ao usuário do host ao sair.
trap '[[ -n "${HOST_UID:-}" ]] && chown -R "${HOST_UID}:${HOST_GID:-$HOST_UID}" "$RUN"' EXIT

if [[ "$RESUME" == 1 && -f "$RUN/params.env" && -f "$RUN/progress.tsv" ]]; then
  sed -i '/^FINISHED=/d; /^ABORTED=/d' "$RUN/params.env"
  echo "RESUMED=$(date -Iseconds) commit ${GIT_COMMIT:-unknown}" >> "$RUN/params.env"
else
cat > "$RUN/params.env" <<EOF
RUN_NAME=${RUN_NAME}
SEEDS=${SEEDS}
SCENARIOS=${SCENARIOS}
PLAN=${PLAN[*]}
EPOCHS_SCRATCH=${EPOCHS_SCRATCH}
F1=${F1}
F2=${F2}
F3=${F3}
BATCH=${BATCH}
DELTA_PCK=${DELTA_PCK}
VAL_INTERVAL=${VAL_INTERVAL}
GIT_COMMIT=${GIT_COMMIT:-unknown}
REFERENCE=${REFERENCE:-}
STARTED=$(date -Iseconds)
EOF
printf 'time\tseed\tscenario\tstep\tstatus\tseconds\tlog\n' > "$RUN/progress.tsv"
fi

config_of() {
  case "$1" in
    A-RAW) echo a-raw ;; B) echo b ;; C-RAW) echo c-raw ;; D) echo d ;;
    *) echo "cenário desconhecido: $1" >&2; return 1 ;;
  esac
}

progress() {  # seed scenario step status seconds log
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "$@" >> "$RUN/progress.tsv"
}

gpu_ok() {
  python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null
}

# Etapa já concluída numa execução anterior (só vale com RESUME=1).
already_done() {  # seed scenario step output
  [[ "$RESUME" == 1 && -f "$4" ]] &&
    awk -F'\t' -v s="$1" -v c="$2" -v n="$3" \
      '$2==s && $3==c && $4==n && $5=="done" {f=1} END {exit !f}' "$RUN/progress.tsv"
}

# Executa um passo com o log em arquivo e registra início/fim no progress.tsv.
step() {  # seed scenario step log -- comando...
  local seed=$1 scen=$2 name=$3 log=$4; shift 5
  if ! gpu_ok; then
    progress "$seed" "$scen" "$name" abort 0 "$log"
    echo "ABORTED=gpu $(date -Iseconds)" >> "$RUN/params.env"
    echo "[$(date +%H:%M:%S)] GPU INDISPONÍVEL antes de seed ${seed} · ${scen} · ${name} — execução interrompida"
    exit 3
  fi
  progress "$seed" "$scen" "$name" start 0 "$log"
  echo "[$(date +%H:%M:%S)] seed ${seed} · ${scen} · ${name}"
  local t0=$SECONDS
  "$@" > "$log" 2>&1
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    progress "$seed" "$scen" "$name" done $((SECONDS - t0)) "$log"
  else
    progress "$seed" "$scen" "$name" fail $((SECONDS - t0)) "$log"
    echo "    FALHOU (código ${rc}) — veja ${log}"
  fi
  return $rc
}

for item in "${PLAN[@]}"; do
    seed=${item%%:*}; scen=${item#*:}
    cfg=$(config_of "$scen") || exit 1
    dir="$RUN/seed_${seed}"
    work="$dir/cenario_${scen}"
    mkdir -p "$dir/logs" "$dir/tables"

    if [[ "$scen" == A-RAW || "$scen" == B ]]; then
      train_args=(--epochs "$EPOCHS_SCRATCH")
    else
      train_args=(--epochs-fase1 "$F1" --epochs-fase2 "$F2" --epochs-fase3 "$F3" --delta-pck "$DELTA_PCK")
    fi
    [[ -n "$VAL_INTERVAL" ]] && train_args+=(--val-interval "$VAL_INTERVAL")

    if already_done "$seed" "$scen" train "$work/best_PCK.pth"; then
      echo "[$(date +%H:%M:%S)] seed ${seed} · ${scen} · train já concluído — pulando"
    else
      rm -rf "$work"  # treino incompleto de uma execução anterior: recomeça limpo
      step "$seed" "$scen" train "$dir/logs/train_${scen}.log" -- \
        python scripts/training/train.py --cenario "$scen" "${train_args[@]}" \
          --batch-size "$BATCH" --seed "$seed" --work-dir "$work" || continue
    fi

    for split in val train; do
      out="$dir/tables/finetuned_cenario_${scen}_${split}.json"
      already_done "$seed" "$scen" "eval_${split}" "$out" && continue
      step "$seed" "$scen" "eval_${split}" "$dir/logs/eval_${scen}_${split}.log" -- \
        python scripts/evaluation/evaluate.py --checkpoint "$work/best_PCK.pth" \
          --config "configs/cenario_${cfg}.py" --split "$split" --output-dir "$dir/tables"
    done

    [[ "$KEEP_EPOCH_CKPTS" == 1 ]] || find "$work" -name 'epoch_*.pth' -delete
done

echo "FINISHED=$(date -Iseconds)" >> "$RUN/params.env"
python scripts/training/monitor.py "$RUN" --once --no-system \
  --summary-json "$RUN/summary.json" > "$RUN/SUMMARY.txt" 2>&1
echo "Concluído: $RUN (resumo em $RUN/SUMMARY.txt)"
