#!/usr/bin/env bash
# Cenários da Tabela III com várias sementes: para cada semente e cada cenário, treina e avalia
# o melhor checkpoint no val e no treino (gap). Mesmo protocolo dos runs originais.
#
# Roda DENTRO do container de fine-tuning — use `make docker-seeds` dentro de um tmux.
# Acompanhe com `make monitor` (scripts/training/monitor.py), que lê o progress.tsv daqui.
#
# Parâmetros (variáveis de ambiente):
#   SEEDS="1 2 3 4 5 6 7 8 9"     sementes
#   SCENARIOS="A-RAW B C-RAW D"   A-RAW/B = do zero (fase única); C-RAW/D = transfer learning (3 fases)
#   EPOCHS_SCRATCH=150            épocas de A-RAW e B
#   F1=45 F2=60 F3=45             épocas das fases do transfer learning
#   BATCH=64  DELTA_PCK=0.05      VAL_INTERVAL= (vazio = o do config)
#   RUN_NAME=<data>_seeds         pasta da execução em runs/
#   KEEP_EPOCH_CKPTS=0            1 = mantém epoch_*.pth (por padrão só os best_*.pth ficam)

set -uo pipefail

SEEDS=${SEEDS:-"1 2 3 4 5 6 7 8 9"}
SCENARIOS=${SCENARIOS:-"A-RAW B C-RAW D"}
EPOCHS_SCRATCH=${EPOCHS_SCRATCH:-150}
F1=${F1:-45}; F2=${F2:-60}; F3=${F3:-45}
BATCH=${BATCH:-64}
DELTA_PCK=${DELTA_PCK:-0.05}
VAL_INTERVAL=${VAL_INTERVAL:-}
KEEP_EPOCH_CKPTS=${KEEP_EPOCH_CKPTS:-0}
RUN_NAME=${RUN_NAME:-$(date +%Y%m%d_%H%M%S)_seeds}
RUN="runs/${RUN_NAME}"

mkdir -p "$RUN"
# O container roda como root: devolve a posse dos arquivos ao usuário do host ao sair.
trap '[[ -n "${HOST_UID:-}" ]] && chown -R "${HOST_UID}:${HOST_GID:-$HOST_UID}" "$RUN"' EXIT

cat > "$RUN/params.env" <<EOF
RUN_NAME=${RUN_NAME}
SEEDS=${SEEDS}
SCENARIOS=${SCENARIOS}
EPOCHS_SCRATCH=${EPOCHS_SCRATCH}
F1=${F1}
F2=${F2}
F3=${F3}
BATCH=${BATCH}
DELTA_PCK=${DELTA_PCK}
VAL_INTERVAL=${VAL_INTERVAL}
GIT_COMMIT=${GIT_COMMIT:-unknown}
STARTED=$(date -Iseconds)
EOF
printf 'time\tseed\tscenario\tstep\tstatus\tseconds\tlog\n' > "$RUN/progress.tsv"

config_of() {
  case "$1" in
    A-RAW) echo a-raw ;; B) echo b ;; C-RAW) echo c-raw ;; D) echo d ;;
    *) echo "cenário desconhecido: $1" >&2; return 1 ;;
  esac
}

progress() {  # seed scenario step status seconds log
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "$@" >> "$RUN/progress.tsv"
}

# Executa um passo com o log em arquivo e registra início/fim no progress.tsv.
step() {  # seed scenario step log -- comando...
  local seed=$1 scen=$2 name=$3 log=$4; shift 5
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

for seed in $SEEDS; do
  for scen in $SCENARIOS; do
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

    step "$seed" "$scen" train "$dir/logs/train_${scen}.log" -- \
      python scripts/training/train.py --cenario "$scen" "${train_args[@]}" \
        --batch-size "$BATCH" --seed "$seed" --work-dir "$work" || continue

    for split in val train; do
      step "$seed" "$scen" "eval_${split}" "$dir/logs/eval_${scen}_${split}.log" -- \
        python scripts/evaluation/evaluate.py --checkpoint "$work/best_PCK.pth" \
          --config "configs/cenario_${cfg}.py" --split "$split" --output-dir "$dir/tables"
    done

    [[ "$KEEP_EPOCH_CKPTS" == 1 ]] || find "$work" -name 'epoch_*.pth' -delete
  done
done

echo "FINISHED=$(date -Iseconds)" >> "$RUN/params.env"
python scripts/training/monitor.py "$RUN" --once --no-system \
  --summary-json "$RUN/summary.json" > "$RUN/SUMMARY.txt" 2>&1
echo "Concluído: $RUN (resumo em $RUN/SUMMARY.txt)"
