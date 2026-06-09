#!/usr/bin/env bash
# Ablação do flip horizontal: roda A-RAW (from scratch) e C-RAW (transfer learning),
# ambos SEM RandomFlip, de ponta a ponta (treino + avaliação train/val).
# Comparar com A e C (que têm flip) isola o impacto do flip. Autocontido em
# results/runs/<timestamp>_raw/.
#
# Uso (no container, com data/results/scripts/configs/src montados):
#   EPOCHS_A=150 F1=45 F2=60 F3=45 BATCH=64 GIT_COMMIT=$(git rev-parse HEAD) bash scripts/run_raw.sh
set -eo pipefail

EPOCHS_A=${EPOCHS_A:-150}
F1=${F1:-45}; F2=${F2:-60}; F3=${F3:-45}
BATCH=${BATCH:-64}
GIT_COMMIT=${GIT_COMMIT:-unknown}

TS=$(date +%Y%m%d_%H%M%S)
export RUN="results/runs/${TS}_raw"
LOG="$RUN/logs"
mkdir -p "$LOG" "$RUN/tables" "$RUN/checkpoints"

{
  echo "timestamp = ${TS}"
  echo "commit    = ${GIT_COMMIT}"
  echo "host      = $(hostname)"
  echo "scenarios = A-RAW, C-RAW (ablação do flip horizontal — sem augmentation nenhuma)"
  echo "params    = EPOCHS_A=${EPOCHS_A} fases_C=${F1}/${F2}/${F3} BATCH=${BATCH}"
} > "$RUN/PROVENANCE.txt"
cat "$RUN/PROVENANCE.txt"

step() { echo -e "\n========== $1 ==========\n"; }

# ── A-RAW: from scratch, sem flip ────────────────────────────────────────────
step "TREINO A-RAW (from scratch, sem flip, ${EPOCHS_A} épocas)"
python scripts/train.py --cenario A-RAW --epochs "$EPOCHS_A" --batch-size "$BATCH" \
    --work-dir "$RUN/checkpoints/cenario_A-RAW" 2>&1 | tee "$LOG/train_A-RAW.log"

step "AVALIAÇÃO A-RAW — val"
python scripts/evaluate.py --checkpoint "$RUN/checkpoints/cenario_A-RAW/best_PCK.pth" \
    --config configs/cenario_a-raw.py --split val   --output-dir "$RUN/tables" 2>&1 | tee "$LOG/eval_A-RAW_val.log"
step "AVALIAÇÃO A-RAW — train"
python scripts/evaluate.py --checkpoint "$RUN/checkpoints/cenario_A-RAW/best_PCK.pth" \
    --config configs/cenario_a-raw.py --split train --output-dir "$RUN/tables" 2>&1 | tee "$LOG/eval_A-RAW_train.log"

# ── C-RAW: transfer learning (3 fases), sem flip ─────────────────────────────
step "TREINO C-RAW (transfer learning, sem flip, fases ${F1}/${F2}/${F3})"
python scripts/train.py --cenario C-RAW --epochs-fase1 "$F1" --epochs-fase2 "$F2" --epochs-fase3 "$F3" \
    --batch-size "$BATCH" --work-dir "$RUN/checkpoints/cenario_C-RAW" 2>&1 | tee "$LOG/train_C-RAW.log"

step "AVALIAÇÃO C-RAW — val"
python scripts/evaluate.py --checkpoint "$RUN/checkpoints/cenario_C-RAW/best_PCK.pth" \
    --config configs/cenario_c-raw.py --split val   --output-dir "$RUN/tables" 2>&1 | tee "$LOG/eval_C-RAW_val.log"
step "AVALIAÇÃO C-RAW — train"
python scripts/evaluate.py --checkpoint "$RUN/checkpoints/cenario_C-RAW/best_PCK.pth" \
    --config configs/cenario_c-raw.py --split train --output-dir "$RUN/tables" 2>&1 | tee "$LOG/eval_C-RAW_train.log"

# ── Resumo ───────────────────────────────────────────────────────────────────
step "RESUMO (impacto do flip = comparar com A/C que têm flip)"
SUMMARY="$RUN/SUMMARY.md"
python - <<'PY' | tee "$SUMMARY"
import json, os
from pathlib import Path
tables = Path(os.environ["RUN"]) / "tables"

def load(cen, split):
    p = tables / f"finetuned_cenario_{cen}_{split}.json"
    return json.loads(p.read_text()) if p.exists() else None

def pct(x): return f"{x*100:.2f}%" if x is not None else "—"

print("# Ablação do flip — A-RAW e C-RAW (sem flip)\n")
print("| Cenário | PCK@0.2 | PDJ@0.5 | OKS | MPJPE-2D |")
print("|---|---:|---:|---:|---:|")
for cen, nome in [("A-RAW", "A-RAW (scratch, sem flip)"), ("C-RAW", "C-RAW (TL, sem flip)")]:
    r = load(cen, "val")
    if r:
        print(f"| {nome} | {pct(r['pck_02'])} | {pct(r['pdj_05'])} | {pct(r['oks'])} | {r['mpjpe_2d_px']:.2f} px |")

print("\n## Over/underfitting (train vs val)\n")
print("| Cenário | PCK train | PCK val | gap |")
print("|---|---:|---:|---:|")
for cen, nome in [("A-RAW", "A-RAW"), ("C-RAW", "C-RAW")]:
    tr, va = load(cen, "train"), load(cen, "val")
    if tr and va:
        print(f"| {nome} | {pct(tr['pck_02'])} | {pct(va['pck_02'])} | {(tr['pck_02']-va['pck_02'])*100:.1f} pp |")

print("\n> Impacto do flip = (A com flip) − (A-RAW) e (C com flip) − (C-RAW).")
print("> A=46,06% · C=58,39% (val, com flip) — comparar com os valores acima.")
PY

echo -e "\n[run_raw] CONCLUÍDO. Run em ${RUN}/ (tables/, checkpoints/, logs/, SUMMARY.md, PROVENANCE.txt)"
