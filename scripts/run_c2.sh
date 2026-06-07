#!/usr/bin/env bash
# Roda o Cenário C2 (Transfer Learning fase única — ablação do progressive unfreezing):
# treino + avaliação (val e train), autocontido em results/runs/<timestamp>_c2/.
#
# Uso (no container, com data/results/scripts/configs/src montados):
#   EPOCHS=150 BATCH=64 GIT_COMMIT=$(git rev-parse HEAD) bash scripts/run_c2.sh
set -eo pipefail

EPOCHS=${EPOCHS:-150}
BATCH=${BATCH:-64}
GIT_COMMIT=${GIT_COMMIT:-unknown}

TS=$(date +%Y%m%d_%H%M%S)
RUN="results/runs/${TS}_c2"
LOG="$RUN/logs"
mkdir -p "$LOG" "$RUN/tables" "$RUN/checkpoints"

{
  echo "timestamp = ${TS}"
  echo "commit    = ${GIT_COMMIT}"
  echo "host      = $(hostname)"
  echo "scenario  = C2 (Transfer Learning, fase única, frozen_stages=2)"
  echo "params    = EPOCHS=${EPOCHS} BATCH=${BATCH}"
} > "$RUN/PROVENANCE.txt"
cat "$RUN/PROVENANCE.txt"

step() { echo -e "\n========== $1 ==========\n"; }

step "TREINO Cenário C2 (TL fase única, ${EPOCHS} épocas)"
python scripts/train.py --cenario C2 --epochs "$EPOCHS" --batch-size "$BATCH" \
    --work-dir "$RUN/checkpoints/cenario_C2" 2>&1 | tee "$LOG/train_C2.log"

step "AVALIAÇÃO Cenário C2 — val"
python scripts/evaluate.py --checkpoint "$RUN/checkpoints/cenario_C2/best_PCK.pth" \
    --config configs/cenario_c2.py --split val   --output-dir "$RUN/tables" 2>&1 | tee "$LOG/eval_C2_val.log"
step "AVALIAÇÃO Cenário C2 — train"
python scripts/evaluate.py --checkpoint "$RUN/checkpoints/cenario_C2/best_PCK.pth" \
    --config configs/cenario_c2.py --split train --output-dir "$RUN/tables" 2>&1 | tee "$LOG/eval_C2_train.log"

echo -e "\n[run_c2] CONCLUÍDO. Run em ${RUN}/ (tables/, checkpoints/, logs/, PROVENANCE.txt)"
