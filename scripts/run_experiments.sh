#!/usr/bin/env bash
# Roda Cenário A e C de ponta a ponta (treino + avaliação train/val) e gera um
# resumo consolidado. Pensado para "fire and forget" via `docker run -d`.
#
# Uso (dentro do container, com data/results/scripts/configs/src montados):
#   bash scripts/run_experiments.sh
#   EPOCHS_A=200 F1=60 F2=80 F3=60 BATCH=32 bash scripts/run_experiments.sh
#
# Saídas:
#   results/logs/exp_<timestamp>/            — logs de cada etapa (.log)
#   results/tables/finetuned_cenario_*.json  — métricas (4 métricas) train e val
#   results/logs/exp_<timestamp>/SUMMARY.md  — tabela consolidada A vs C (train/val)
set -eo pipefail

EPOCHS_A=${EPOCHS_A:-150}        # épocas do Cenário A (from scratch)
F1=${F1:-45}; F2=${F2:-60}; F3=${F3:-45}   # épocas das 3 fases do Cenário C (TL)
BATCH=${BATCH:-32}

TS=$(date +%Y%m%d_%H%M%S)
LOG="results/logs/exp_${TS}"
mkdir -p "$LOG" results/tables
echo "[run_experiments] início ${TS} | EPOCHS_A=${EPOCHS_A} fases C=${F1}/${F2}/${F3} batch=${BATCH}"
echo "[run_experiments] logs em ${LOG}/"

step() { echo -e "\n========== $1 ==========\n"; }

# ── Cenário A: from scratch ──────────────────────────────────────────────────
step "TREINO Cenário A (from scratch, ${EPOCHS_A} épocas)"
python scripts/train.py --cenario A --epochs "$EPOCHS_A" --batch-size "$BATCH" 2>&1 | tee "$LOG/train_A.log"

step "AVALIAÇÃO Cenário A — val"
python scripts/evaluate.py --checkpoint results/checkpoints/cenario_A/best_PCK.pth \
    --config configs/cenario_a.py --split val   2>&1 | tee "$LOG/eval_A_val.log"
step "AVALIAÇÃO Cenário A — train"
python scripts/evaluate.py --checkpoint results/checkpoints/cenario_A/best_PCK.pth \
    --config configs/cenario_a.py --split train 2>&1 | tee "$LOG/eval_A_train.log"

# ── Cenário C: transfer learning (3 fases) ───────────────────────────────────
step "TREINO Cenário C (transfer learning, fases ${F1}/${F2}/${F3})"
python scripts/train.py --cenario C --epochs-fase1 "$F1" --epochs-fase2 "$F2" --epochs-fase3 "$F3" \
    --batch-size "$BATCH" 2>&1 | tee "$LOG/train_C.log"

step "AVALIAÇÃO Cenário C — val"
python scripts/evaluate.py --checkpoint results/checkpoints/cenario_C/best_PCK.pth \
    --config configs/cenario_c.py --split val   2>&1 | tee "$LOG/eval_C_val.log"
step "AVALIAÇÃO Cenário C — train"
python scripts/evaluate.py --checkpoint results/checkpoints/cenario_C/best_PCK.pth \
    --config configs/cenario_c.py --split train 2>&1 | tee "$LOG/eval_C_train.log"

# ── Resumo consolidado ───────────────────────────────────────────────────────
step "RESUMO CONSOLIDADO"
SUMMARY="$LOG/SUMMARY.md"
python - <<'PY' | tee "$SUMMARY"
import json, os
from pathlib import Path

def load(cen, split):
    p = Path("results/tables") / f"finetuned_cenario_{cen}_{split}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())

def pct(x): return f"{x*100:.2f}%" if x is not None else "—"

print(f"# Resumo do experimento\n")
print("## Métricas no val\n")
print("| Cenário | PCK@0.2 | PDJ@0.5 | OKS | MPJPE-2D |")
print("|---|---:|---:|---:|---:|")
for cen, nome in [("A", "A (from scratch)"), ("C", "C (transfer learning)")]:
    r = load(cen, "val")
    if r:
        print(f"| {nome} | {pct(r['pck_02'])} | {pct(r['pdj_05'])} | {pct(r['oks'])} | {r['mpjpe_2d_px']:.2f} px |")

print("\n## Over/underfitting (train vs val)\n")
print("| Cenário | PCK train | PCK val | gap |")
print("|---|---:|---:|---:|")
for cen, nome in [("A", "A (from scratch)"), ("C", "C (transfer learning)")]:
    tr, va = load(cen, "train"), load(cen, "val")
    if tr and va:
        gap = (tr['pck_02'] - va['pck_02']) * 100
        print(f"| {nome} | {pct(tr['pck_02'])} | {pct(va['pck_02'])} | {gap:.1f} pp |")
print("\n> gap grande (train ≫ val) = overfitting · gap pequeno = generaliza bem")
PY

echo -e "\n[run_experiments] CONCLUÍDO. Resumo: ${SUMMARY}"
