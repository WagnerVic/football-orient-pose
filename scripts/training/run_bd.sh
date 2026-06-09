#!/usr/bin/env bash
# Cenários com augmentation — completa a matriz 2×2 + ladder fino do TL.
#
# Lado scratch (1 run): B = B-FULL (flip + geométrica + oclusão + blur).
# Lado TL (ladder fino, 3 runs):
#   D-GEOM = flip + geométrica (RandomBBoxTransform)
#   D-OCCL = D-GEOM + oclusão (RandomErasing)
#   D      = D-FULL = D-OCCL + blur (MotionBlur)
# Cada degrau adiciona UMA aug → atribui o efeito de cada mecanismo nas extremidades:
#   geométrico = D-GEOM − C-FLIP   ·   oclusão = D-OCCL − D-GEOM   ·   blur = D-FULL − D-OCCL
#
# Protocolo idêntico aos runs anteriores (paridade): scratch 150 ép., TL 45/60/45, batch 64.
# Autocontido em results/runs/<timestamp>_bd/.
#
# Uso (no container, com data/results/scripts/configs/src montados):
#   EPOCHS_B=150 F1=45 F2=60 F3=45 BATCH=64 GIT_COMMIT=$(git rev-parse HEAD) bash scripts/training/run_bd.sh
set -eo pipefail

EPOCHS_B=${EPOCHS_B:-150}
F1=${F1:-45}; F2=${F2:-60}; F3=${F3:-45}
BATCH=${BATCH:-64}
GIT_COMMIT=${GIT_COMMIT:-unknown}

TS=$(date +%Y%m%d_%H%M%S)
export RUN="results/runs/${TS}_bd"
LOG="$RUN/logs"
mkdir -p "$LOG" "$RUN/tables" "$RUN/checkpoints"

{
  echo "timestamp = ${TS}"
  echo "commit    = ${GIT_COMMIT}"
  echo "host      = $(hostname)"
  echo "scenarios = B (scratch+full) · D-GEOM · D-OCCL · D (TL ladder: geom/+oclusão/+blur)"
  echo "params    = EPOCHS_B=${EPOCHS_B} fases_TL=${F1}/${F2}/${F3} BATCH=${BATCH}"
} > "$RUN/PROVENANCE.txt"
cat "$RUN/PROVENANCE.txt"

step() { echo -e "\n========== $1 ==========\n"; }

train_scratch() {  # $1 = cenario, $2 = config-stem
  local cen="$1" cfg="$2"
  step "TREINO ${cen} (from scratch, ${EPOCHS_B} épocas)"
  python scripts/training/train.py --cenario "$cen" --epochs "$EPOCHS_B" --batch-size "$BATCH" \
      --work-dir "$RUN/checkpoints/cenario_${cen}" 2>&1 | tee "$LOG/train_${cen}.log"
  step "AVALIAÇÃO ${cen} — val"
  python scripts/evaluation/evaluate.py --checkpoint "$RUN/checkpoints/cenario_${cen}/best_PCK.pth" \
      --config "configs/cenario_${cfg}.py" --split val   --output-dir "$RUN/tables" 2>&1 | tee "$LOG/eval_${cen}_val.log"
  step "AVALIAÇÃO ${cen} — train"
  python scripts/evaluation/evaluate.py --checkpoint "$RUN/checkpoints/cenario_${cen}/best_PCK.pth" \
      --config "configs/cenario_${cfg}.py" --split train --output-dir "$RUN/tables" 2>&1 | tee "$LOG/eval_${cen}_train.log"
}

train_tl() {  # $1 = cenario, $2 = config-stem
  local cen="$1" cfg="$2"
  step "TREINO ${cen} (transfer learning, fases ${F1}/${F2}/${F3})"
  python scripts/training/train.py --cenario "$cen" --epochs-fase1 "$F1" --epochs-fase2 "$F2" --epochs-fase3 "$F3" \
      --batch-size "$BATCH" --work-dir "$RUN/checkpoints/cenario_${cen}" 2>&1 | tee "$LOG/train_${cen}.log"
  step "AVALIAÇÃO ${cen} — val"
  python scripts/evaluation/evaluate.py --checkpoint "$RUN/checkpoints/cenario_${cen}/best_PCK.pth" \
      --config "configs/cenario_${cfg}.py" --split val   --output-dir "$RUN/tables" 2>&1 | tee "$LOG/eval_${cen}_val.log"
  step "AVALIAÇÃO ${cen} — train"
  python scripts/evaluation/evaluate.py --checkpoint "$RUN/checkpoints/cenario_${cen}/best_PCK.pth" \
      --config "configs/cenario_${cfg}.py" --split train --output-dir "$RUN/tables" 2>&1 | tee "$LOG/eval_${cen}_train.log"
}

# ── Lado scratch: B = B-FULL ─────────────────────────────────────────────────
train_scratch "B" "b"

# ── Lado TL: ladder fino geom → +oclusão → +blur ─────────────────────────────
train_tl "D-GEOM" "d-geom"
train_tl "D-OCCL" "d-occl"
train_tl "D"      "d"

# ── Resumo ───────────────────────────────────────────────────────────────────
step "RESUMO (ladder de augmentation + atribuição por mecanismo)"
SUMMARY="$RUN/SUMMARY.md"
python - <<'PY' | tee "$SUMMARY"
import json, os
from pathlib import Path
tables = Path(os.environ["RUN"]) / "tables"

# Referências dos runs anteriores (val split) para comparar os degraus.
C_FLIP = 0.5839   # TL + flip (degrau anterior ao D-GEOM)
A_FLIP = 0.4606   # scratch + flip
BASELINE = 0.4176 # zero-shot COCO
# PCK por grupo do baseline COCO (extremidades) — do relatório §8.
BASE_GRP = {"wrist": 0.435, "elbow": 0.508, "knee": 0.583, "ankle": 0.594}

def load(cen, split):
    p = tables / f"finetuned_cenario_{cen}_{split}.json"
    return json.loads(p.read_text()) if p.exists() else None

def pct(x): return f"{x*100:.2f}%" if x is not None else "—"
def pp(x):  return f"{x*100:+.2f}pp" if x is not None else "—"

print("# Cenários B/D — augmentation (matriz 2×2 + ladder fino do TL)\n")

print("## Consolidado (val split)\n")
print("| Cenário | PCK@0.2 | PDJ@0.5 | OKS | MPJPE-2D |")
print("|---|---:|---:|---:|---:|")
rows = [("B", "B-FULL (scratch, flip+geom+oclusão+blur)"),
        ("D-GEOM", "D-GEOM (TL, +geométrica)"),
        ("D-OCCL", "D-OCCL (TL, +oclusão)"),
        ("D", "D-FULL (TL, +blur)")]
for cen, nome in rows:
    r = load(cen, "val")
    if r:
        print(f"| {nome} | {pct(r['pck_02'])} | {pct(r['pdj_05'])} | {pct(r['oks'])} | {r['mpjpe_2d_px']:.2f} px |")

print("\n## Over/underfitting (train vs val, PCK@0.2)\n")
print("| Cenário | PCK train | PCK val | gap |")
print("|---|---:|---:|---:|")
for cen, nome in rows:
    tr, va = load(cen, "train"), load(cen, "val")
    if tr and va:
        print(f"| {nome} | {pct(tr['pck_02'])} | {pct(va['pck_02'])} | {(tr['pck_02']-va['pck_02'])*100:.1f} pp |")

print("\n## Atribuição por mecanismo (lado TL, PCK@0.2 val)\n")
print("Cada degrau adiciona UMA aug sobre o anterior:\n")
dg, do, df = load("D-GEOM","val"), load("D-OCCL","val"), load("D","val")
print("| Mecanismo | comparação | Δ |")
print("|---|---|---:|")
if dg: print(f"| geométrico | D-GEOM − C-FLIP | {pp(dg['pck_02']-C_FLIP)} |")
if dg and do: print(f"| oclusão | D-OCCL − D-GEOM | {pp(do['pck_02']-dg['pck_02'])} |")
if do and df: print(f"| blur | D-FULL − D-OCCL | {pp(df['pck_02']-do['pck_02'])} |")
if df: print(f"| **total aug forte** | D-FULL − C-FLIP | **{pp(df['pck_02']-C_FLIP)}** |")

print(f"\n> Referências: C-FLIP={pct(C_FLIP)} · A-FLIP={pct(A_FLIP)} · baseline={pct(BASELINE)} (val).")

print("\n## Foco extremidades (PCK@0.2 por grupo vs baseline COCO)\n")
print("Onde o fine-tuning ainda perdia pro baseline (§8). Verde = passou o baseline.\n")
print("| Grupo | baseline | D-GEOM | D-OCCL | D-FULL |")
print("|---|---:|---:|---:|---:|")
for g in ("wrist", "elbow", "knee", "ankle"):
    def cell(r):
        if not r: return "—"
        v = r["pck_per_group"].get(g)
        if v is None: return "—"
        mark = " ✅" if v >= BASE_GRP[g] else ""
        return f"{v*100:.1f}%{mark}"
    print(f"| {g} | {BASE_GRP[g]*100:.1f}% | {cell(dg)} | {cell(do)} | {cell(df)} |")

print("\n> Hipótese-chave: algum degrau (geom/oclusão/blur) recupera as extremidades acima do baseline?")
PY

echo -e "\n[run_bd] CONCLUÍDO. Run em ${RUN}/ (tables/, checkpoints/, logs/, SUMMARY.md, PROVENANCE.txt)"
