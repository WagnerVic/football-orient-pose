#!/usr/bin/env bash
# Baixa o Cascade R-CNN (COCO) do model zoo do mmdet e roda o benchmark (Épico #113).
# Pensado para rodar DENTRO da imagem Docker do finetuning (tem mmdet + mim).
# Uso (no container):  bash scripts/evaluation/eval_cascade.sh
set -euo pipefail

DEST=${1:-/tmp/cascade}
MODEL=cascade-rcnn_r50_fpn_1x_coco

echo ">> Baixando $MODEL para $DEST ..."
mim download mmdet --config "$MODEL" --dest "$DEST"

CFG="$DEST/$MODEL.py"
CKPT=$(ls "$DEST"/cascade*r50_fpn_1x_coco*.pth | head -1)
echo ">> config:     $CFG"
echo ">> checkpoint: $CKPT"

python scripts/evaluation/eval_detectors.py \
    --detector cascade-rcnn --device cuda \
    --config "$CFG" --checkpoint "$CKPT" \
    --save-predictions --viz 3
