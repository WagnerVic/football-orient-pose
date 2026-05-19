#!/usr/bin/env bash
# Baixa e descompacta os pesos dos modelos de pose estimation.
# Uso: bash scripts/download_models.sh
#
# RTMPose:  não precisa de setup (rtmlib baixa automaticamente na 1ª inferência).
# HRNet:    baixa zip do Google Drive e descompacta o ONNX.
# OpenPose: baixa zip do Google Drive e descompacta o caffemodel (pendente upload).

set -euo pipefail

WEIGHTS_DIR="models/weights"
mkdir -p "$WEIGHTS_DIR"

# ---------------------------------------------------------------------------
# HRNet-W48 (256×192) — ONNX exportado, hospedado no Google Drive
# ---------------------------------------------------------------------------
HRNET_ONNX="$WEIGHTS_DIR/hrnet_w48_coco_256x192.onnx"
HRNET_GDRIVE_ID="1dNC22Hvp-oHqb6vYKuQhs7TQDLoanB1K"

if [ ! -f "$HRNET_ONNX" ]; then
    echo "[HRNet] Baixando pesos ONNX do Google Drive..."
    uv run gdown "$HRNET_GDRIVE_ID" -O "$WEIGHTS_DIR/hrnet_w48_coco_256x192.zip"
    echo "[HRNet] Descompactando..."
    unzip -o "$WEIGHTS_DIR/hrnet_w48_coco_256x192.zip" -d "$WEIGHTS_DIR"
    rm "$WEIGHTS_DIR/hrnet_w48_coco_256x192.zip"
    echo "[HRNet] Pronto."
else
    echo "[HRNet] ONNX já existe, pulando."
fi

# ---------------------------------------------------------------------------
# OpenPose COCO — caffemodel hospedado no Google Drive (pendente upload)
# ---------------------------------------------------------------------------
PROTO="$WEIGHTS_DIR/openpose_pose_coco.prototxt"
CAFFE="$WEIGHTS_DIR/openpose_pose_iter_440000.caffemodel"
# OPENPOSE_GDRIVE_ID="TODO"  # preencher após upload no Drive

if [ ! -f "$PROTO" ]; then
    echo "[OpenPose] Baixando prototxt..."
    wget -q --show-progress -O "$PROTO" \
        "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt"
else
    echo "[OpenPose] prototxt já existe, pulando."
fi

if [ ! -f "$CAFFE" ]; then
    echo "[OpenPose] AVISO: caffemodel pendente — faça upload no Google Drive e configure OPENPOSE_GDRIVE_ID no script."
    # Quando disponível, descomente e preencha o ID:
    # uv run gdown "$OPENPOSE_GDRIVE_ID" -O "$WEIGHTS_DIR/openpose.zip"
    # unzip -o "$WEIGHTS_DIR/openpose.zip" -d "$WEIGHTS_DIR"
    # rm "$WEIGHTS_DIR/openpose.zip"
else
    echo "[OpenPose] caffemodel já existe, pulando."
fi

echo ""
echo "Modelos em $WEIGHTS_DIR:"
ls -lh "$WEIGHTS_DIR"
