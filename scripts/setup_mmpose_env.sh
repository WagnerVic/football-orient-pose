#!/usr/bin/env bash
# Cria um venv dedicado para a stack de fine-tuning (MMPose), pinado nas mesmas
# versões do Dockerfile.finetuning para reproduzir o ambiente do treino local.
#
# Uso: bash scripts/setup_mmpose_env.sh
set -euo pipefail

VENV=".venv-mmpose"
PY="$VENV/bin/python"

echo "[1/6] Criando venv ($VENV) com Python 3.11..."
uv venv --python 3.11 "$VENV"

echo "[2/6] Instalando torch 2.4.0 + cu121..."
uv pip install --python "$PY" \
    torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu121

echo "[3/6] numpy<2.0 (evita quebra de ABI do xtcocotools)..."
uv pip install --python "$PY" "numpy<2.0"

echo "[4/6] mmengine + mmcv 2.2.0 (wheel oficial para torch2.4/cu121)..."
uv pip install --python "$PY" "mmengine>=0.10.0"
uv pip install --python "$PY" "mmcv==2.2.0" \
    -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4.0/index.html

echo "[5/6] mmpose + deps de runtime (augmentation/metrics/io)..."
uv pip install --python "$PY" "mmpose>=1.3.0"
uv pip install --python "$PY" albumentations opencv-python scipy matplotlib tqdm

echo "[6/6] Instalando o projeto (editable, sem deps)..."
uv pip install --python "$PY" -e . --no-deps

echo ""
echo "=== Verificação ==="
"$PY" - <<'PYEOF'
import torch, mmengine, mmcv, mmpose, numpy
print("torch   ", torch.__version__, "cuda?", torch.cuda.is_available())
print("mmengine", mmengine.__version__)
print("mmcv    ", mmcv.__version__)
print("mmpose  ", mmpose.__version__)
print("numpy   ", numpy.__version__)
PYEOF
echo "OK — ambiente pronto em $VENV"
