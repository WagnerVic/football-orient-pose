# Fine-tuning RTMPose-X — Épico 1

Infraestrutura de código para treinar o **RTMPose-X** nos 4 cenários da matriz
experimental 2×2 do trabalho de RNP e avaliar qualquer checkpoint com as 4
métricas do projeto (PDJ@0.5, PCK@0.2, OKS, MPJPE-2D).

> Épico: [#56](https://github.com/WagnerVic/football-orient-pose/issues/56) ·
> US: [#57](https://github.com/WagnerVic/football-orient-pose/issues/57) ·
> [#61](https://github.com/WagnerVic/football-orient-pose/issues/61) ·
> [#67](https://github.com/WagnerVic/football-orient-pose/issues/67)

## A matriz experimental 2×2

|  | Sem Augmentation | Com Augmentation |
|--|------------------|------------------|
| **From Scratch** (W₀ aleatório) | **Cenário A** | **Cenário B** |
| **Transfer Learning** (W₀ COCO) | **Cenário C** | **Cenário D** |

- **A/B** (`from scratch`): `load_from=None`, `frozen_stages=0`, LR uniforme, fase única.
- **C/D** (`transfer learning`): pesos COCO + *progressive unfreezing* em 3 fases
  (`frozen_stages` 4 → 2 → 1), com a fase 3 condicionada a `Δ PCK(fase2−fase1) > 5pp`.
- **A/C** sem augmentation · **B/D** com MotionBlur + RandomErasing (RandomFlip em todos).

## Componentes (e como mapeiam para as issues)

| Arquivo | Papel | Issue |
|---------|-------|-------|
| [`src/.../finetuning/dataset.py`](../src/football_orient_pose/finetuning/dataset.py) | `DSP3Dataset` — adapter 3DSP → protocolo MMPose; `visibility=0` p/ keypoints derivados `[0,7,8,9]`; `flip_indices` H3WB | [#57](https://github.com/WagnerVic/football-orient-pose/issues/57) · [#58](https://github.com/WagnerVic/football-orient-pose/issues/58) · [#60](https://github.com/WagnerVic/football-orient-pose/issues/60) |
| [`configs/cenario_a.py`](../configs/cenario_a.py) | Cenário A — from scratch, sem aug | [#62](https://github.com/WagnerVic/football-orient-pose/issues/62) |
| [`configs/cenario_b.py`](../configs/cenario_b.py) | Cenário B — from scratch, MotionBlur + RandomErasing | [#59](https://github.com/WagnerVic/football-orient-pose/issues/59) · [#63](https://github.com/WagnerVic/football-orient-pose/issues/63) |
| [`configs/cenario_c.py`](../configs/cenario_c.py) | Cenário C — TL, 3 fases (sem aug) | [#64](https://github.com/WagnerVic/football-orient-pose/issues/64) |
| [`configs/cenario_d.py`](../configs/cenario_d.py) | Cenário D — TL, 3 fases (com aug) | [#65](https://github.com/WagnerVic/football-orient-pose/issues/65) |
| [`scripts/train.py`](../scripts/train.py) | Orquestrador `--cenario [A\|B\|C\|D]` (fases + decisão Δ PCK) | [#66](https://github.com/WagnerVic/football-orient-pose/issues/66) |
| [`scripts/evaluate.py`](../scripts/evaluate.py) | Avaliação no val split — PDJ/PCK/OKS/MPJPE | [#69](https://github.com/WagnerVic/football-orient-pose/issues/69) |
| Letterboxing via `TopdownAffine(use_udp=True)` nos configs | 100×100 → 288×384 sem distorção | [#68](https://github.com/WagnerVic/football-orient-pose/issues/68) |
| [`src/.../estimators/rtmpose.py`](../src/football_orient_pose/estimators/rtmpose.py) | `RTMPoseEstimator.from_checkpoint()` p/ carregar `.pth` fine-tunado | [#70](https://github.com/WagnerVic/football-orient-pose/issues/70) |

## Setup do ambiente

A stack MMPose roda numa **venv dedicada** (`.venv-mmpose`) — a `.venv` principal do
projeto usa um torch incompatível com o `mmcv`. Versões fixadas (validadas):
torch 2.4.0+cu121 · mmcv 2.2.0 · mmpose 1.3.2 · mmdet 3.3.0 · numpy<2.0 · setuptools<81.

```bash
make finetuning-env          # cria .venv-mmpose com a stack pinada
make finetuning-checkpoint   # baixa os pesos COCO p/ Transfer Learning (C/D)
```

> `make finetuning-env` chama [`scripts/setup_mmpose_env.sh`](../scripts/setup_mmpose_env.sh).
> Requer `uv` instalado.

## Rodando os experimentos

```bash
# Smoke test rápido (subconjunto, ~30s) — valida o pipeline ponta-a-ponta
make finetuning-smoke

# Treino completo de cada cenário
make train-a   # = python scripts/train.py --cenario A
make train-c   # Cenário C: 3 fases de progressive unfreezing

# Treino leve p/ GPU pequena (ex.: RTX 4050 6GB)
PYTHONPATH=src .venv-mmpose/bin/python scripts/train.py \
  --cenario A --epochs 10 --batch-size 8 --val-interval 2
```

Flags úteis do `train.py`: `--epochs`, `--epochs-fase{1,2,3}`, `--batch-size`,
`--val-interval`, `--delta-pck`, `--device {cuda,cpu}`, `--work-dir`.

Avaliação de um checkpoint salvo:

```bash
make evaluate CKPT=results/checkpoints/cenario_A/best_PCK.pth CONFIG=configs/cenario_a.py
# ou direto:
PYTHONPATH=src .venv-mmpose/bin/python scripts/evaluate.py \
  --checkpoint results/checkpoints/cenario_A/best_PCK.pth \
  --config configs/cenario_a.py --split val
```

## Detalhes não óbvios (lidos de cabeça nas correções)

- **bbox = crop inteiro.** O `bbox` no JSON do 3DSP é a posição do crop no frame
  broadcast *original*, mas `img/NNN.jpg` já é o crop 100×100 e os keypoints estão
  em coordenadas do crop. O dataset usa `bbox=[0,0,100,100]`; usar o `(x,y)` do
  frame jogaria a ROI para fora do crop e **zeraria a SimCC loss** (treino sem sinal).
- **Dados em `data/3dsp/train/<clip>`** (não `data/train`). Os configs usam `data_root="data/3dsp"`.
- **`default_scope="mmpose"`** nos configs e **`mmdet` instalado** são obrigatórios:
  sem eles o `register_all_modules()` e a construção do `TopdownPoseEstimator` falham.

## Rodar no RTX 4090 via SSH **sem sudo** (recomendado)

Não é preciso Docker nem `sudo`/`apt`: o `setup_mmpose_env.sh` instala tudo em user-space
com `uv` (a stack MMPose vai para `.venv-mmpose`). Num servidor com driver NVIDIA já
instalado (sempre é, em máquina com GPU), basta:

```bash
git clone <repo> && cd football-orient-pose
make setup                   # extrai os dados (data/3dsp/...)
make finetuning-env          # cria .venv-mmpose (user-space, sem sudo)
make finetuning-checkpoint   # baixa pesos COCO
make train-c                 # roda o cenário (usa a GPU diretamente via PyTorch)
```

> Se `uv` não estiver instalado: `curl -LsSf https://astral.sh/uv/install.sh | sh` (também user-space).
> Usamos `opencv-python-headless` justamente para não depender de `libGL.so.1` (lib de sistema).

## Docker — quando o host **não consegue baixar** a stack

Se o 4090 via SSH não tem internet (ou banda) para baixar torch/mmcv/mmpose (~vários GB),
o Docker resolve: **constrói-se a imagem onde há internet** e transfere-se o resultado. A
imagem traz a stack validada **+ pesos COCO baked** → no host **não baixa nada**.

```bash
# 1. Numa máquina COM internet (ex.: seu laptop):
make docker-save                  # build + salva em finetuning-image.tar.gz

# 2. Transfere o tarball para o host:
scp finetuning-image.tar.gz user@host-4090:~/

# 3. No host (não baixa nada):
docker load -i finetuning-image.tar.gz
make docker-train CENARIO=C       # ou: docker compose -f docker-compose.finetuning.yml run --rm train ...
```

**Pré-requisito de GPU no host (uma vez, por um admin com sudo):** NVIDIA Container Toolkit —
`sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`. Depois
disso, usuários no grupo `docker` rodam **sem sudo**.

> **Cheque sem sudo se o host já está pronto:** `docker info | grep -i nvidia` (deve listar o
> runtime `nvidia`) ou `docker run --rm --gpus all ubuntu nvidia-smi`. Se funcionar, é só
> `docker load` + `docker-train`. Se o runtime nvidia **não** estiver configurado e você não
> tiver sudo, peça ao admin do host para rodar o comando `nvidia-ctk` acima (passo único).
