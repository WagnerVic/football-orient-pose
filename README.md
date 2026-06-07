# football-orient-pose
Football player pose estimation and body orientation analysis from broadcast video using YOLO11 + HRNet

## Setup

### 1. Dataset

Os datasets estão compactados em `.zip` para versionamento no Git. Após clonar o repositório:

**Linux / macOS**
```bash
make setup        # Descompacta os .zip para data/
make clean-data   # Remove data/ para re-extrair do zero
make help         # Lista os comandos disponíveis
```

**Windows (PowerShell)**
```powershell
.\setup_data.ps1          # Descompacta os .zip para data/
.\setup_data.ps1 -Clean   # Remove data/ para re-extrair do zero
```

> Se o PowerShell bloquear a execução, rode antes: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

Estrutura após extração:
```
data/
└── 3dsp/
    └── train/
        └── 00001/
            ├── img/       # Imagens dos jogadores (100×100 px)
            └── posture/   # Anotações de pose (JSON, formato H3WB-17)
```

### 2. Variáveis de ambiente

```bash
cp .env.example .env
# edite DATA_DIR se o dataset estiver em outro caminho
```

### 3. Pesos dos modelos

Os pesos não são versionados no Git (`models/weights/` está no `.gitignore`).
Execute o script abaixo para baixar todos os modelos:

```bash
bash scripts/download_models.sh
```

O que o script faz por modelo:

| Modelo | Ação |
|--------|------|
| **RTMPose** | Nada — o `rtmlib` baixa automaticamente na 1ª inferência |
| **HRNet-W48** | Baixa `.zip` do Google Drive e descompacta o ONNX (~450 MB) |
| **OpenPose** | Pendente — upload no Drive em andamento |

> **Dependência:** o script usa `gdown` para baixar do Google Drive. Já incluído nas dependências do projeto (`uv sync` instala automaticamente).

---

## Estrutura do projeto

```
football-orient-pose/
│
├── configs/
│   └── split.json                  # Split oficial train/val (seed=42, 160/40 clips)
│
├── data/                           # Dataset 3DSP — ignorado pelo Git
│   └── train/
│       └── 00001/
│           ├── img/                # Frames do jogador (100×100 px, BGR)
│           └── posture/            # Anotações H3WB-17 por frame (JSON)
│
├── docs/                           # Documentação por projeto (ver docs/README.md)
│   ├── vision/                     # Projeto de visão (estimadores, baseline)
│   └── finetuning/                 # Projeto Transfer Learning / RNP
│
├── models/
│   └── weights/                    # Pesos dos modelos — ignorado pelo Git
│       ├── hrnet_w48_coco_256x192.onnx        # ONNX para inferência
│       └── hrnet_w48_coco_256x192.onnx.data   # Pesos externos do ONNX
│
├── notebooks/
│   ├── 01_rtmpose_validation.ipynb # Validação RTMPose — PDJ 93.62%
│   └── 02_hrnet_validation.ipynb   # Validação HRNet   — PDJ 88.90%
│
├── results/
│   ├── figures/                    # Gráficos gerados (ignorado pelo Git)
│   └── tables/                     # Métricas JSON por modelo/split
│
├── scripts/
│   └── download_models.sh          # Baixa pesos do Google Drive e descompacta
│
├── src/football_orient_pose/
│   ├── pose.py                     # BasePoseEstimator (ABC — interface unificada)
│   │
│   ├── estimators/
│   │   ├── rtmpose.py              # RTMPoseEstimator  — PDJ 93.62% (val)
│   │   ├── hrnet.py                # HRNetEstimator    — PDJ 88.90% (val)
│   │   └── openpose.py             # OpenPoseEstimator — pendente pesos
│   │
│   ├── evaluation/
│   │   ├── metrics.py              # PDJ, PCK, OKS, MPJPE-2D, F1
│   │   ├── evaluate.py             # CLI: --model rtmpose|hrnet|openpose
│   │   └── baseline.py             # Baseline de orientação corporal
│   │
│   └── utils/
│       ├── keypoint_mapping.py     # coco17_to_h3wb17()
│       ├── data_io.py              # load_clip_image(), load_keypoints_2d()
│       ├── dataset.py              # DSPDataset (PyTorch DataLoader)
│       └── skeleton.py             # H3WB_SWAP_PAIRS
│
├── tests/                          # 53 testes unitários
│   ├── test_pose.py
│   ├── test_rtmpose_estimator.py
│   ├── test_hrnet_estimator.py
│   ├── test_openpose_estimator.py
│   ├── test_metrics.py
│   ├── test_dataset.py
│   └── test_data_split.py
│
├── .env.example                    # Template de variáveis de ambiente
└── pyproject.toml
```

---

## Avaliação

```bash
# Avaliar um modelo no split de validação
uv run python -m football_orient_pose.evaluation.evaluate \
  --model rtmpose \
  --split val \
  --device cuda \
  --data-dir data \
  --split-config configs/split.json \
  --output-dir results/tables
```

| Modelo | PDJ@0.5 | Referência (paper) |
|--------|---------|-------------------|
| RTMPose-X | **93.62%** | 89.51% |
| HRNet-W48 | **88.90%** | ~56.08% |
| OpenPose  | — | inédito |

---

## Fine-tuning (Épico 1)

Infraestrutura para treinar o RTMPose-X nos 4 cenários da matriz experimental 2×2
(From Scratch vs Transfer Learning × Com vs Sem augmentation) e avaliar checkpoints
com PDJ@0.5, PCK@0.2, OKS e MPJPE-2D.

### Setup — Build da imagem Docker

Requer Docker + NVIDIA Container Toolkit (GPU). Os pesos COCO para Transfer Learning
são baixados automaticamente durante o build.

```bash
docker compose -f docker-compose.finetuning.yml build
```

Ou pull da imagem pré-buildada:

```bash
docker pull phael777/football-finetuning:latest
```

### Treino — Cenários individuais

```bash
# Cenário A — From Scratch, sem augmentation (baseline)
docker compose -f docker-compose.finetuning.yml run --rm train \
    python scripts/train.py --cenario A --epochs 50

# Cenário B — From Scratch, com augmentation
docker compose -f docker-compose.finetuning.yml run --rm train \
    python scripts/train.py --cenario B --epochs 50

# Cenário C — Transfer Learning, sem augmentation [ALTA PRIORIDADE]
docker compose -f docker-compose.finetuning.yml run --rm train \
    python scripts/train.py --cenario C \
    --epochs-fase1 15 --epochs-fase2 20 --epochs-fase3 15

# Cenário D — Transfer Learning, com augmentation
docker compose -f docker-compose.finetuning.yml run --rm train \
    python scripts/train.py --cenario D \
    --epochs-fase1 15 --epochs-fase2 20 --epochs-fase3 15
```

### Treino — Todos os cenários em sequência

```bash
for CENARIO in A B C D; do
  docker compose -f docker-compose.finetuning.yml run --rm train \
    python scripts/train.py --cenario $CENARIO
done
```

### Treino orquestrado A + C (fire-and-forget) — usado nos experimentos

`scripts/run_experiments.sh` roda os Cenários A e C de ponta a ponta
(treino → avaliação train/val de cada um → resumo consolidado) num container
**destacado** (`-d`), que sobrevive à queda da sessão SSH. Foi este o comando usado
para gerar os resultados abaixo (RTX 4090, GPU via CDI — `--gpus all`, sem sudo):

```bash
cd ~/football-orient-pose && git pull   # garante o código atualizado no host

docker run -d --name exp --gpus all --shm-size=16g \
  -v ~/football-orient-pose/data:/workspace/data:ro \
  -v ~/football-orient-pose/results:/workspace/results \
  -v ~/football-orient-pose/scripts:/workspace/scripts:ro \
  -v ~/football-orient-pose/configs:/workspace/configs:ro \
  -v ~/football-orient-pose/src:/workspace/src:ro \
  -e EPOCHS_A=200 -e F1=60 -e F2=80 -e F3=60 \
  football-finetuning:latest bash scripts/run_experiments.sh
```

Parâmetros (via `-e`): `EPOCHS_A` (épocas do A), `F1/F2/F3` (épocas das 3 fases do C),
`BATCH` (default 32). Sem `-e`, usa os defaults `EPOCHS_A=150`, `F1/F2/F3=45/60/45`.

Acompanhar o progresso e ver os resultados:

```bash
docker logs -f exp                            # log ao vivo (Ctrl+C só para de seguir; o treino continua)
docker ps -a | grep exp                       # status: "Up" = rodando | "Exited (0)" = terminou ok

cat results/logs/exp_*/SUMMARY.md             # tabela consolidada A vs C (val + diagnóstico train/val)
ls  results/logs/exp_*/                        # logs por etapa: train_A.log, eval_A_val.log, ...
ls  results/tables/finetuned_cenario_*.json    # as 4 métricas por cenário e split (train/val)
```

Os checkpoints (`results/checkpoints/`) e os logs/JSONs (`results/`) ficam no host
mesmo após remover o container (`docker rm exp`).

### Avaliação de checkpoint

```bash
# Avaliar um checkpoint no val split
docker compose -f docker-compose.finetuning.yml run --rm train \
    python scripts/evaluate.py \
    --checkpoint results/checkpoints/cenario_C/best_PCK.pth \
    --split val

# Avaliar todos os cenários treinados
for CENARIO in A B C D; do
  docker compose -f docker-compose.finetuning.yml run --rm train \
    python scripts/evaluate.py \
    --checkpoint results/checkpoints/cenario_${CENARIO}/best_PCK.pth \
    --split val
done
```

Checkpoints salvos em `results/checkpoints/cenario_{A,B,C,D}/best_PCK.pth`.
Métricas salvas em `results/tables/finetuned_cenario_{A,B,C,D}_val.json`.

### Resultados preliminares (Cenários A e C)

| Modelo | PDJ@0.5 | PCK@0.2 | OKS | MPJPE-2D |
|---|---|---|---|---|
| RTMPose-X zero-shot | 93,6% | 41,8% | 81,8% | 4,81 px |
| **Cenário A** (from scratch, val) | 90,7% | 47,0% | 80,1% | 5,34 px |
| **Cenário C** (TL fase 2, val) | **95,2%** | **61,5%** | **87,3%** | **3,63 px** |

Relatório completo: [docs/finetuning/relatorio-preliminar.md](docs/finetuning/relatorio-preliminar.md)
