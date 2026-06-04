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
├── docs/
│   ├── epic2_retrospectiva.md      # Retrospectiva técnica do Épico 2
│   └── validation/
│       └── rtmpose_zero_shot.md    # Resultados RTMPose no 3DSP
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
com PDJ@0.5, PCK@0.2, OKS e MPJPE-2D. **Documentação completa:** [docs/finetuning.md](docs/finetuning.md).

```bash
make finetuning-env          # cria a venv dedicada .venv-mmpose (stack MMPose)
make finetuning-checkpoint   # baixa os pesos COCO (Transfer Learning)
make finetuning-smoke        # smoke test ponta-a-ponta (~30s)
make train-a                 # treina o Cenário A (from scratch, sem augmentation)
make help                    # lista todos os comandos
```
