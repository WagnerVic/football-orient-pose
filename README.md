# football-orient-pose
Football player pose estimation and body orientation analysis from broadcast video using YOLO11 + HRNet

## Setup

Os datasets estão compactados em `.zip` para versionamento no Git. Após clonar o repositório, execute:

```bash
make setup        # Descompacta os .zip para data/
make clean-data   # Remove data/ para re-extrair do zero
make help         # Lista os comandos disponíveis
```

### Estrutura após setup

```
football-orient-pose/
├── 3dsp.zip              # Dataset compactado (versionado no Git)
├── data/                  # Dados extraídos (ignorado pelo Git)
│   └── 3dsp/
│       ├── train/
│       │   └── 00001/
│       │       ├── img/       # Imagens dos jogadores
│       │       └── posture/   # Anotações de pose (JSON)
│       └── ...
├── Makefile
└── README.md
```
