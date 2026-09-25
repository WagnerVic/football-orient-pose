# Estimação de Pose de Jogadores de Futebol em Vídeo de Transmissão

> **Um Pipeline Quantitativo para Estimação de Pose de Jogadores de Futebol em Vídeo de Transmissão:**
> Seleção de Detector, Benchmark de Estimadores e Adaptação ao Domínio

**Autores:** Wagner Victor Alves de Menezes (202403929) · Victor Gabriel Ribeiro Jacome (202403926) ·
Raphael Alves de Lima Soares (202403922) · André Guilherme Alves do Carmo (202301423)

Bacharelado em Inteligência Artificial — Instituto de Informática, **Universidade Federal de Goiás (UFG)**
Disciplina: **Visão Computacional** — Prof. Ricardo Augusto Pereira Franco

![Pipeline em vídeo real — pose de todos os jogadores](results/showcase/gifs/all_players/brazil_01.gif)

> _O pipeline aplicado a uma transmissão real (Seleção Brasileira): detecção + estimação de pose de
> todos os jogadores, frame a frame, de forma 100% automática._

---

## 📌 Para o professor — guia de avaliação

Este repositório contém **um trabalho de Visão Computacional completo**: um pipeline que, a partir de
**vídeo de transmissão (broadcast)**, estima a **pose 2D** dos jogadores. A contribuição central é
**metodológica e quantitativa** — cada estágio do pipeline (detector e estimador) é **escolhido por
experimento**, e o estimador é **adaptado ao domínio por fine-tuning**, fornecendo a avaliação
numérica que faltava a esse problema.

### Por onde começar (ordem de leitura sugerida)

1. **O artigo** → [docs/artigo/artigo-vc.md](docs/artigo/artigo-vc.md) — **a entrega principal**.
   Contém problema, objetivos, metodologia, resultados e conclusão. _(Versão LaTeX/SBC em
   [Template_SBC/template-latex/sbc-template.tex](Template_SBC/template-latex/sbc-template.tex).)_
2. **Este README** — mapa do repositório: o que é cada pasta e como reproduzir os resultados.
3. **Os relatórios técnicos** → [docs/](docs/README.md) — o detalhe por trás de cada número do artigo
   (comparação de detectores, benchmark de estimadores, matriz de fine-tuning).

### Onde está cada entrega

| Entrega | Onde |
|---|---|
| **Artigo** (texto) | [docs/artigo/artigo-vc.md](docs/artigo/artigo-vc.md) + LaTeX em [Template_SBC/template-latex/](Template_SBC/template-latex/) |
| **Código** | [src/football_orient_pose/](src/football_orient_pose/) (biblioteca) + [scripts/](scripts/) (CLIs) |
| **Resultados e figuras** | [results/](results/) (tabelas, showcase, GIFs) + relatórios em [docs/](docs/README.md) |
| **Documentação técnica** | [docs/README.md](docs/README.md) (índice de tudo) |

---

## 🎯 Resultados principais

O trabalho decide cada estágio do pipeline por experimento. Os números-chave:

**1. Seleção do detector** — 4 detectores comparados (2 one-stage, 2 two-stage) com anotação humana.
[docs/vision/epic-113-detectores.md](docs/vision/epic-113-detectores.md)

| Detector | mAP | Recall | Precision |
|---|---|---|---|
| **YOLO26x** (escolhido) | **84,4** | **95,4%** | **98,7%** |
| Cascade R-CNN | 68,3 | 93,4% | 54,5% |
| Faster R-CNN | 65,1 | 94,7% | 47,8% |
| RetinaNet | 61,9 | 93,7% | 57,3% |

**2. Benchmark de estimadores (zero-shot, dataset 3DSP)** — RTMPose-X vence OpenPose e HRNet-W48.
[docs/vision/baseline-rtmpose-zero-shot.md](docs/vision/baseline-rtmpose-zero-shot.md)
O diagnóstico que motiva o resto do trabalho: o estimador **detecta bem a região** (PDJ@0,5 ≈ 93%) mas
**localiza mal o keypoint exato** (PCK@0,2 baixo) — gargalo de **precisão de localização**.

**3. Adaptação ao domínio (fine-tuning, matriz 2×2)** — Transfer Learning × Augmentation.
[docs/finetuning/epico-2/epic2-relatorio-final.md](docs/finetuning/epico-2/epic2-relatorio-final.md)

> **PCK@0,2: 41,8% (zero-shot) → 67,5% (RTMPose-X fine-tunado)** — a receita campeã combina
> _transfer learning_ + _augmentation geométrica_, com **quase nenhum overfitting** (gap treino→val
> de 1,1 pp).

**Conclusão central:** o gargalo do domínio é **adaptação ao domínio, não capacidade do modelo** — e
o ganho é **barato** (fine-tuning leve sobre pesos COCO).

---

## 🧩 O pipeline em 5 estágios

```
vídeo → [1] detecção → [2] crop justo → [3] estimação de pose → [4] reprojeção → frame anotado
```

| # | Estágio | O que faz | Código |
|---|---|---|---|
| 1 | **Detecção** | YOLO26x localiza cada jogador (caixas justas) | [detection.py](src/football_orient_pose/detection.py) |
| 2 | **Crop justo** | recorta cada jogador com _letterbox_ (sem distorção) | [crop.py](src/football_orient_pose/crop.py) |
| 3 | **Pose** | RTMPose-X (fine-tunado) estima os keypoints no crop | [estimators/rtmpose.py](src/football_orient_pose/estimators/rtmpose.py) |
| 4 | **Reprojeção** | devolve o esqueleto às coordenadas do frame original | [pipeline.py](src/football_orient_pose/pipeline.py) |

A função [`pose_all()`](src/football_orient_pose/pipeline.py) aplica o pipeline a **todos** os
jogadores detectados de um frame (foi o que gerou o GIF do topo).

---

## 🗂️ Mapa do repositório

| Pasta / arquivo | O que é | Por que importa para a avaliação |
|---|---|---|
| [docs/artigo/](docs/artigo/) | o artigo (Markdown) | **entrega principal** — leia primeiro |
| [Template_SBC/template-latex/](Template_SBC/template-latex/) | o artigo no template SBC (LaTeX) | versão final/formatada do artigo |
| [docs/](docs/README.md) | relatórios técnicos por projeto (visão + fine-tuning) + backlog | a fundamentação de cada número do artigo |
| [src/football_orient_pose/](src/football_orient_pose/) | a biblioteca: `detection`, `crop`, `pipeline`, `pose` + subpacotes `estimators/`, `evaluation/`, `finetuning/`, `utils/` | o coração do código |
| [scripts/](scripts/) | CLIs por etapa: `clips/`, `evaluation/`, `pipeline/`, `setup/`, `training/` | como cada experimento foi rodado |
| [results/](results/) | `tables/` (métricas), `showcase/` (frames + GIFs), `detection_viz/`, `checkpoints/`, `training_runs/` | os resultados gerados |
| [tests/](tests/) | suíte de testes unitários (`pytest`) | qualidade/reprodutibilidade do código |
| [notebooks/](notebooks/) | validação interativa dos estimadores | exploração e checagem |
| [configs/](configs/) | `split.json` (split oficial 80/20 por clip, seed 42) | reprodutibilidade do split |
| [Makefile](Makefile) | atalhos para setup, treino, avaliação e showcase | ponto único de entrada dos comandos |

---

## ▶️ Como reproduzir

> Ambiente gerenciado com **[uv](https://github.com/astral-sh/uv)**. Instale as dependências com
> `uv sync`. Os alvos abaixo estão todos no [Makefile](Makefile) (`make help` lista todos).

**1. Dados e pesos**
```bash
make setup                              # descompacta o dataset 3DSP para data/
bash scripts/setup/download_models.sh   # baixa os pesos dos modelos
cp .env.example .env                    # ajuste DATA_DIR se necessário
```

**2. Avaliar (estimadores e detector)**
```bash
make evaluate CKPT=<checkpoint.pth>     # avalia um estimador no split val (PCK/PDJ/OKS/MPJPE)
make docker-eval-detector               # avalia o detector
make docker-detectors-table             # gera a tabela comparativa de detectores
```

**3. Fine-tuning (matriz 2×2)** — detalhe em [docs/finetuning/epico-1/guia.md](docs/finetuning/epico-1/guia.md)
```bash
make train-a    # From Scratch, sem augmentation (baseline)
make train-b    # From Scratch, com augmentation
make train-c    # Transfer Learning, sem augmentation
make train-d    # Transfer Learning, com augmentation  ← receita campeã (67,5%)
```

**4. Pipeline em vídeo real (o showcase)**
```bash
make pose-all-brazil    # roda o pipeline completo nos clips da Seleção Brasileira
make gifs               # monta os GIFs animados a partir dos frames
```

**5. Testes**
```bash
uv run pytest           # suíte unitária em tests/
```

---

## 🔭 Escopo e decisões

- **Orientação corporal:** fora do escopo desta entrega — registrada como **trabalho futuro** no
  artigo. O pipeline atual entrega a **pose 2D**, base necessária para a orientação.
- **Anotação de keypoints em vídeo real:** em **backlog** (ver
  [docs/backlog/README.md](docs/backlog/README.md)) — sem _ground truth_ anotado nos clips reais, as
  métricas de pose ficam restritas ao dataset 3DSP; o showcase em vídeo real é, por ora, **qualitativo**.
- **Relação com o fine-tuning (projeto RNP):** o fine-tuning do RTMPose-X é a **etapa de adaptação ao
  domínio** deste mesmo pipeline — por isso integra a contribuição do artigo. Os relatórios completos
  da matriz experimental estão em [docs/finetuning/](docs/finetuning/).

---

## 📚 Documentação completa

Índice de todos os relatórios técnicos (visão, fine-tuning e backlog): **[docs/README.md](docs/README.md)**.
