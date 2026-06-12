# Projeto Transfer Learning (RNP) — Visão Geral do que foi feito

> Documento-mapa para entender **tudo** que entrou no `main` via o PR **#92**
> ("Épico 1 — Infraestrutura de Fine-tuning"). Apesar do rótulo, o PR avançou em
> **três épicos** do projeto de finetuning. Aqui está o panorama, o código
> destrinchado, os resultados, o que o code review pegou e o que falta.

---

## 0. Em uma frase

Foi construído um **pipeline de fine-tuning do RTMPose-X** para o dataset de futebol
3DSP, e já foram rodados **2 dos 4 experimentos** da matriz 2×2 (Cenários A e C),
mostrando que **transfer learning melhora a precisão** (PCK@0.2: baseline 41,8% → C
**61,5%**) — com a ressalva de que o ganho se concentra no tronco e as extremidades
ainda são fracas.

---

## 1. A pergunta de pesquisa (contexto RNP)

O RTMPose zero-shot **detecta** bem os keypoints (PDJ@0.5 = 93,6%) mas **erra a posição
exata** no domínio de futebol broadcast (PCK@0.2 = 41,8%) — crops 100×100, motion blur,
poses atípicas. **Pergunta:** dá para maximizar o PCK (precisão) via fine-tuning?

**Desenho experimental (matriz 2×2)** definido com o professor — varia dois fatores:

| | Sem Augmentation | Com Augmentation |
|---|---|---|
| **From Scratch** (W₀ aleatório) | **Cenário A** | **Cenário B** |
| **Transfer Learning** (W₀ COCO) | **Cenário C** | **Cenário D** |

A comparação **A vs C** é o coração: prova (com experimento, não "fé") se os pesos COCO
agregam valor. Ordem de prioridade: A → C → D → B.

---

## 2. O que o PR #92 entregou, por épico

O PR fechou formalmente só issues do Épico 1 (#57–#70), mas o que entrou no `main`
cruza épicos:

| Épico | O que é | Arquivos principais |
|---|---|---|
| **1 — Infraestrutura** | Código que faz o pipeline rodar | `src/.../finetuning/dataset.py`, `configs/cenario_*.py`, `scripts/{train,evaluate,smoke_cenario_a}.py`, `src/.../estimators/rtmpose.py`, `src/.../utils/keypoint_mapping.py`, Docker/Makefile, `docs/finetuning/guia.md` |
| **2 — Execução** | Rodar os experimentos de verdade | `scripts/training/run_experiments.sh` + as rodadas de A e C (checkpoints, JSONs em `results/tables/`) |
| **3 — Análise** | Interpretar os resultados | `docs/finetuning/epic1-relatorio-preliminar.md` |

> Ou seja: o "Épico 1" smoke-test virou, na prática, um primeiro corte do projeto inteiro.

---

## 3. Épico 1 — A infraestrutura, peça por peça

### 3.1 O fluxo de ponta a ponta

```
3DSP (JSON + JPG 100×100)
   │  DSP3Dataset  (converte p/ protocolo MMPose)
   ▼
GetBBoxCenterScale + TopdownAffine(use_udp)   ← letterboxing 100×100 → 288×384
   ▼
RTMPose-X  (CSPNeXt-X backbone + RTMCCHead/SimCC)
   │  treino: KLDiscretLoss (SimCC), best checkpoint por PCK
   ▼
evaluate.py  (inferência no val → 4 métricas vs baseline)
```

### 3.2 `DSP3Dataset` — a ponte 3DSP → MMPose
[`src/football_orient_pose/finetuning/dataset.py`](../../src/football_orient_pose/finetuning/dataset.py)

- Herda `BaseDataset` (mmengine) e se registra no `DATASETS` do MMPose.
- Para cada clip do split, lê os 20 frames: imagem 100×100 + 17 keypoints H3WB + bbox.
- **Keypoints derivados [0,7,8,9]** (center_hips, center_body, center_shoulder, neck) são
  médias, não anotações — recebem `visibility=0`, então **não são supervisionados** no treino.
- **`bbox=[0,0,w,h]`** (w=h=100): o crop inteiro é a ROI. (O x,y do JSON é a posição no frame
  original; usá-lo jogaria a ROI pra fora e zeraria a loss — foi um bug corrigido.)
- **`flip_indices`** (espelhamento horizontal L↔R) propagado por amostra:
  `[0,4,5,6,1,2,3,7,8,9,10,14,15,16,11,12,13]` — conferido item a item no review.

### 3.3 Letterboxing (sem distorção)
O crop 1:1 (100×100) vira a entrada 3:4 do modelo (288×384) via `GetBBoxCenterScale` +
`TopdownAffine(use_udp=True)`. O MMPose ajusta o aspect ratio **expandindo** a caixa e
preenchendo com cinza (padding) — não estica a imagem. A afim **inversa** devolve os
keypoints no espaço do crop. (Detalhe: usa `padding=1.25` padrão, então a escala real é
~2,3× + margem, não os "2,88×/48px" idealizados — consistente entre treino e eval, que é
o que importa.)

### 3.4 Os 4 configs
[`configs/cenario_{a,b,c,d}.py`](../../configs/) — mesma espinha (codec SimCC 288×384,
CSPNeXt-X, RTMCCHead, `KLDiscretLoss beta=10`). O que muda:

| | `load_from` | `frozen_stages` | Augmentation |
|---|---|---|---|
| **A** | None (aleatório) | 0 | não |
| **B** | None | 0 | MotionBlur(3–9) + RandomErasing(0.3) |
| **C** | COCO | 4 (→ destrava por fase) | não |
| **D** | COCO | 4 (→ destrava por fase) | sim |

> A augmentation vive no `train_pipeline` do config (idioma do MMPose), **não** numa flag
> do dataset — decisão consciente (era a issue #59).

### 3.5 `train.py` — o orquestrador
[`scripts/training/train.py`](../../scripts/training/train.py)

- **A/B (from scratch):** uma fase só. LR uniforme 1e-3, `--epochs` (default 50).
- **C/D (transfer learning):** 3 fases de **progressive unfreezing** (descongela o backbone
  aos poucos):

| Fase | frozen_stages | LR head | LR backbone | Épocas (default) |
|---|---|---|---|---|
| 1 — só a cabeça | 4 (tudo congelado) | 1e-3 | — | 15 |
| 2 — destrava topo | 2 | 1e-4 | 1e-5 | 20 |
| 3 — **condicional** | 1 | 1e-4 | 1e-6 | 15 |

- **Fase 3 só roda se** `Δ PCK(fase2−fase1) > --delta-pck` (default 5pp), medido com o
  **PCK estrito** do artigo (via `run_evaluation`).
- O **`param_scheduler`** (warmup ~10% + cosseno até `eta_min`) é reconstruído por fase
  conforme o nº de épocas — ambos corrigidos no code review (ver §6).

### 3.6 `evaluate.py` — as 4 métricas
[`scripts/evaluation/evaluate.py`](../../scripts/evaluation/evaluate.py)

Carrega o checkpoint, roda inferência no split e calcula, com as funções do projeto
([`evaluation/metrics.py`](../../src/football_orient_pose/evaluation/metrics.py)):

| Métrica | Mede | Normalização |
|---|---|---|
| **PCK@0.2** (alvo) | precisão | `0.2 × ref` (largura ombros/quadris) — **estrito** |
| **PDJ@0.5** | detecção | `0.5 × torso` (mais frouxo) |
| **OKS** | similaridade tipo-COCO | sigmas por keypoint |
| **MPJPE-2D** | erro em pixels | absoluto |

- Antes de medir, **deriva os keypoints [0,7,8,9]** das predições (`derive_h3wb_centers`),
  igual o GT e o baseline fazem — senão o modelo seria penalizado em joints que nem treina.
- ⚠️ **Duas escalas de PCK que confundem:** o **treino** monitora `PCKAccuracy(norm='bbox')`
  (leniente, ~0,8–0,9), mas o número **do artigo** é o PCK estrito do `evaluate.py` (~0,4–0,6).
  São coisas diferentes — não compare os dois.

### 3.7 Outros componentes
- `RTMPoseEstimator.from_checkpoint()` — carrega modelo `.pth` treinado (saída H3WB direta).
- **Infra:** `Dockerfile.finetuning` (stack mmcv/mmpose/mmdet + pesos COCO), `Makefile`
  (`make train-a/-c`, `make finetuning-smoke`, `make evaluate`), `setup_mmpose_env.sh`
  (ambiente sem sudo via `uv`). Tudo em [`docs/finetuning/guia.md`](guia.md).

---

## 4. Épico 2 — O que foi efetivamente rodado

[`scripts/training/run_experiments.sh`](../../scripts/training/run_experiments.sh) encadeia treino + avaliação
(train e val) dos Cenários **A e C** num container só (fire-and-forget) e gera um `SUMMARY.md`.
Hardware: **RTX 4090** (Docker), madrugada de **03–04/jun/2026**. Cenários **B e D ainda não
foram rodados**.

---

## 5. Épico 3 — Resultados (preliminares)

Fonte: [`docs/finetuning/epic1-relatorio-preliminar.md`](epic1-relatorio-preliminar.md).

### 5.1 Comparação no val (800 frames)

| Modelo | PDJ@0.5 | PCK@0.2 | OKS | MPJPE-2D |
|---|---:|---:|---:|---:|
| RTMPose-X zero-shot (baseline) | 93,6% | 41,8% | 81,8% | 4,81 px |
| **A** — from scratch | 90,7% | 47,0% | 80,1% | 5,34 px |
| **C** — transfer learning (fase 2) | **95,2%** | **61,5%** | **87,3%** | **3,63 px** |

### 5.2 Overfitting (os "3 números": treino / val / baseline)

| Cenário | PCK train | PCK val | gap | leitura |
|---|---:|---:|---:|---|
| **A** | 83,9% | 47,0% | ~37 pp | 🔴 overfitting forte (3.200 imgs é pouco p/ from scratch) |
| **C** (fase 2) | 76,6% | 61,5% | ~15 pp | 🟢 generaliza bem (COCO age como regularizador) |

### 5.3 Por grupo anatômico (PCK@0.2, val) — a ressalva importante

O TL melhora muito o **tronco** mas as **extremidades regridem vs. o baseline**:

| Grupo | Baseline | C (TL) | Δ |
|---|---:|---:|---:|
| head | 50,4% | 81,4% | 🟢 +31 |
| shoulder | 30,4% | 73,5% | 🟢 +43 |
| hip | 22,8% | 63,6% | 🟢 +41 |
| **wrist** | 43,5% | 39,8% | 🔴 −3,7 |
| **knee** | 58,3% | 56,6% | 🔴 −1,7 |
| elbow | 50,8% | 50,6% | ≈ 0 |

> Por isso o MPJPE de A piora (extremidades com erro grande puxam a média). No C o MPJPE
> melhora no agregado (3,63), mas os membros continuam o ponto fraco. **Hipótese para B/D:
> a augmentation recupera as extremidades?**

### 5.4 A Fase 3 piorou (atenção)
A Fase 3 do progressive unfreezing **degradou** o C: 61,5% → 58,4% PCK. O relatório
recomenda **parar na Fase 2** quando a fase 3 não ajuda. ⚠️ Mas o código atual ainda não
faz isso automaticamente (ver §7).

---

## 6. Code review (#92) — o que foi pego e corrigido

Foram corrigidos no commit `cf9fa5f` (verificados no código):

| Sev | Item | Status |
|---|---|---|
| 🔴 | **A1** — LR scheduler não acompanhava as épocas (warmup/cosseno fixos) | ✅ corrigido (`_scheduler_override`) |
| 🔴 | **A2** — gating da fase 3 usava PCK leniente (escala errada) | ✅ corrigido (PCK estrito via `run_evaluation`) |
| 🟡 | **M2** — MotionBlur era 3–5 | ✅ virou 3–9 |
| 🟡 | **M3** — `--device` ignorado | ✅ removido |
| 🟢 | **B1–B6** — letterboxing doc, bbox w/h, DRY da derivação, `from_checkpoint` staticmethod, etc. | ✅ |

Decisões aceitas (não-bloqueantes): **M1** (augmentation no config, não flag), **M4** (best por
PCK leniente — documentado), **M5** (baseline já existe via pipeline antigo).
Review completo: [`.refs/RNP/CODE-REVIEW-PR92.md`](../../.refs/RNP/CODE-REVIEW-PR92.md).

---

## 7. Pontos em aberto / riscos

1. 🔴 **Checkpoint final do TL pega a última fase, não a melhor.** Se a fase 3 roda e piora
   (como aconteceu: 58,4 < 61,5), o `train.py` ainda copia a fase 3 como `best_PCK.pth`. O
   relatório escolheu a fase 2 na mão. Fix simples: `final = ckpt_f3 if pck_f3 > pck_f2 else ckpt_f2`.
2. 🟡 **Números só de A e C, 1 run cada, sem proveniência (commit/seed) registrada** — re-rodar
   com os fixes do review antes de fechar conclusões.
3. 🟡 **Métrica de seleção (best por PCK leniente) ≠ métrica-alvo (PCK estrito)** — pode escolher
   um checkpoint subótimo; aceito como follow-up.

---

## 8. Próximos passos

- [ ] Re-rodar **A** e **C** com os fixes (`cf9fa5f`: scheduler/gating) — números limpos.
- [ ] Rodar **Cenário B** (from scratch + aug) e **Cenário D** (TL + aug) → fechar a matriz 2×2.
- [ ] Corrigir o bug do checkpoint final da fase 3 (§7.1).
- [ ] Acompanhar o **breakdown por grupo** (não só o PCK global) — ver se B/D recuperam as extremidades.
- [ ] Consolidar a matriz final no relatório e alimentar o artigo RNP (Seções 3 e 4).

---

## Glossário rápido

- **RTMPose-X / SimCC:** modelo de pose top-down; SimCC prevê cada keypoint como 2 classificações (eixos X e Y).
- **H3WB-17:** formato de 17 keypoints do 3DSP; 4 deles ([0,7,8,9]) são médias calculadas.
- **From scratch vs Transfer Learning:** pesos iniciais aleatórios vs. pesos COCO.
- **Progressive unfreezing:** descongelar o backbone em fases, do topo para a base. Técnica de
  Howard & Ruder (2018, ULMFiT) — *gradual unfreezing* + *discriminative fine-tuning* (LR por camada).
  Fundamentação: Yosinski et al. (2014) (camadas baixas gerais), Kumar et al. (2022, LP-FT) (treinar a
  cabeça primeiro evita distorcer features), Lee et al. (2023, *surgical fine-tuning*) (tunar subconjunto
  > tunar tudo sob *distribution shift* — justifica a fase 3 condicional). PDFs em
  `.task-context/input/referencias/.refs/RNP/artigos/`.
- **Letterboxing:** redimensionar mantendo proporção, preenchendo o resto com padding (sem esticar).
- **PCK/PDJ/OKS/MPJPE:** precisão / detecção / similaridade-COCO / erro em pixels.
