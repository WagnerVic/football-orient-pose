# Documentação

A documentação é organizada **por projeto** (há dois projetos no mesmo repositório).

## `vision/` — Projeto de Visão (Football Orient Pose)
O diferencial de visão: do frame bruto à pose desenhada, benchmarkado e com métrica.

**Pipeline ponta-a-ponta (comece por aqui):**

| Arquivo | O que é |
|---|---|
| [`vision/epic-126-pipeline.md`](vision/epic-126-pipeline.md) | ⭐ **Relatório do pipeline** detect→crop→pose, showcase "todos os jogadores" (replica o Reis) + GIFs do Brasil |
| [`vision/epic-113-detectores.md`](vision/epic-113-detectores.md) | Benchmark de 4 detectores (mAP/AP via pycocotools) — **YOLO26x vence** |
| [`vision/formato-clips.md`](vision/formato-clips.md) · [`vision/formato-crops.md`](vision/formato-crops.md) | Specs dos estágios `data/clips/` (frames inteiros) e `data/crops/` (crop justo) |
| [`vision/handoff-extracao-clips.md`](vision/handoff-extracao-clips.md) | Handoff da extração de clips reais (examples + Brasil) |

**Estimadores e baseline (Épico 2):**

| Arquivo | O que é |
|---|---|
| [`vision/baseline-rtmpose-zero-shot.md`](vision/baseline-rtmpose-zero-shot.md) | Baseline zero-shot RTMPose-X no 3DSP (fonte canônica: PDJ 93,6% / PCK 41,8%) |
| [`vision/epic2-entrega-final.md`](vision/epic2-entrega-final.md) | Entrega final do Épico 2 (estimadores, 3 modelos) |
| [`vision/epic2-retrospectiva.md`](vision/epic2-retrospectiva.md) | Retrospectiva técnica do Épico 2 |
| [`backlog/`](backlog/) | Trabalho adiado (ex.: experimento crop justo×frouxo — depende de keypoint GT) |
| [`vision/_planning/`](vision/_planning/) | Artefatos de processo (plans/specs) — histórico |

## `finetuning/` — Projeto Transfer Learning (RNP)
Fine-tuning do RTMPose-X na matriz experimental 2×2 (Épico 1 do projeto RNP).

| Arquivo | O que é |
|---|---|
| [`finetuning/epico-1/visao-geral.md`](finetuning/epico-1/visao-geral.md) | **Comece por aqui** — panorama completo do que foi feito (infra + execução + resultados + pendências) |
| [`finetuning/epico-1/guia.md`](finetuning/epico-1/guia.md) | Guia de setup e execução do pipeline de fine-tuning (ambiente, Docker, comandos) |
| [`finetuning/epico-1/epic1-relatorio-preliminar.md`](finetuning/epico-1/epic1-relatorio-preliminar.md) | Relatório preliminar (pré-fixes do review) — Cenários A e C |
| [`finetuning/epico-2/epic2-relatorio-final.md`](finetuning/epico-2/epic2-relatorio-final.md) | ⭐ **Relatório FINAL do Épico 2 (canônico)** — matriz 2×2 completa (10 modelos); funde A/C + B/D. Receita campeã: TL + aug geométrica (D-FULL 67,5% · extremidades resolvidas) |
| [`finetuning/epico-2/epic2-relatorio-a-c.md`](finetuning/epico-2/epic2-relatorio-a-c.md) | *(histórico/detalhe)* parcial A/C — A/C/RAW + ablações C2 e flip |
| [`finetuning/epico-2/epic2-relatorio-bd.md`](finetuning/epico-2/epic2-relatorio-bd.md) | *(histórico/detalhe)* B/D — cenários com augmentation, ladder fino do TL e atribuição por mecanismo |

> **Nota:** PDF/HTML de relatórios não são versionados (ver `.gitignore`) — gere sob demanda a partir do `.md` fonte.
