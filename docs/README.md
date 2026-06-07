# Documentação

A documentação é organizada **por projeto** (há dois projetos no mesmo repositório).

## `vision/` — Projeto de Visão (Football Orient Pose)
Pipeline base de estimação de pose: estimadores zero-shot e métricas.

| Arquivo | O que é |
|---|---|
| [`vision/baseline-rtmpose-zero-shot.md`](vision/baseline-rtmpose-zero-shot.md) | Resultados do baseline zero-shot RTMPose-X no 3DSP (fonte canônica: PDJ 93,6% / PCK 41,8%) |
| [`vision/epic2-entrega-final.md`](vision/epic2-entrega-final.md) | Entrega final do Épico 2 (estimadores, 3 modelos) |
| [`vision/epic2-retrospectiva.md`](vision/epic2-retrospectiva.md) | Retrospectiva técnica do Épico 2 |
| [`vision/_planning/`](vision/_planning/) | Artefatos de processo (plans/specs do workflow Superpowers) — histórico |

## `finetuning/` — Projeto Transfer Learning (RNP)
Fine-tuning do RTMPose-X na matriz experimental 2×2 (Épico 1 do projeto RNP).

| Arquivo | O que é |
|---|---|
| [`finetuning/guia.md`](finetuning/guia.md) | Guia de setup e execução do pipeline de fine-tuning (ambiente, Docker, comandos) |
| [`finetuning/relatorio-preliminar.md`](finetuning/relatorio-preliminar.md) | Relatório de resultados preliminares do Épico 1 (Cenários A e C) |

> **Nota:** PDF/HTML de relatórios não são versionados (ver `.gitignore`) — gere sob demanda a partir do `.md` fonte.
