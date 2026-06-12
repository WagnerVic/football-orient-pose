# Épico #113 — Comparação dos 4 Detectores de Pessoa

> **Status:** scaffold. As tabelas e a escolha do vencedor são **preenchidas após rodar os 4
> detectores na GPU** (`scripts/evaluation/eval_detectors.py`). Este documento fecha a US #117 /
> task #118 quando os números reais entrarem.

## Objetivo

O baseline (Reis et al., 2023) usou apenas YOLOv3, sem nenhuma métrica quantitativa de detecção.
O diferencial compara **4 detectores** (2 one-stage + 2 two-stage) contra **GT humano** e escolhe o
melhor para alimentar o crop do finalizador. Métrica = pycocotools (COCOeval).

## Protocolo

- **Dados:** 3 examples, 60 frames (1280×720), **740 caixas** de jogador anotadas no Roboflow
  (`data/annotations/examples_bbox/`). Nota: o `gt.txt` do 3DSP é pseudo-GT YOLO (enviesado) — por
  isso usamos GT humano.
- **Detectores:** `yolo26` (one-stage anchor-free), `retinanet` (one-stage), `faster-rcnn`
  (two-stage), `cascade-rcnn` (two-stage, mmdet). Todos pré-treinados COCO, classe person.
- **Métricas:** mAP@[.5:.95], AP50, AP75, AR por tamanho (small/medium/large), e precision/recall/F1
  no ponto de operação (conf ≥ 0.3, IoU ≥ 0.5).
- **Comando:** `python scripts/evaluation/eval_detectors.py --detector <nome> --device cuda`
  (cascade exige `--config` + `--checkpoint` do model zoo do mmdet).
- Tabela: `python scripts/evaluation/detectors_table.py`.

> ℹ️ **AR_large** tende a ser **N/A (—)** nestes examples: os jogadores no broadcast são pequenos
> (área < 96²px), então quase não há objetos "large" no critério COCO. O foco real é **AR_small**
> (jogador distante) e **AR_medium**.

## Resultados `[PREENCHER após rodar]`

<!-- colar a saída de detectors_table.py aqui -->

| Detector | mAP | AP50 | AP75 | AR_s | AR_m | AR_l | F1 |
|---|---|---|---|---|---|---|---|
| yolo26 | — | — | — | — | — | — | — |
| retinanet | — | — | — | — | — | — | — |
| faster-rcnn | — | — | — | — | — | — | — |
| cascade-rcnn | — | — | — | — | — | — | — |

## Análise `[PREENCHER]`

- One-stage vs two-stage: …
- Recall em jogadores pequenos/distantes (AR_small) — crítico para não perder o finalizador no
  fundo de campo: …
- Custo/velocidade × precisão: …

## Decisão `[PREENCHER]`

**Detector vencedor:** `___` — justificativa: …

Esse detector é o que vai gerar os crops do finalizador (examples + Brasil) para a anotação de
keypoints (US #109) e o pipeline (Épico #126).

## Figuras

`results/detections/<detector>_<clip>_<frame>.png` — GT (verde) × predição (vermelho), geradas com
`--viz N`.
