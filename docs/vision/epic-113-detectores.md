# Épico #113 — Comparação dos Detectores de Pessoa

> **Status:** concluído. Vencedor decidido (US #117 / task #118): **YOLO26x**.

## 1. Objetivo

O baseline (Reis et al., 2023) usou apenas YOLOv3, **sem nenhuma métrica quantitativa** de detecção.
O diferencial deste trabalho compara detectores **one-stage** e **two-stage** contra **ground-truth
humano** e escolhe o melhor para alimentar o crop do finalizador (Épicos #119 e #126).

## 2. Protocolo

- **Dados:** 3 clips de `data/clips/examples`, **60 frames** (1280×720), **740 caixas** de jogador
  anotadas à mão no Roboflow (`data/annotations/examples_bbox/`). O `gt.txt` do 3DSP é pseudo-GT
  gerado por YOLO (enviesado) — por isso usamos anotação humana.
- **Métricas:** `pycocotools` (COCOeval) → mAP@[.5:.95], AP50, AP75, AR por tamanho; e
  precision/recall/F1 no **ponto de operação** (conf ≥ 0,3, IoU ≥ 0,5).
- **Detectores avaliados:**
  | Detector | Tipo | Backbone / peso | Tamanho |
  |---|---|---|---|
  | YOLO26x | one-stage anchor-free | `yolo26x.pt` (ultralytics) | grande |
  | RetinaNet | one-stage | ResNet50-FPN (torchvision, COCO_V1) | ~130 MB |
  | Faster R-CNN | two-stage | ResNet50-FPN (torchvision, COCO_V1) | ~160 MB |
  | Cascade R-CNN | two-stage (cascata de IoU) | ResNet50-FPN (mmdet, 1x_coco) | ~265 MB |

  Conjunto **completo do desafio: 2 one-stage (YOLO26x, RetinaNet) + 2 two-stage
  (Faster R-CNN, Cascade R-CNN)**.
- **Reprodução:** `python scripts/evaluation/eval_detectors.py --detector <nome> --device cuda`
  (`--weights yolo26x.pt` para a variante grande do YOLO); tabela via
  `scripts/evaluation/detectors_table.py`. Rodado numa RTX 4050 (6 GB) — é inferência leve.

### Distribuição de tamanho do GT (contextualiza o AR)

| Tamanho (COCO) | Caixas | % | Observação |
|---|---|---|---|
| small (<32²px) | 5 | 0,7% | ruído estatístico (n=5) |
| **medium** (32²–96²px) | **735** | **99,3%** | jogador típico ~60×60px → **métrica relevante** |
| large (>96²px) | 0 | 0% | inexistente → COCOeval reporta `AR_large = n/a` |

Por isso, nas tabelas, **`AR_medium` é o número de recall-por-tamanho que importa**; `AR_small` é
ruído e `AR_large` é N/A.

## 3. Resultados

| Detector | mAP | AP50 | AP75 | AR_med | Precision | Recall | F1 | nº predições |
|---|---|---|---|---|---|---|---|---|
| **YOLO26x** | **84,4** | **95,0** | **91,6** | **87,5** | **98,7** | **95,4** | **97,0** | 715 |
| Cascade R-CNN | 68,3 | 90,7 | 77,5 | 73,8 | 54,5 | 93,4 | 68,9 | 1267 |
| Faster R-CNN | 65,1 | 91,9 | 74,7 | 70,3 | 47,8 | 94,7 | 63,6 | 1466 |
| RetinaNet | 61,9 | 91,1 | 70,3 | 68,2 | 57,3 | 93,7 | 71,1 | 1210 |

*(GT = 740 caixas; valores em %.)*

### Comentário por detector

**YOLO26x — vencedor.** Domina **todas** as métricas. O traço mais notável: é o único **alto em
recall E precision simultaneamente** (95,4% / 98,7%). Previu **715 caixas ≈ as 740 do GT**, ou seja,
acha os jogadores **e ignora a torcida** — não precisa de pós-filtro. O AP75 de **91,6%** indica
caixas muito justas, o que se traduz diretamente em **crops melhores** para a estimação de pose a
jusante. Recall de 95,4% (o maior) significa que é o que **menos perde o finalizador**.

**Faster R-CNN — 2º em qualidade, mas "sujo".** Bom mAP (65,1) e recall altíssimo (94,7%), mas
**precision de 47,8%**: cuspiu **1466 caixas** (~2× o GT), boa parte na **torcida/banco** (pessoas
reais, não anotadas como jogador, contadas como falso-positivo). O AP75 (74,7) é bem inferior ao do
YOLO26x → caixas mais frouxas. Os falsos-positivos seriam filtráveis na seleção do finalizador, mas
ainda assim fica atrás em localização.

**Cascade R-CNN — o melhor dos two-stage.** mAP 68,3 (o maior entre os two-stage) e, em especial,
**AP75 de 77,5%** — superior ao Faster (74,7) e ao RetinaNet (70,3). Isso confirma o efeito da
**cascata de IoU crescente**: refinando a caixa em estágios, ele **localiza mais justo** que os
demais two-stage. Ainda assim sofre do mesmo mal: cuspiu **1267 caixas** (torcida) → precision 54,5%.
Conclusão: melhora a localização entre os two-stage, mas continua **muito atrás do YOLO26x** em todas
as frentes.

**RetinaNet — equilibrado, porém atrás.** Perfil parecido com o Faster (mesmo backbone): recall alto
(93,7%), precision intermediária (57,3%), mAP 61,9. Não lidera em nenhuma métrica.

## 4. A jornada metodológica: por que quase escolhemos errado

A **primeira rodada** usou o `yolo26n` (**nano**, ~5 MB) e deu um resultado **enganoso**:

| Métrica | yolo26**n** (nano) | yolo26**x** (grande) |
|---|---|---|
| mAP | 51,0 | **84,4** |
| AP75 | 58,3 | **91,6** |
| Recall | 73,8 | **95,4** |
| Precision | 89,5 | **98,7** |
| F1 | 80,9 | **97,0** |

Com o nano, o YOLO ficava **atrás** do Faster/RetinaNet, e cheguei a recomendar o Faster R-CNN. O
problema: comparar `yolo26n` (5 MB) com **ResNet50-FPN (~160 MB)** é uma comparação de **pesos-pena
diferentes** — injusta. O nano tinha confiança baixa em jogadores **ocluídos/colados** (situação
típica de finalização), perdendo-os abaixo do limiar de confiança.

Ao **igualar a capacidade** (`yolo26x`), o quadro **se inverteu completamente**. **Lição:** a
capacidade do modelo precisa ser comparável para a comparação ser válida — um achado metodológico
relevante para o artigo.

## 5. Análise transversal

- **Precision é contaminada nos two-stage, não no YOLO26x.** O GT marca só jogadores em campo; os
  two-stage detectam a torcida → falso-positivo artificial. O YOLO26x, calibrado, não cai nessa
  (precision 98,7% com recall 95,4%).
- **Recall é o critério crítico** (imune à torcida) — "não perder o finalizador". YOLO26x lidera.
- **AP75 importa para o downstream** (caixa justa → crop justo → melhor pose). YOLO26x lidera com
  folga (91,6 vs 74,7).
- **One-stage moderno × two-stage:** um one-stage anchor-free de capacidade adequada (YOLO26x)
  supera os **dois** two-stage clássicos em **velocidade e qualidade** simultaneamente.
- **Entre os two-stage, o Cascade lidera** (mAP 68,3, AP75 77,5): a cascata de IoU melhora a
  localização vs Faster/RetinaNet — mas não o suficiente para ameaçar o YOLO26x.

## 6. Decisão (US #117 / #118)

> ## ✅ Vencedor: **YOLO26x**
>
> Escolhido por **dominar todas as métricas**, contra **os 4 detectores** (2 one-stage + 2
> two-stage) — mAP (84,4), AP50 (95,0), AP75 (91,6), AR_medium (87,5), precision (98,7), recall
> (95,4) e F1 (97,0). Em especial: **maior recall** (não perde o finalizador) + **maior AP75** (caixa
> justa → melhor crop) + **maior precision** (quase sem falso-positivo de torcida). Não é vitória "em
> um critério": é vitória **em todos**, sobre **todos**.

Esse detector gera os crops do finalizador nos **examples** (Exp. de crop, Épico #119) e nos clips
do **Brasil** (pipeline ponta-a-ponta, Épico #126), destravando a anotação de keypoints (US #109).

## 7. Limitações e trabalho futuro

- **AR_small** é ruído (apenas 5 caixas pequenas no GT). Para medir bem o recall em jogadores muito
  distantes seria preciso anotar mais frames com jogadas de campo aberto.
- A precision dos two-stage poderia ser "corrigida" com **filtro de região do campo** (ignorar
  detecções fora do gramado), mas isso não altera a conclusão (o YOLO26x já vence sem isso).
- O Cascade R-CNN rodou via Docker (`make docker-eval-cascade`) — `mmdet` só existe na imagem do
  finetuning, não no venv local. Os 3 demais rodam direto (torchvision/ultralytics auto-baixam pesos).

## 8. Figuras

Geradas com `--viz 3` (GT em verde × predição em vermelho), em `results/detections/`:
- `yolo26_<clip>_<frame>.png` — caixas justas, coladas nos jogadores, sem torcida.
- `faster-rcnn_<clip>_<frame>.png` — recall alto, porém com caixas na arquibancada (falso-positivo).

## 9. Arquivos

- Resultados por detector: `results/tables/detector_{yolo26,cascade-rcnn,faster-rcnn,retinanet}.json`
- Tabela LaTeX (artigo): `results/tables/detectores.tex`
- Código: `scripts/evaluation/eval_detectors.py`, `scripts/evaluation/detectors_table.py`,
  `scripts/evaluation/eval_cascade.sh`, `src/football_orient_pose/evaluation/detection_metrics.py`
