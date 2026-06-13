# Épico #126 — Pipeline ponta-a-ponta (detecção → crop → pose)

Relatório completo do pipeline que costura todas as peças do diferencial numa única passagem: do
frame bruto do broadcast até a **pose desenhada no frame real**. É a vitrine qualitativa do trabalho —
o detector vencedor, o crop otimizado e o melhor modelo de pose, juntos.

---

## 1. Objetivo e contexto

O baseline (**Reis et al., 2023**) faz `YOLOv3 → crop → OpenPose` para estimar a pose de **TODOS os
jogadores** do campo (recorta cada caixa, roda OpenPose em cada uma e re-cola todos os esqueletos no
frame — Fig. 5 do paper), **sem métrica** e validado só visualmente. **Não existe "finalizador" no
Reis** — esse conceito vem da outra referência (AutoSoccerPose / dataset 3DSP, sobre *soccer shots*),
onde o *shooter* é o único jogador com keypoint GT.

Isso define **duas vistas** do mesmo pipeline (mesmas peças, alvos diferentes):

| Vista | Quem é poseado | Para quê | Anotação |
|---|---|---|---|
| **Showcase "todos"** (replica o Reis) | **todos** os jogadores | vitrine qualitativa | ❌ automático |
| **Vista do finalizador** | só o *shooter* | base da **métrica** (PDJ/PCK/OKS) | ✅ (3DSP / US #109) |

O **finalizador não é o foco conceitual** — é um **artefato de métrica** (único com GT). Por isso o
**Brasil**, que não tem anotação de finalizador, roda **100% automático** na vista "todos".

Nosso diferencial junta as três peças que construímos em épicos anteriores:

- **Detector vencedor** — YOLO26x (Épico #113, ver `epic-113-detectores.md`).
- **Crop justo** — `crop.py` (US #101), letterbox sem distorção.
- **Pose** — RTMPose-X, zero-shot **ou** o fine-tunado do Épico 2.

E entrega **três produtos**:
1. **Showcase "todos os jogadores"** — esqueletos de todos sobre o frame real (replica o Reis; §8).
2. **Vista do finalizador** — o esqueleto do *shooter* sobre o frame (figura-chave da métrica).
3. **Estágio formal `data/crops/`** — os crops justos do finalizador + parâmetros de reprojeção,
   insumo da futura anotação de keypoints (US #109).

Importante: o pipeline é **qualitativo** (não calcula PDJ/PCK/OKS). A métrica do crop justo×frouxo
depende de keypoint GT, que está no backlog (`docs/backlog/crop-justo-vs-dataset-3dsp.md`).

---

## 2. Arquitetura

```
frame inteiro (1280×720)
      │
      ▼  [1] DETECTAR        YOLO26x → caixas de todas as pessoas (xyxy + conf)
      │
      ▼  [2] FINALIZADOR     casa a caixa do shooter (3DSP gt.txt) + rastreia no clip
      │
      ▼  [3] CROP JUSTO      make_crop(frame, caixa, "tight") → 100×100 + CropParams
      │
      ▼  [4] POSE            estimador → 17 keypoints H3WB (espaço do crop)
      │
      ▼  [5] REPROJETAR      crop_to_frame(keypoints) → desenha esqueleto no frame e no crop
```

O `pipeline.py` é **agnóstico** ao detector e ao estimador — recebe qualquer `Detector`/
`BasePoseEstimator`, então roda local (zero-shot) ou no Docker (fine-tunado/MMPose) sem mudar nada.

---

## 3. Componentes (estágio a estágio)

### 3.1 Detecção
- `detection.py:YOLO26Detector` (peso `yolo26x.pt`, o vencedor) → `list[(xyxy, conf)]`.
- `detections_to_arrays` converte para `(N,4)` + `(N,)`.

### 3.2 Seleção e rastreamento do finalizador
Nos **examples** o finalizador é de graça: o 3DSP marca o `shooter_tracklet_id` e a caixa por frame
no `gt.txt`. Lemos isso com `data_io.load_finisher_boxes(clip)` → `{frame: xyxy}` (converte
xywh→xyxy, filtra a tracklet do shooter).

A seleção evoluiu em duas etapas (ver §6.1):
- `pipeline.select_finisher(boxes, ref_box)` — escolhe a detecção de **maior IoU** com a referência
  (heurística de maior área se não houver referência).
- `pipeline.track_finisher(boxes_por_frame, ref_boxes)` — **rastreia o clip inteiro**: ancora no
  frame de maior IoU com a referência e **propaga** seguindo a caixa anterior por IoU. Robusto a
  aglomerado (jogadores colados) e a desalinhamento de índice de frame.

### 3.3 Crop justo
- `crop.make_crop(frame, caixa, "tight")` → crop `100×100` (letterbox, preserva aspecto) +
  `CropParams` (origem, escala, padding) — a **chave de reprojeção** frame↔crop.

### 3.4 Pose — duas vias
- **Zero-shot:** `RTMPoseEstimator` (rtmlib/ONNX) — RTMPose-X **body7** pré-treinado, usado cru. Roda
  **local** (4050). Sai em COCO-17 → convertido para H3WB-17.
- **Fine-tunado:** `RTMPoseEstimator.from_checkpoint(...)` (MMPose) — o melhor do Épico 2 (**D-OCCL**,
  PCK 67%), que **parte do mesmo body7** e treina nos crops do 3DSP. Requer **Docker** (stack MMPose).
  Sai direto em H3WB-17.

Ambos passam por `predict_h3wb(crop) -> (17,2)`. Detalhe crítico: os keypoints derivados **0,7,8,9**
são recalculados (ver §6.3).

### 3.5 Reprojeção e visualização
- `crop.crop_to_frame(keypoints, params)` leva os keypoints do crop para o frame inteiro.
- `utils/viz.draw_skeleton(img, kps)` desenha ossos+juntas H3WB (`skeleton.H3WB_BONES`), coloridos por
  lateralidade (esquerda=verde, direita=laranja, centro=azul).
- `utils/viz.draw_boxes` marca todas as detecções (cinza) e o finalizador (vermelho).

---

## 4. Estágio formal `data/crops/`

O pipeline também **formaliza** o estágio com crop (spec em `docs/vision/formato-crops.md`):

```
data/crops/<fonte>/<clip_id>/
├── img/NNN.jpg            # crop justo 100×100 do finalizador
├── crop_params/NNN.json   # CropParams + finisher_bbox_xyxy + source_frame (chave de reprojeção)
└── info.ini               # metadados
```

Escrito por `crops_io.write_crop_clip`, relido por `crops_io.load_crop_params`. Versionado (crops são
pequenos). Quando a US #109 anotar keypoints, cria `posture/NNN.json` ao lado, e o GT projeta via
`crop.py`. Os 3 examples já estão gerados em `data/crops/examples/` (60 crops).

---

## 5. As duas vias de pose (resumo)

| | Zero-shot (RTMPose body7) | Fine-tunado (D-OCCL) |
|---|---|---|
| Backend | rtmlib / ONNX | MMPose |
| Onde roda | **local** (4050) | **Docker** (imagem do finetuning) |
| Precisão (val 3DSP) | PCK ~42% | **PCK ~67%** |
| Saída | COCO-17 → H3WB | H3WB direto |
| Robustez | generalista | igualmente robusto (ver §6.3) |

Os dois são o **mesmo RTMPose-X body7**; o fine-tunado é esse base + treino no 3DSP. O zero-shot é o
ponto de partida do transfer learning.

---

## 6. Bugs encontrados e corrigidos (a parte importante)

O pipeline funcionou de primeira no esqueleto, mas três problemas reais apareceram na validação. Vale
documentar porque a investigação rendeu aprendizado.

### 6.1 Troca de finalizador em aglomerado
**Sintoma:** nos frames 17/20 do `example_01`, o crop pegava o jogador **vizinho**, não o finalizador.
**Causa:** o match per-frame com a caixa do 3DSP é ambíguo quando dois jogadores estão colados (IoU
~0,4 pra ambos) → às vezes ganhava o vizinho. Não era offset constante.
**Correção:** `track_finisher` — ancora no frame de maior IoU e **propaga seguindo a caixa anterior**
(rastreamento), em vez de re-casar com a referência a cada frame. Passou a seguir o mesmo jogador.

### 6.2 Caminho do 3DSP `test` diferente por máquina
**Sintoma:** no SSH, "sem caixa de referência do 3DSP → heurística".
**Causa:** na 4050 o dataset fica em `data/test`; no SSH, em `data/3dsp/test`.
**Correção:** `_resolve_test_root` tenta os dois layouts automaticamente.

### 6.3 Esqueleto do fine-tunado embolado — os centros derivados (o bug grande)
**Sintoma:** o fine-tunado produzia um esqueleto **completamente quebrado** (linhas atravessando o
crop), enquanto o zero-shot ficava coerente.

**Hipótese errada (registrada por honestidade):** primeiro atribuí ao **crop** — supus que o
fine-tunado tinha superajustado ao crop **frouxo** do 3DSP e quebrava no nosso crop **justo** (fora da
distribuição). Implementei até o `pose_on_crops` pra rodar nos crops frouxos do dataset… **e quebrou
do mesmo jeito.** Isso **derrubou a hipótese**: se quebra até na distribuição de treino, **não é o
crop**.

**Causa real:** os keypoints **0, 7, 8, 9** (Center of Hips, Center of Body, Center of Shoulder, Neck)
são **derivados** (médias) e **não foram supervisionados no treino** — o modelo **prediz lixo** neles.
Como o esqueleto H3WB tem **muitos ossos passando por esses centros** (tronco, ombros, pescoço),
desenhar o lixo embola tudo, **em qualquer crop**. O `evaluate.py` do Épico 2 já tratava isso
(chamava `derive_h3wb_centers`), por isso lá dava PCK 67%; o zero-shot nunca sofreu porque o
`coco17_to_h3wb17` **calcula** os centros.
**Correção:** `RTMPoseFinetunedEstimator.predict_h3wb` agora chama `derive_h3wb_centers` — recalcula
os 4 centros das juntas reais.

**Lição (corrige a narrativa):** o fine-tunado **NÃO é sensível ao crop**. Funciona bem no justo e no
frouxo. O esqueleto quebrado era um **bug de pós-processamento** (centros não recalculados na
visualização), não uma limitação do modelo. → **Não** afirmar "fine-tunado só funciona no crop
frouxo": é falso.

---

## 7. Resultados (showcase)

Após as correções, validado de ponta a ponta na 4050 e no SSH:

- **Zero-shot + crop justo (4050):** esqueletos coerentes no finalizador, no frame e no crop.
- **Fine-tunado D-OCCL + crops do 3DSP (`make docker-pose-3dsp`):** coerente (distribuição de treino).
- **Fine-tunado D-OCCL + nosso crop justo (`make docker-pipeline-finetuned`):** **coerente** — o
  showcase definitivo: **YOLO26x → crop justo → melhor pose**, no frame real do broadcast.

As imagens ficam em `results/pipeline/<clip>/` e `results/pose_crops/<id>/` (gitignored — geradas,
pesadas; reproduzíveis rodando os comandos). Algumas figuras-chave podem ser versionadas em docs no
futuro.

---

## 8. Showcase "pose em todos os jogadores" (replica o Reis) — Brasil

A vista do finalizador depende de uma caixa de referência (3DSP) que **não existe no Brasil**. A
solução — e que **replica fielmente o baseline Reis** — é posear **todos** os jogadores detectados, sem
seleção. Assim o Brasil ganha um showcase **100% automático**, sem a marcação manual do finalizador
(#110 deixa de bloquear).

### 8.1 Como funciona
Por frame: `YOLO26x` detecta todas as pessoas → para **cada** caixa, `make_crop` (justo) → `predict_h3wb`
→ `crop_to_frame` (reprojeta) → `draw_skeleton` no frame. Todos os esqueletos são re-colados no frame
original (mesma ideia da Fig. 5 do Reis). O núcleo é `pipeline.pose_all(frame, boxes, pose,
min_box_height)`, que reusa `crop_and_pose` num loop sobre as caixas.

Decisões:
- **Estilo único, sem destacar ninguém** — fiel ao Reis (a vista do finalizador, com destaque, é a
  *vista da métrica*, separada).
- **Filtro `min_box_height` (40px)** — descarta jogador distante (crop minúsculo → esqueleto embolado;
  o próprio Reis nota que jogadores muito pequenos falham).
- **Crops não são persistidos** — são intermediários em memória (recorta → pose → desenha → descarta).
  O único produto salvo é **1 PNG compositado por frame** em `results/pose_all/<clip>/` (gitignored).
  Isso evita ~1300 imagenzinhas por jogador; nada disso vai pro git.
- **`data/crops/` (versionado) não é tocado** — continua só do finalizador (insumo da anotação).

### 8.2 Resultado
Validado de ponta a ponta nos 5 clips do Brasil (10 frames cada):
- **Zero-shot (local, 4050):** esqueletos coerentes em todos os jogadores no frame real.
- **Fine-tunado D-OCCL (Docker/SSH):** **também coerente** — e este é um **achado relevante**: a
  D-OCCL foi treinada nos crops do **3DSP** (outro time, outro broadcast, outro ângulo), e ainda assim
  **generalizou para o Brasil**. Mostra que o fine-tuning **não superajustou** à distribuição do 3DSP.

### 8.3 Fora de escopo (depois)
- **GIF/animação** juntando os frames de um clip (ex. o frame inicial com as detecções) — backlog.
- **Métrica no Brasil** — impossível (sem keypoint GT); a métrica fica no finalizador (3DSP/US #109).

---

## 9. Como rodar

**Zero-shot (local, 4050):**
```bash
python scripts/pipeline/demo_examples.py --pose rtmpose --device cuda
```

**Fine-tunado, pipeline completo (Docker/SSH) — crop justo:**
```bash
make docker-pipeline-finetuned \
  CKPT=results/runs/20260608_014649_bd/checkpoints/cenario_D-OCCL/best_PCK.pth
```

**Fine-tunado, só inferência nos crops do 3DSP (Docker/SSH):**
```bash
make docker-pose-3dsp \
  CKPT=results/runs/20260608_014649_bd/checkpoints/cenario_D-OCCL/best_PCK.pth
```

**Showcase "todos os jogadores" — Brasil (§8):**
```bash
# zero-shot, local (4050) — também serve p/ examples: ROOT=data/clips/examples
make pose-all-brazil

# fine-tunado D-OCCL, Docker/SSH
make docker-pose-all-brazil \
  CKPT=results/runs/20260608_014649_bd/checkpoints/cenario_D-OCCL/best_PCK.pth
```

Observações: o fine-tunado exige a imagem do finetuning (MMPose) e o checkpoint (vive em `results/`,
montado); o zero-shot baixa o ONNX na 1ª vez. O 3DSP test é auto-resolvido (`data/test` ou
`data/3dsp/test`).

---

## 10. Achados e lições

- **O crop não era o problema do fine-tunado** — eram os centros derivados não recalculados. O modelo
  é robusto ao enquadramento.
- **Rastreamento > match per-frame** para selecionar o finalizador em cenas aglomeradas.
- **Separar "predizer" de "avaliar"** e "detecção" de "pose" deixou o pipeline modular: dá pra trocar
  o estimador (zero-shot ↔ fine-tunado) sem tocar no resto.
- O **transfer learning** (fine-tunado) parte do **mesmo** modelo do zero-shot (RTMPose-X body7); a
  diferença é o treino no 3DSP, que rendeu +25pp de PCK.
- O **fine-tunado generaliza para o Brasil** (domínio diferente do 3DSP) na vista "todos" — não ficou
  preso à distribuição de treino.
- **Reis = pose de todos; finalizador = artefato de métrica.** Confundir os dois leva a "focar no
  finalizador" como se fosse o objetivo — não é. A vista "todos" é a replicação fiel do baseline.

---

## 11. Estado: feito vs backlog

| Item | Status |
|---|---|
| Pipeline detect→crop→pose | ✅ feito, funcionando (zero-shot e fine-tunado) |
| Estágio `data/crops/` formal | ✅ feito (3 examples gerados) |
| Vista do finalizador (frame + esqueleto) | ✅ feito |
| Showcase "todos os jogadores" (replica o Reis) | ✅ feito |
| Pipeline no **Brasil** (#130) — showcase "todos" | ✅ feito (zero-shot e fine-tunado, automático) |
| Métrica do crop justo×frouxo (#119) | ⬜ backlog (precisa keypoint GT — US #109) |
| GIF/animação dos frames | ⬜ backlog |

---

## 12. Arquivos e issues

**Núcleo:** `pipeline.py` (run_pipeline, crop_and_pose, select_finisher, track_finisher, **pose_all**),
`crops_io.py`, `utils/viz.py:draw_skeleton`, `utils/data_io.py:load_finisher_boxes`,
`estimators/rtmpose.py` (fix dos centros).
**Scripts:** `scripts/pipeline/demo_examples.py`, `scripts/pipeline/pose_on_crops.py`,
`scripts/pipeline/pose_all_players.py` (showcase "todos").
**Docker/Make:** `docker-pipeline-finetuned`, `docker-pose-3dsp`, `pose-all-brazil`,
`docker-pose-all-brazil`.
**Docs:** `docs/vision/formato-crops.md`, este relatório.
**Reúso:** `detection.py`, `crop.py`, `pose.py`/`estimators`, `utils/skeleton.py`,
`utils/keypoint_mapping.py`, `evaluation/detection_metrics.py:iou_matrix`.

**Issues:** Épico #126 (US #127 / task #128). Gera o insumo da US #109. Brasil = #130 (showcase feito).
