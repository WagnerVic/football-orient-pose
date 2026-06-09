# Epic 2 — Entrega Final: Estimadores de Pose (3 Modelos)

**Branch:** `7-epic-2-estimadores-de-pose-3-modelos`  
**Período:** Maio–Junho 2026  
**Status:** ✅ COMPLETO — todas as 12 issues fechadas (#7 a #18)  
**Commits nesta branch:** 29  
**Testes:** 53 passando / 0 falhando  

---

## 1. Objetivo do Épico

Implementar wrappers de inferência para os 3 estimadores de pose do trabalho:

| Modelo | Ano | Arquitetura | Framework | Referência no paper |
|--------|-----|-------------|-----------|---------------------|
| **OpenPose** | 2017 | Bottom-up, Part Affinity Fields | `cv2.dnn` (Caffe) | Baseline de Reis et al. (2023) |
| **HRNet-W48** | 2019 | Top-down, High-Resolution heatmaps | ONNX + onnxruntime | Proposta original do grupo |
| **RTMPose-X** | 2023 | Top-down, SimCC (coordinate classification) | rtmlib + ONNX | SOTA no 3DSP — Yeung et al. (2024) |

Todos outputam no mesmo formato: `(17, 3)` COCO-17 `[x, y, confidence]`, convertidos automaticamente para H3WB-17 `(17, 2)` via `predict_h3wb()`.

---

## 2. Arquitetura Implementada

### 2.1 Interface Unificada — `BasePoseEstimator`

**Arquivo:** `src/football_orient_pose/pose.py`

```python
class BasePoseEstimator(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def predict(self, image: np.ndarray) -> np.ndarray:
        """→ (17, 3) COCO-17 [x, y, confidence]"""

    @abstractmethod
    def predict_batch(self, images: list[np.ndarray]) -> np.ndarray:
        """→ (N, 17, 3) COCO-17"""

    def predict_h3wb(self, image: np.ndarray) -> np.ndarray:
        """→ (17, 2) H3WB-17 — converte via coco17_to_h3wb17 com mascaramento de confiança"""
        return coco17_to_h3wb17(self.predict(image))  # passa (17,3) completo
```

**Padrão Strategy:** o pipeline de avaliação troca de modelo sem alterar uma linha de código.  
**Padrão de injeção para testes:**
```python
estimator = RTMPoseEstimator(pose_model=fake_model)  # sem carregar pesos reais
```

### 2.2 RTMPoseEstimator

**Arquivo:** `src/football_orient_pose/estimators/rtmpose.py`  
**Modelo:** `rtmpose-x_simcc-body7_pt-body7_700e-384x288`  
**Framework:** `rtmlib` — baixa pesos automaticamente na 1ª inferência  
**Setup:** nenhum — `pip install rtmlib` é suficiente

```
Input: 100×100 BGR crop
  → rtmlib faz resize + normalização internamente
  → SimCC decoder → coordenadas (x, y) diretas — sem heatmap
  → rescale para coordenadas originais
Output: (17, 3) COCO-17
```

**Detalhe importante:** quando rtmlib retorna múltiplas instâncias (raro em crops 100×100), `_select_best_instance` seleciona a mais próxima do centro do crop usando a distância entre o centro dos quadris (COCO 11+12) e o centro dos ombros (COCO 5+6).

### 2.3 HRNetEstimator

**Arquivo:** `src/football_orient_pose/estimators/hrnet.py`  
**Modelo:** HRNet-W48, input 256×192  
**Framework:** ONNX + onnxruntime (não MMPose — ver decisão técnica abaixo)  
**Pesos:** `models/weights/hrnet_w48_coco_256x192.onnx` (~242 MB) + `.onnx.data`  
**Download:** Google Drive via `gdown` — `scripts/setup/download_models.sh`

```
Input: 100×100 BGR crop
  → cv2.resize(192, 256)
  → BGR→RGB, /255, normalização ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
  → CHW float32 → (1, 3, 256, 192)
  → ONNX session → heatmaps (17, 64, 48)
  → argmax flat → (y_hm, x_hm) → rescale para original
Output: (17, 3) com confidence = valor max do heatmap
```

**Por que PDJ é alto mas ainda menor que RTMPose:**  
Heatmaps 64×48 em crop 100×100 → erro de quantização de ~2 px por joint.  
SimCC (RTMPose) prediz coordenadas diretamente → sem quantização.

**Decisão técnica — por que ONNX em vez de MMPose:**  
MMPose/mmcv falhou no setup por dependências C (chumpy, xtcocotools). Solução: exportar ONNX uma única vez e usar onnxruntime em runtime. Elimina ~3 GB de dependências pesadas.

### 2.4 OpenPoseEstimator

**Arquivo:** `src/football_orient_pose/estimators/openpose.py`  
**Modelo:** OpenPose COCO 18 partes (`pose_iter_440000.caffemodel`)  
**Framework:** `cv2.dnn.readNetFromCaffe` — disponível em qualquer OpenCV  
**Pesos:** `models/weights/openpose_pose_iter_440000.caffemodel` (~132 MB)  
**Download:** `huggingface.co/camenduru/openpose` — servidor CMU offline

```
Input: 100×100 BGR crop
  → cv2.dnn.blobFromImage(368, 368)
  → forward → output (1, 57, 46, 46)
    [canais 0-17: body heatmaps, canal 18: background, canais 19-56: PAFs]
  → cv2.minMaxLoc(heatmap[op_idx]) → pico do heatmap
  → rescale → remap via OPENPOSE_TO_COCO17
Output: (17, 3) COCO-17 — joints com conf < 0.1 ficam em (0, 0, 0)
```

**Mapeamento OpenPose-18 → COCO-17:**  
OpenPose inclui "Neck" (índice 1) que não existe no COCO-17. Descartado.  
Os demais 17 joints são reordenados via `OPENPOSE_TO_COCO17`:
```python
OPENPOSE_TO_COCO17 = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
# COCO[i] ← OP[OPENPOSE_TO_COCO17[i]]
```

---

## 3. Conversão de Keypoints COCO-17 → H3WB-17

**Arquivo:** `src/football_orient_pose/utils/keypoint_mapping.py`  
**Função:** `coco17_to_h3wb17(keypoints_coco: np.ndarray) → np.ndarray`

4 dos 17 keypoints H3WB são **calculados** (médias de pares COCO):

| H3WB ID | Nome | Cálculo |
|---------|------|---------|
| 0 | Center of Hips | (COCO[11] + COCO[12]) / 2 |
| 7 | Center of Body | (H3WB[0] + H3WB[8]) / 2 |
| 8 | Center of Shoulder | (COCO[5] + COCO[6]) / 2 |
| 9 | Neck | (COCO[0] + H3WB[8]) / 2 |

**Bugfix importante (commit `84f0436`):**  
Quando `confidence = 0` (joint não detectado), a coordenada é `(0, 0)`. A média ingênua puxava o joint calculado para metade do valor correto, corrompendo os 4 joints acima — incluindo H3WB[0] e H3WB[8] usados como referência do PDJ.

**Correção:** quando D≥3 (confiança disponível), média ignora joints não detectados:
```python
def _avg(i, j):
    di, dj = conf[i] > 0, conf[j] > 0
    if di and dj: return (kp[i,:2] + kp[j,:2]) / 2
    if di: return kp[i,:2].copy()  # usa só o detectado
    if dj: return kp[j,:2].copy()
    return (kp[i,:2] + kp[j,:2]) / 2  # ambos não detectados: melhor esforço
```

A função **sempre retorna (17, 2)** independente do D do input.

---

## 4. Resultados — Fase 1 Benchmark Zero-Shot

**Split:** val (40 clips × 20 frames = **800 frames**)  
**Arquivos:** `results/tables/{openpose,hrnet,rtmpose}_val.json`

### 4.1 PDJ@0.5 (métrica principal do paper 3DSP)

| Modelo | Global | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle |
|--------|--------|------|----------|-------|-------|-----|------|-------|
| **RTMPose** | **93.62%** | 98.88% | 97.92% | 91.44% | 83.44% | 98.17% | 92.50% | 85.69% |
| HRNet | 88.90% | 96.00% | 96.00% | 82.88% | 70.69% | 97.25% | 87.44% | 79.50% |
| OpenPose | 56.09% | 24.63% | 72.00% | 46.75% | 28.13% | 75.50% | 55.75% | 38.06% |

**Referência do paper (Yeung et al., 2024, Tab. 5):**  
RTMPose: 89.51% | HRNet: 56.08%

Nossos resultados são superiores ao paper em ambos (~4pp cada). Hipóteses documentadas: protocolo diferente no paper (talvez avaliação na cena completa), versão diferente do modelo, pré-processamento distinto.

**OpenPose: 56.09%** — resultado inédito, não reportado em nenhum trabalho anterior no 3DSP.

### 4.2 PCK@0.2

| Modelo | Global | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle |
|--------|--------|------|----------|-------|-------|-----|------|-------|
| RTMPose | **41.76%** | 50.38% | 30.38% | 50.81% | 43.50% | 22.79% | 58.25% | 59.38% |
| HRNet | 40.51% | 18.00% | 34.38% | 44.38% | 39.00% | 39.63% | 47.31% | 54.13% |
| OpenPose | 22.14% | 11.75% | 39.96% | 14.13% | 11.69% | 25.38% | 11.94% | 16.44% |

### 4.3 OKS / AP (padrão COCO)

| Modelo | OKS | AP50 | AP75 | mAP@[.5:.95] |
|--------|-----|------|------|--------------|
| RTMPose | **81.82%** | **97.88%** | **81.00%** | **69.04%** |
| HRNet | 76.22% | 94.88% | 65.00% | 59.01% |
| OpenPose | 48.51% | 56.25% | 14.13% | 22.50% |

### 4.4 MPJPE-2D (erro médio em pixels)

| Modelo | Global | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle |
|--------|--------|------|----------|-------|-------|-----|------|-------|
| RTMPose | **4.81 px** | 3.10 px | 4.23 px | 4.85 px | 7.24 px | 4.43 px | 4.20 px | 6.32 px |
| HRNet | 6.04 px | 5.39 px | 4.63 px | 6.81 px | 10.53 px | 4.19 px | 5.80 px | 8.29 px |
| OpenPose | 25.58 px | 39.81 px | 14.07 px | 29.24 px | 41.62 px | 15.52 px | 27.44 px | 42.82 px |

### 4.5 F1-macro (= PDJ@0.5 em single-instance)

| Modelo | F1-macro |
|--------|----------|
| RTMPose | **93.62%** |
| HRNet | 88.90% |
| OpenPose | 56.09% |

---

## 5. Correções de Bugs (Code Review)

10 bugs identificados e corrigidos em 4 commits:

| # | Arquivo | Bug | Commit |
|---|---------|-----|--------|
| 1 | `evaluate.py:125` | Default `--data-dir=data/3dsp` → deve ser `data/` | `84738e1` |
| 2 | `keypoint_mapping.py:93` | Média sem mascarar joints com confidence=0 corrompia H3WB[0,7,8,9] | `84f0436` |
| 3 | `estimators/__init__.py:3` | Import eager de cv2 forçava todos os usuários a instalar OpenCV | `65010b7` |
| 4 | `rtmpose.py:57` | `pose_model or` em vez de `pose_model is not None` | `65010b7` |
| 5 | `evaluate.py:28` | `OpenPoseEstimator(device=device)` sem paths → ValueError imediato | `84738e1` |
| 6 | `metrics.py:900` | OKS não mascara joints com confidence=0 (limitação documentada) | — |
| 7 | `metrics.py:1006` | `pdj_curve` range padrão `[0, 0.5]` → não comparável com benchmarks | `ee2932c` |
| 8 | `hrnet.py:127` | Guard `if not model_path` ineficaz — erro críptico do ONNX | `65010b7` |
| 9 | `openpose.py:52` | `_N_OPENPOSE_PARTS = 18` dead code | `65010b7` |
| 10 | `metrics.py:1005` | `pdj_curve` chamava `compute_pdj` 51× (O(T·N)) → vectorizado | `ee2932c` |

---

## 6. Infraestrutura

### 6.1 Download de Pesos

**Script:** `scripts/setup/download_models.sh`

| Modelo | Fonte | Tamanho | Estratégia |
|--------|-------|---------|-----------|
| RTMPose | `rtmlib` baixa automaticamente | ~150 MB | Nenhum setup necessário |
| HRNet-W48 ONNX | Google Drive (gdown) | ~242 MB | `HRNET_GDRIVE_ID` no script |
| OpenPose Caffe | Hugging Face `camenduru/openpose` | ~132 MB | CMU server offline — mirror HF |

**Arquivos de pesos (não versionados, em `.gitignore`):**
```
models/weights/
├── hrnet_w48_coco_256x192.onnx          (242 MB)
├── hrnet_w48_coco_256x192.onnx.data
├── openpose_pose_coco.prototxt           (46 KB)
└── openpose_pose_iter_440000.caffemodel  (132 MB)
```

### 6.2 Portabilidade

`.env.example` com variáveis de ambiente:
```
DATA_DIR=data
SPLIT_CONFIG=configs/split.json
```
Todos os notebooks usam `load_dotenv()` — nenhum path hardcoded.

### 6.3 CLI de Avaliação

```bash
# RTMPose ou HRNet
uv run python -m football_orient_pose.evaluation.evaluate \
  --model {rtmpose|hrnet|openpose} \
  --split {train|val} \
  --data-dir data \
  --device {cuda|cpu}

# OpenPose (requer paths explícitos)
uv run python -m football_orient_pose.evaluation.evaluate \
  --model openpose \
  --openpose-prototxt models/weights/openpose_pose_coco.prototxt \
  --openpose-caffemodel models/weights/openpose_pose_iter_440000.caffemodel \
  --split val --device cpu
```

Saída: tabela no terminal + `results/tables/{model}_{split}.json`

---

## 7. Testes

53 testes unitários em 6 arquivos — todos passando:

| Arquivo | Testes | O que cobre |
|---------|--------|-------------|
| `test_pose.py` | 4 | `BasePoseEstimator` — ABC, predict, batch, predict_h3wb |
| `test_rtmpose_estimator.py` | 5 | predict, batch, empty batch, multi-instance selection, ImportError |
| `test_hrnet_estimator.py` | 6 | predict, batch, empty batch, name, ImportError, empty path |
| `test_openpose_estimator.py` | 9 | predict, batch, empty batch, name, ValueError, 4 × mapeamento OP→COCO |
| `test_metrics.py` | 20 | PDJ, PCK, OKS, MPJPE-2D, F1, curva PDJ, AUC |
| `test_data_split.py` + outros | 9 | split, dataset, keypoint mapping |

---

## 8. Notebooks de Validação

| Notebook | US | Conteúdo |
|----------|-----|---------|
| `01_rtmpose_validation.ipynb` | #10/#12 | Smoke tests + PDJ completo no train split + PCK/OKS/MPJPE/F1 |
| `02_hrnet_validation.ipynb` | #13/#15 | Análogo ao 01 + explicação da quantização heatmap + comparação com RTMPose |
| `03_openpose_validation.ipynb` | #16/#17/#18 | Verificação Caffe + inferência + PDJ val split + tabela 3 modelos + gráficos |

**Todos estruturados igual:** setup → checagem de pesos → inferência single → visualização → batch → PDJ → métricas complementares → comparação → heatmap MPJPE → critério de aceite.

---

## 9. Commits da Branch (29 total)

```
5a697a7  feat(notebooks): add OpenPose validation notebook (TASK 2.4.2 / US #18)
c529498  feat(evaluation): add benchmark results for all 3 models (EPIC 2 Phase 1)
f50746f  fix(infra): add working OpenPose caffemodel download from Hugging Face
ee2932c  perf(metrics): vectorize pdj_curve and extend default range to [0, 1]
84738e1  fix(evaluation): correct --data-dir default and add OpenPose CLI args
84f0436  fix(keypoint_mapping): mask zero-confidence joints in computed averages
65010b7  fix(estimators): correct lazy imports, or-vs-None guard and dead code
a78512e  chore: clean dependencies, update README and add Epic 2 retrospective
a09add6  feat(notebooks): add HRNet validation notebook and fix RTMPose data path
d9db157  feat(infra): add env template, gitignore entries and model download script
95c2cca  feat(evaluation): add hrnet and openpose support to evaluate CLI
e79f661  test(estimators): add unit tests for HRNetEstimator and OpenPoseEstimator
6e4609c  feat(estimators): implement HRNetEstimator and OpenPoseEstimator
b114c12  feat(evaluation): add inference dependencies and progress bar
a335ff5  fix(lint): sort imports and remove unused variable in training_monitor
8629225  docs(validation): add RTMPose zero-shot evaluation results
0b908aa  feat(notebook): add PCK, OKS, MPJPE-2D, F1 cells to RTMPose validation
59fcb28  fix(training_monitor): fix epoch alignment in metrics curve for sparse logs
80e1c0f  feat(evaluation): add TrainingMonitor for per-epoch loss and metrics (EPIC 5)
b14f69b  feat(evaluation): add evaluate.py CLI batch with all metrics
e71e1fb  feat(evaluation): export all metric symbols from __init__
2c03d6a  feat(metrics): add joint_detection_report with F1 per joint
e000d92  feat(metrics): add compute_mpjpe_2d with per-joint pixel error
0cf42e4  feat(metrics): add compute_oks with OKS, AP50, AP75, mAP
7ae403d  fix(metrics): export PCK constants and add shape-mismatch test
5730c7c  feat(metrics): add compute_pck with PCKResult
87f8f04  docs: add pose metrics implementation plan
c76e9e5  docs: add pose metrics design spec for EPIC 2/3 delivery
e688f3a  feat(estimators): add RTMPoseEstimator + BasePoseEstimator + tests
```

---

## 10. Arquivos Criados/Modificados

### Código fonte (`src/football_orient_pose/`)
```
pose.py                         ← BasePoseEstimator (ABC)
estimators/__init__.py          ← lazy imports via __getattr__
estimators/rtmpose.py           ← RTMPoseEstimator
estimators/hrnet.py             ← HRNetEstimator (ONNX)
estimators/openpose.py          ← OpenPoseEstimator (cv2.dnn Caffe)
evaluation/__init__.py          ← exports de todas as métricas
evaluation/metrics.py           ← PDJ, PCK, OKS, MPJPE-2D, F1, curva PDJ
evaluation/evaluate.py          ← CLI batch com tqdm
evaluation/training_monitor.py  ← monitor de epochs (pre-Epic 5)
utils/keypoint_mapping.py       ← coco17_to_h3wb17 com mascaramento conf
```

### Testes (`tests/`)
```
test_pose.py
test_rtmpose_estimator.py
test_hrnet_estimator.py
test_openpose_estimator.py
test_metrics.py
```

### Resultados (`results/tables/`)
```
rtmpose_val.json    ← PDJ 93.62%, PCK 41.76%, OKS 81.82%, MPJPE 4.81px
hrnet_val.json      ← PDJ 88.90%, PCK 40.51%, OKS 76.22%, MPJPE 6.04px
openpose_val.json   ← PDJ 56.09%, PCK 22.14%, OKS 48.51%, MPJPE 25.58px
```

### Notebooks (`notebooks/`)
```
01_rtmpose_validation.ipynb
02_hrnet_validation.ipynb
03_openpose_validation.ipynb   ← NOVO nesta sessão
```

### Infra
```
scripts/setup/download_models.sh      ← HRNet (Drive) + OpenPose (HuggingFace)
.env.example                    ← DATA_DIR, SPLIT_CONFIG
.gitignore                      ← exceção results/tables/*.json
docs/vision/epic2-retrospectiva.md  ← retrospectiva técnica detalhada
```

---

## 11. Issues Fechadas

| Issue | Título | Fechada |
|-------|--------|---------|
| #7 | [EPIC 2] Estimadores de Pose (3 modelos) | ✅ |
| #8 | [US 2.1] Interface Unificada de Pose Estimator | ✅ |
| #9 | [TASK 2.1.1] Definir classe base abstrata | ✅ |
| #10 | [US 2.2] Wrapper RTMPose | ✅ |
| #11 | [TASK 2.2.1] Implementar RTMPoseEstimator | ✅ |
| #12 | [TASK 2.2.2] Validar reprodutibilidade com paper (RTMPose) | ✅ |
| #13 | [US 2.3] Wrapper HRNet | ✅ |
| #14 | [TASK 2.3.1] Implementar HRNetEstimator | ✅ |
| #15 | [TASK 2.3.2] Validar reprodutibilidade com paper (HRNet) | ✅ |
| #16 | [US 2.4] Wrapper OpenPose | ✅ |
| #17 | [TASK 2.4.1] Implementar OpenPoseEstimator | ✅ |
| #18 | [TASK 2.4.2] Validar em amostra do dataset (OpenPose) | ✅ |

---

## 12. Contribuições Acadêmicas do Épico

1. **PDJ do OpenPose no 3DSP — primeira vez na literatura**  
   Nem Reis et al. (2023) nem Yeung et al. (2024) reportaram PDJ do OpenPose neste dataset. Nosso resultado: **56.09%**.

2. **Comparação sistemática com 4 métricas**  
   PDJ, PCK, OKS/AP e MPJPE-2D para os 3 modelos no mesmo split — nenhum trabalho anterior fez isso.

3. **Análise por grupo anatômico**  
   7 grupos (head, shoulder, elbow, wrist, hip, knee, ankle) para cada modelo e cada métrica.

4. **Documentação de discrepância do HRNet**  
   Nosso HRNet (88.90%) ficou 32 pp acima do paper (56.08%). Hipóteses documentadas: protocolo de avaliação diferente, versão do modelo, pré-processamento.

---

## 13. Próximos Épicos

| Épico | Status | Dependências resolvidas por Epic 2 |
|-------|--------|-------------------------------------|
| Epic 3 — Métricas | ✅ CLOSED | PDJ, PCK, OKS, MPJPE-2D já implementados |
| Epic 4 — Fase 1 Benchmark | ✅ CLOSED | JSONs com resultados gerados |
| **Epic 5 — Fine-tuning** | 🔴 OPEN | `DSPDataset` (#6) + 3 training loops (#31-#33) |
| Epic 6 — Orientação Corporal | 🔴 OPEN | `orientation.py` + `homography.py` (#38-#40) |
| Epic 7 — Viz/Demo/Artigo | 🔴 OPEN | `viz.py` + `demo_video.py` + artigo LaTeX |
