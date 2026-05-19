# Design: Métricas Completas de Pose Estimation (EPIC 2/3 + prep EPIC 5)

**Data:** 2026-05-19
**Issues:** #10, #11, #12, #20, #22, #23, #24
**Branch:** 7-epic-2-estimadores-de-pose-3-modelos

---

## Contexto

O `RTMPoseEstimator` está implementado e todos os 9 testes unitários passam. A única
métrica implementada até agora é PDJ (`compute_pdj`, `pdj_curve`, `pdj_auc`). Há uma
entrega parcial hoje que requer:

1. Todas as métricas padrão de 2D pose estimation
2. Notebook de validação executável com visualizações
3. CLI de avaliação batch
4. Framework de monitoramento de treino (para EPIC 5)

---

## Métricas Padrão para 2D Pose Estimation

| Métrica | Threshold padrão | Normalização | Literatura |
|---------|-----------------|--------------|------------|
| PDJ | 0.5 | torso (hips↔shoulder) | 3DSP, AutoSoccerPose |
| PCK | 0.2 | max(shoulder_width, hip_width) | MPII, LSP |
| OKS | — | sigmas COCO por joint + área bbox | COCO |
| AP / mAP | 0.5:0.05:0.95 | via OKS | COCO |
| MPJPE-2D | — | pixels absolutos | Human3.6M adaptado |
| Detection F1 | via PDJ thr | binário por joint | análise complementar |

> **Nota:** Confusion matrix e F1 macro se aplicam ao EPIC 6 (orientação como
> classificação em N direções). Para pose, o equivalente é `joint_detection_report`.

---

## Arquitetura

```
src/football_orient_pose/
├── evaluation/
│   ├── __init__.py          # exporta tudo
│   ├── metrics.py           # MODIFICAR — adicionar PCK, OKS, MPJPE-2D, detection report
│   ├── evaluate.py          # NOVO — CLI batch
│   └── training_monitor.py  # NOVO — callbacks para EPIC 5
notebooks/
└── 01_rtmpose_validation.ipynb  # MODIFICAR — adicionar células de todas as métricas
tests/
└── test_metrics.py          # MODIFICAR — adicionar testes de PCK, OKS, MPJPE
```

---

## Especificação: `metrics.py`

### Constantes novas

```python
# IDs H3WB para PCK reference
PCK_LEFT_SHOULDER_ID  = 14
PCK_RIGHT_SHOULDER_ID = 11
PCK_LEFT_HIP_ID       = 1
PCK_RIGHT_HIP_ID      = 4

# Sigmas H3WB mapeados dos sigmas COCO-17
H3WB_SIGMAS = np.array([
    0.107,  # 0: Center of Hips       (média hip L/R)
    0.107,  # 1: Left Hip
    0.087,  # 2: Left Knee
    0.089,  # 3: Left Ankle
    0.107,  # 4: Right Hip
    0.087,  # 5: Right Knee
    0.089,  # 6: Right Ankle
    0.093,  # 7: Center of Body       ((hip+shoulder)/2)
    0.079,  # 8: Center of Shoulder   (média shoulder L/R)
    0.052,  # 9: Neck                 ((nose+shoulder)/2)
    0.026,  # 10: Head (nose)
    0.079,  # 11: Right Shoulder
    0.072,  # 12: Right Elbow
    0.062,  # 13: Right Wrist
    0.079,  # 14: Left Shoulder
    0.072,  # 15: Left Elbow
    0.062,  # 16: Left Wrist
], dtype=np.float32)

DSP_CROP_AREA = 10_000.0  # 100×100 px² — área fixa dos crops 3DSP
```

### `compute_pck`

```python
@dataclass(frozen=True)
class PCKResult:
    threshold: float
    global_score: float
    per_joint: np.ndarray       # (17,)
    per_group: dict[str, float]
    valid_frames: int

def compute_pck(
    predicted: np.ndarray,      # (17,2) ou (N,17,2) H3WB
    target: np.ndarray,
    threshold: float = 0.2,
) -> PCKResult:
    """ref_size = max(dist(LS[14], RS[11]), dist(LH[1], RH[4]))"""
```

Frames inválidos (ref_size == 0) são descartados do denominador, igual ao PDJ.

### `compute_oks`

```python
@dataclass(frozen=True)
class OKSResult:
    global_oks: float
    per_joint_oks: np.ndarray   # (17,)
    ap: float                   # mAP@[0.5:0.05:0.95]
    ap50: float
    ap75: float
    ap_per_threshold: dict[float, float]
    valid_frames: int

def compute_oks(
    predicted: np.ndarray,      # (17,2) ou (N,17,2) H3WB
    target: np.ndarray,
    area: float = DSP_CROP_AREA,
    sigmas: np.ndarray | None = None,  # None → H3WB_SIGMAS
) -> OKSResult:
```

Fórmula OKS por frame:
`OKS_i = mean(exp(-d_j² / (2 * s² * sigma_j²)))` onde `s = sqrt(area)`.

AP por threshold `t`: fração de frames com `OKS_i >= t`.
`mAP = mean(AP@t para t em [0.50, 0.55, ..., 0.95])`.

### `compute_mpjpe_2d`

```python
@dataclass(frozen=True)
class MPJPE2DResult:
    global_mpjpe: float         # erro médio em pixels
    per_joint: np.ndarray       # (17,) pixels
    per_group: dict[str, float]

def compute_mpjpe_2d(
    predicted: np.ndarray,      # (17,2) ou (N,17,2)
    target: np.ndarray,
) -> MPJPE2DResult:
    """Erro Euclidiano médio em pixels, sem normalização."""
```

### `joint_detection_report`

```python
@dataclass(frozen=True)
class JointDetectionReport:
    threshold: float
    precision: np.ndarray       # (17,)
    recall: np.ndarray          # (17,)
    f1: np.ndarray              # (17,)
    f1_macro: float
    per_group: dict[str, float] # F1 médio por grupo

def joint_detection_report(
    predicted: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
    torso_ids: tuple[int, int] = (0, 8),
) -> JointDetectionReport:
    """
    Cada joint é tratado como detecção binária (PDJ-like).
    TP = dentro do threshold, FP = fora, FN = FP do ponto de vista do GT.
    Para single-instance: precision_j = recall_j = PDJ_j (são equivalentes).
    F1_j = 2 * PDJ_j / (PDJ_j + 1) ... simplificado para harmônica.
    """
```

> **Nota implementação:** Em avaliação single-instance (1 pred / 1 GT por frame),
> precision e recall por joint são numericamente iguais ao PDJ por joint. O valor
> diferenciado de precision vs recall só surge com múltiplas instâncias (EPIC 4/5).
> `f1_macro` = média do F1 por joint — reportar junto com PDJ global como cross-check.

### Exportações (`__init__.py`)

Adicionar: `PCKResult`, `compute_pck`, `OKSResult`, `compute_oks`,
`MPJPE2DResult`, `compute_mpjpe_2d`, `JointDetectionReport`, `joint_detection_report`.

---

## Especificação: `evaluate.py`

```
python -m football_orient_pose.evaluation.evaluate \
  --model rtmpose \
  --split val \
  --data-dir data/3dsp \
  --split-config configs/split.json \
  --output-dir results/tables/
```

### Fluxo

1. Parse args
2. Carregar split de `split.json`
3. Instanciar estimador via factory `{rtmpose: RTMPoseEstimator}`
4. Loop sobre clips e frames: `predict_h3wb()` + carregar GT
5. Calcular PDJ, PCK, OKS, MPJPE-2D, detection report
6. Salvar `results/tables/{model}_{split}.json`
7. Imprimir tabela formatada no terminal

### Output JSON

```json
{
  "model": "rtmpose",
  "split": "val",
  "n_clips": 40,
  "n_frames": 800,
  "pdj": { "global": 0.895, "per_group": { "head": 0.92 } },
  "pck": { "global": 0.872, "per_group": { "head": 0.91 } },
  "oks": { "global_oks": 0.81, "ap": 0.77, "ap50": 0.91, "ap75": 0.83 },
  "mpjpe_2d": { "global_px": 4.2, "per_group": { "head": 3.1 } },
  "f1_macro": 0.889
}
```

---

## Especificação: `training_monitor.py`

Módulo independente para ser usado no EPIC 5 (fine-tuning). **Não bloqueia a entrega de hoje.**

```python
class TrainingMonitor:
    """Registra métricas por época e salva JSON + plots."""

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_pdj: float | None = None,
        val_pck: float | None = None,
        val_oks: float | None = None,
    ) -> None: ...

    def save(self, output_dir: Path) -> None:
        """Salva history.json e gera plots de loss e métricas por época."""

    def plot_loss_curve(self) -> plt.Figure: ...
    def plot_metrics_curve(self) -> plt.Figure: ...
```

Saída esperada ao final do treino:
- `results/training/{model}/history.json`
- `results/training/{model}/loss_curve.png`
- `results/training/{model}/metrics_curve.png`

---

## Especificação: Notebook `01_rtmpose_validation.ipynb`

Células a adicionar após as células de PDJ existentes:

| # | Tipo | Conteúdo |
|---|------|---------|
| A | code | `compute_pck(predictions, targets)` — tabela por grupo |
| B | code | `compute_oks(predictions, targets)` — OKS, AP50, AP75, mAP |
| C | code | `compute_mpjpe_2d(predictions, targets)` — erro em pixels |
| D | code | `joint_detection_report(...)` — F1 por joint, f1_macro |
| E | code | Bar chart: PDJ/PCK/F1 por grupo anatômico (matplotlib) |
| F | code | Heatmap: erro MPJPE-2D por joint (visualização tipo confusion) |
| G | markdown | Tabela-resumo: todas as métricas vs referências do paper |

---

## Testes

Adicionar em `tests/test_metrics.py`:

- `test_compute_pck_returns_perfect_score_for_identical_keypoints`
- `test_compute_pck_uses_max_shoulder_hip_reference`
- `test_compute_pck_discards_frames_with_zero_reference`
- `test_compute_oks_returns_one_for_identical_keypoints`
- `test_compute_oks_returns_ap_one_for_perfect_predictions`
- `test_compute_mpjpe_2d_returns_zero_for_identical_keypoints`
- `test_joint_detection_report_returns_f1_one_for_perfect_predictions`

---

## Critérios de Aceitação (entrega de hoje)

- [ ] `compute_pck`, `compute_oks`, `compute_mpjpe_2d`, `joint_detection_report` implementados e testados
- [ ] `evaluate.py` roda via CLI sem erros no split val (40 clips)
- [ ] Notebook executa end-to-end com smoke test (2 clips)
- [ ] Full run (200 clips) com PDJ@0.5 dentro de ±2% de 89.51%
- [ ] `training_monitor.py` criado (mesmo que sem testes agora)
- [ ] Issues #11, #22, #23, #24 fechadas
- [ ] Issues #10 e #12 fechadas após full run do notebook
