# Pose Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar todas as métricas padrão de 2D pose estimation (PCK, OKS/AP, MPJPE-2D, F1 por joint), CLI de avaliação batch, framework de monitoramento de treino, e atualizar o notebook de validação do RTMPose com todas as métricas.

**Architecture:** Todas as métricas ficam em `evaluation/metrics.py` seguindo o padrão `XxxResult` dataclass já estabelecido pelo `PDJResult`. A CLI em `evaluate.py` orquestra tudo. O `training_monitor.py` é independente e preparado para o EPIC 5.

**Tech Stack:** numpy, matplotlib, argparse — sem dependências novas.

---

## Mapa de Arquivos

| Arquivo | Ação | Responsabilidade |
|---------|------|-----------------|
| `src/football_orient_pose/evaluation/metrics.py` | Modificar | Adicionar PCK, OKS, MPJPE-2D, JointDetectionReport |
| `src/football_orient_pose/evaluation/__init__.py` | Modificar | Exportar novos símbolos |
| `src/football_orient_pose/evaluation/evaluate.py` | Criar | CLI batch de avaliação |
| `src/football_orient_pose/evaluation/training_monitor.py` | Criar | Monitoramento por época (EPIC 5) |
| `tests/test_metrics.py` | Modificar | Testes de PCK, OKS, MPJPE-2D, JointDetectionReport |
| `notebooks/01_rtmpose_validation.ipynb` | Modificar | Células das novas métricas |

---

## Task 1: Implementar `compute_pck`

**Files:**
- Modify: `src/football_orient_pose/evaluation/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1.1: Escrever os testes que falham**

Adicionar ao final de `tests/test_metrics.py`:

```python
from football_orient_pose.evaluation.metrics import (
    compute_pdj,
    compute_pck,
    PCKResult,
)


def _make_pck_gt(n: int = 5) -> np.ndarray:
    """GT com ombros e quadris separados para ter referência PCK válida."""
    gt = np.zeros((n, 17, 2), dtype=np.float32)
    gt[:, 14] = [30.0, 50.0]   # Left Shoulder  (H3WB 14)
    gt[:, 11] = [70.0, 50.0]   # Right Shoulder (H3WB 11) → shoulder_width = 40
    gt[:, 1]  = [35.0, 80.0]   # Left Hip       (H3WB 1)
    gt[:, 4]  = [65.0, 80.0]   # Right Hip      (H3WB 4)  → hip_width = 30
    return gt


def test_compute_pck_returns_perfect_score_for_identical_keypoints() -> None:
    gt = _make_pck_gt()
    result = compute_pck(gt, gt, threshold=0.2)

    assert isinstance(result, PCKResult)
    assert result.global_score == pytest.approx(1.0)
    assert result.valid_frames == 5
    np.testing.assert_array_equal(result.per_joint, np.ones(17, dtype=np.float32))


def test_compute_pck_uses_max_shoulder_hip_reference() -> None:
    gt = _make_pck_gt(1)
    # Desloca joint 10 (Head) 12 px → dist/ref = 12/40 = 0.30 > threshold 0.2
    pred = gt.copy()
    pred[:, 10] = gt[:, 10] + np.array([12.0, 0.0])

    result = compute_pck(pred, gt, threshold=0.2)

    assert result.per_joint[10] == pytest.approx(0.0)
    # Todos os outros joints estão perfeitos
    assert result.per_joint[:10].mean() == pytest.approx(1.0)
    assert result.per_joint[11:].mean() == pytest.approx(1.0)


def test_compute_pck_discards_frames_with_zero_reference() -> None:
    gt = np.zeros((3, 17, 2), dtype=np.float32)
    # Frame 0: sem referência (todos zeros → shoulder_width = hip_width = 0)
    gt[1:, 14] = [30.0, 50.0]
    gt[1:, 11] = [70.0, 50.0]  # shoulder_width = 40 nos frames 1 e 2

    result = compute_pck(gt, gt, threshold=0.2)

    assert result.valid_frames == 2  # frame 0 descartado
    assert result.global_score == pytest.approx(1.0)
```

- [ ] **Step 1.2: Verificar que os testes falham**

```bash
uv run pytest tests/test_metrics.py::test_compute_pck_returns_perfect_score_for_identical_keypoints -v
```
Esperado: `ImportError: cannot import name 'compute_pck'`

- [ ] **Step 1.3: Implementar `PCKResult` e `compute_pck` em `metrics.py`**

Adicionar após a definição de `PDJ_GROUPS` e antes de `PDJResult`:

```python
PCK_LEFT_SHOULDER_ID = 14
PCK_RIGHT_SHOULDER_ID = 11
PCK_LEFT_HIP_ID = 1
PCK_RIGHT_HIP_ID = 4
```

Adicionar após `PDJResult` e `compute_pdj`:

```python
@dataclass(frozen=True)
class PCKResult:
    """Resultado agregado de PCK."""

    threshold: float
    global_score: float
    per_joint: np.ndarray
    per_group: dict[str, float]
    valid_frames: int


def compute_pck(
    predicted: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.2,
) -> PCKResult:
    """Calcula PCK (Percentage of Correct Keypoints) para keypoints H3WB-17.

    Referência = max(dist(LeftShoulder[14], RightShoulder[11]),
                     dist(LeftHip[1], RightHip[4])).
    Threshold padrão 0.2 (PCK@0.2 é o mais comum na literatura).
    """
    pred = _ensure_keypoint_batch(predicted, "predicted")
    gt = _ensure_keypoint_batch(target, "target")
    if pred.shape != gt.shape:
        raise ValueError(
            f"predicted e target devem ter o mesmo shape: {pred.shape} != {gt.shape}"
        )

    shoulder_width = np.linalg.norm(
        gt[:, PCK_LEFT_SHOULDER_ID] - gt[:, PCK_RIGHT_SHOULDER_ID], axis=1
    )
    hip_width = np.linalg.norm(
        gt[:, PCK_LEFT_HIP_ID] - gt[:, PCK_RIGHT_HIP_ID], axis=1
    )
    ref_size = np.maximum(shoulder_width, hip_width)
    valid = ref_size > 0

    if not np.any(valid):
        raise ValueError("Nenhum frame possui referência de tamanho válida para PCK")

    distances = np.linalg.norm(pred[valid] - gt[valid], axis=2)
    normalized = distances / ref_size[valid, np.newaxis]
    correct = normalized < threshold

    per_joint = correct.mean(axis=0)
    per_group = {
        group_name: float(per_joint[joint_ids].mean())
        for group_name, joint_ids in PDJ_GROUPS.items()
    }
    return PCKResult(
        threshold=threshold,
        global_score=float(correct.mean()),
        per_joint=per_joint.astype(np.float32),
        per_group=per_group,
        valid_frames=int(valid.sum()),
    )
```

- [ ] **Step 1.4: Verificar que os testes passam**

```bash
uv run pytest tests/test_metrics.py -k "pck" -v
```
Esperado: 3 testes PASS.

- [ ] **Step 1.5: Commit**

```bash
git add src/football_orient_pose/evaluation/metrics.py tests/test_metrics.py
git commit -m "feat(metrics): add compute_pck with PCKResult"
```

---

## Task 2: Implementar `compute_oks`

**Files:**
- Modify: `src/football_orient_pose/evaluation/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 2.1: Escrever os testes que falham**

Adicionar ao final de `tests/test_metrics.py`:

```python
from football_orient_pose.evaluation.metrics import compute_oks, OKSResult


def test_compute_oks_returns_one_for_identical_keypoints() -> None:
    gt = np.random.default_rng(42).random((5, 17, 2)).astype(np.float32) * 100

    result = compute_oks(gt, gt)

    assert isinstance(result, OKSResult)
    assert result.global_oks == pytest.approx(1.0)
    assert result.ap == pytest.approx(1.0)
    assert result.ap50 == pytest.approx(1.0)
    assert result.ap75 == pytest.approx(1.0)
    assert result.valid_frames == 5


def test_compute_oks_decreases_with_larger_error() -> None:
    gt = np.zeros((10, 17, 2), dtype=np.float32)
    pred_small = gt.copy()
    pred_small[:, :, 0] += 1.0
    pred_large = gt.copy()
    pred_large[:, :, 0] += 20.0

    result_small = compute_oks(pred_small, gt)
    result_large = compute_oks(pred_large, gt)

    assert result_small.global_oks > result_large.global_oks


def test_compute_oks_ap_per_threshold_has_ten_keys() -> None:
    gt = np.zeros((3, 17, 2), dtype=np.float32)

    result = compute_oks(gt, gt)

    assert len(result.ap_per_threshold) == 10
    assert 0.5 in result.ap_per_threshold
    assert 0.75 in result.ap_per_threshold
    assert 0.95 in result.ap_per_threshold
```

- [ ] **Step 2.2: Verificar que os testes falham**

```bash
uv run pytest tests/test_metrics.py -k "oks" -v
```
Esperado: `ImportError: cannot import name 'compute_oks'`

- [ ] **Step 2.3: Implementar `OKSResult` e `compute_oks` em `metrics.py`**

Adicionar as constantes antes de `PCKResult`:

```python
H3WB_SIGMAS: np.ndarray = np.array([
    0.107,  # 0:  Center of Hips
    0.107,  # 1:  Left Hip
    0.087,  # 2:  Left Knee
    0.089,  # 3:  Left Ankle
    0.107,  # 4:  Right Hip
    0.087,  # 5:  Right Knee
    0.089,  # 6:  Right Ankle
    0.093,  # 7:  Center of Body
    0.079,  # 8:  Center of Shoulder
    0.052,  # 9:  Neck
    0.026,  # 10: Head
    0.079,  # 11: Right Shoulder
    0.072,  # 12: Right Elbow
    0.062,  # 13: Right Wrist
    0.079,  # 14: Left Shoulder
    0.072,  # 15: Left Elbow
    0.062,  # 16: Left Wrist
], dtype=np.float32)

DSP_CROP_AREA: float = 10_000.0  # 100×100 px²
```

Adicionar após `compute_pck`:

```python
@dataclass(frozen=True)
class OKSResult:
    """Resultado OKS com AP calculado em limiares COCO."""

    global_oks: float
    per_joint_oks: np.ndarray
    ap: float
    ap50: float
    ap75: float
    ap_per_threshold: dict[float, float]
    valid_frames: int


def compute_oks(
    predicted: np.ndarray,
    target: np.ndarray,
    area: float = DSP_CROP_AREA,
    sigmas: np.ndarray | None = None,
) -> OKSResult:
    """Calcula OKS e AP para keypoints H3WB-17.

    OKS por frame: mean(exp(-d_j² / (2 * s² * sigma_j²)))
    AP@t: fração de frames com OKS >= t (avaliação single-instance).
    mAP: média de AP@t para t em [0.50, 0.55, ..., 0.95].
    """
    pred = _ensure_keypoint_batch(predicted, "predicted")
    gt = _ensure_keypoint_batch(target, "target")
    if pred.shape != gt.shape:
        raise ValueError(
            f"predicted e target devem ter o mesmo shape: {pred.shape} != {gt.shape}"
        )

    if sigmas is None:
        sigmas = H3WB_SIGMAS

    s = float(np.sqrt(area))
    d = np.linalg.norm(pred - gt, axis=2)                     # (N, 17)
    oks_per_joint = np.exp(-d**2 / (2.0 * s**2 * sigmas**2))  # (N, 17)
    oks_per_frame = oks_per_joint.mean(axis=1)                 # (N,)

    thresholds = [round(0.5 + i * 0.05, 2) for i in range(10)]
    ap_per_threshold = {
        t: float((oks_per_frame >= t).mean()) for t in thresholds
    }

    return OKSResult(
        global_oks=float(oks_per_frame.mean()),
        per_joint_oks=oks_per_joint.mean(axis=0).astype(np.float32),
        ap=float(np.mean(list(ap_per_threshold.values()))),
        ap50=ap_per_threshold[0.5],
        ap75=ap_per_threshold[0.75],
        ap_per_threshold=ap_per_threshold,
        valid_frames=len(pred),
    )
```

- [ ] **Step 2.4: Verificar que os testes passam**

```bash
uv run pytest tests/test_metrics.py -k "oks" -v
```
Esperado: 3 testes PASS.

- [ ] **Step 2.5: Commit**

```bash
git add src/football_orient_pose/evaluation/metrics.py tests/test_metrics.py
git commit -m "feat(metrics): add compute_oks with OKS, AP50, AP75, mAP"
```

---

## Task 3: Implementar `compute_mpjpe_2d`

**Files:**
- Modify: `src/football_orient_pose/evaluation/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 3.1: Escrever os testes que falham**

Adicionar ao final de `tests/test_metrics.py`:

```python
from football_orient_pose.evaluation.metrics import compute_mpjpe_2d, MPJPE2DResult


def test_compute_mpjpe_2d_returns_zero_for_identical_keypoints() -> None:
    gt = np.random.default_rng(7).random((5, 17, 2)).astype(np.float32) * 100

    result = compute_mpjpe_2d(gt, gt)

    assert isinstance(result, MPJPE2DResult)
    assert result.global_mpjpe == pytest.approx(0.0)
    np.testing.assert_array_almost_equal(result.per_joint, np.zeros(17))


def test_compute_mpjpe_2d_measures_euclidean_pixels() -> None:
    gt = np.zeros((4, 17, 2), dtype=np.float32)
    pred = gt.copy()
    pred[:, :, 0] += 3.0  # deslocamento x = 3
    pred[:, :, 1] += 4.0  # deslocamento y = 4 → norma = 5.0

    result = compute_mpjpe_2d(pred, gt)

    assert result.global_mpjpe == pytest.approx(5.0)
    np.testing.assert_array_almost_equal(result.per_joint, np.full(17, 5.0))
```

- [ ] **Step 3.2: Verificar que os testes falham**

```bash
uv run pytest tests/test_metrics.py -k "mpjpe" -v
```
Esperado: `ImportError: cannot import name 'compute_mpjpe_2d'`

- [ ] **Step 3.3: Implementar `MPJPE2DResult` e `compute_mpjpe_2d` em `metrics.py`**

Adicionar após `compute_oks`:

```python
@dataclass(frozen=True)
class MPJPE2DResult:
    """Erro médio em pixels para keypoints 2D."""

    global_mpjpe: float
    per_joint: np.ndarray
    per_group: dict[str, float]


def compute_mpjpe_2d(
    predicted: np.ndarray,
    target: np.ndarray,
) -> MPJPE2DResult:
    """Calcula MPJPE (Mean Per Joint Position Error) em pixels para H3WB-17.

    Sem normalização — erro absoluto em pixels no espaço do crop 100×100.
    """
    pred = _ensure_keypoint_batch(predicted, "predicted")
    gt = _ensure_keypoint_batch(target, "target")
    if pred.shape != gt.shape:
        raise ValueError(
            f"predicted e target devem ter o mesmo shape: {pred.shape} != {gt.shape}"
        )

    error = np.linalg.norm(pred - gt, axis=2)  # (N, 17)
    per_joint = error.mean(axis=0)
    per_group = {
        group_name: float(per_joint[joint_ids].mean())
        for group_name, joint_ids in PDJ_GROUPS.items()
    }
    return MPJPE2DResult(
        global_mpjpe=float(error.mean()),
        per_joint=per_joint.astype(np.float32),
        per_group=per_group,
    )
```

- [ ] **Step 3.4: Verificar que os testes passam**

```bash
uv run pytest tests/test_metrics.py -k "mpjpe" -v
```
Esperado: 2 testes PASS.

- [ ] **Step 3.5: Commit**

```bash
git add src/football_orient_pose/evaluation/metrics.py tests/test_metrics.py
git commit -m "feat(metrics): add compute_mpjpe_2d with per-joint pixel error"
```

---

## Task 4: Implementar `joint_detection_report`

**Files:**
- Modify: `src/football_orient_pose/evaluation/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 4.1: Escrever os testes que falham**

Adicionar ao final de `tests/test_metrics.py`:

```python
from football_orient_pose.evaluation.metrics import (
    joint_detection_report,
    JointDetectionReport,
)


def _make_torso_gt(n: int = 5) -> np.ndarray:
    gt = np.zeros((n, 17, 2), dtype=np.float32)
    gt[:, 0] = [10.0, 0.0]   # Center of Hips
    gt[:, 8] = [10.0, 20.0]  # Center of Shoulder → torso_size = 20
    return gt


def test_joint_detection_report_perfect_predictions() -> None:
    gt = _make_torso_gt()

    result = joint_detection_report(gt, gt, threshold=0.5)

    assert isinstance(result, JointDetectionReport)
    assert result.f1_macro == pytest.approx(1.0)
    np.testing.assert_array_equal(result.f1, np.ones(17, dtype=np.float32))
    np.testing.assert_array_equal(result.precision, result.recall)


def test_joint_detection_report_f1_macro_equals_mean_pdj_per_joint() -> None:
    gt = _make_torso_gt()
    pred = gt.copy()
    # Desloca joint 10 (Head) além do threshold: 15px / 20px torso = 0.75 > 0.5
    pred[:, 10] = gt[:, 10] + np.array([15.0, 0.0])

    pdj = compute_pdj(pred, gt, threshold=0.5)
    report = joint_detection_report(pred, gt, threshold=0.5)

    assert report.f1_macro == pytest.approx(float(pdj.per_joint.mean()))
    assert report.f1[10] == pytest.approx(0.0)
```

- [ ] **Step 4.2: Verificar que os testes falham**

```bash
uv run pytest tests/test_metrics.py -k "detection_report" -v
```
Esperado: `ImportError: cannot import name 'joint_detection_report'`

- [ ] **Step 4.3: Implementar `JointDetectionReport` e `joint_detection_report` em `metrics.py`**

Adicionar após `compute_mpjpe_2d`:

```python
@dataclass(frozen=True)
class JointDetectionReport:
    """Precision/recall/F1 por joint tratando detecção como classificação binária.

    Em avaliação single-instance, precision_j = recall_j = f1_j = PDJ_j por joint.
    """

    threshold: float
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    f1_macro: float
    per_group: dict[str, float]


def joint_detection_report(
    predicted: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
    torso_ids: tuple[int, int] = (0, 8),
) -> JointDetectionReport:
    """Gera relatório de detecção por joint usando PDJ como base.

    Em avaliação single-instance (um crop = um jogador), precision e recall
    por joint são numericamente iguais ao PDJ por joint. O valor diferenciado
    só emerge com múltiplas instâncias por imagem (EPIC 4/5).
    """
    pdj_result = compute_pdj(predicted, target, threshold=threshold, torso_ids=torso_ids)
    per_joint = pdj_result.per_joint
    per_group = {
        group_name: float(per_joint[joint_ids].mean())
        for group_name, joint_ids in PDJ_GROUPS.items()
    }
    return JointDetectionReport(
        threshold=threshold,
        precision=per_joint.copy(),
        recall=per_joint.copy(),
        f1=per_joint.copy(),
        f1_macro=float(per_joint.mean()),
        per_group=per_group,
    )
```

- [ ] **Step 4.4: Verificar que os testes passam**

```bash
uv run pytest tests/test_metrics.py -k "detection_report" -v
```
Esperado: 2 testes PASS.

- [ ] **Step 4.5: Rodar toda a suite de testes**

```bash
uv run pytest tests/test_metrics.py -v
```
Esperado: todos os testes passam (5 antigos + 9 novos = 14 no total).

- [ ] **Step 4.6: Commit**

```bash
git add src/football_orient_pose/evaluation/metrics.py tests/test_metrics.py
git commit -m "feat(metrics): add joint_detection_report with F1 per joint and f1_macro"
```

---

## Task 5: Atualizar exportações do `__init__.py`

**Files:**
- Modify: `src/football_orient_pose/evaluation/__init__.py`

- [ ] **Step 5.1: Substituir o conteúdo do `__init__.py`**

```python
"""Subpacote de avaliação: métricas e comparação com baseline."""

from football_orient_pose.evaluation.metrics import (
    PDJ_GROUPS,
    PDJResult,
    PCKResult,
    OKSResult,
    MPJPE2DResult,
    JointDetectionReport,
    H3WB_SIGMAS,
    DSP_CROP_AREA,
    compute_pdj,
    compute_pck,
    compute_oks,
    compute_mpjpe_2d,
    joint_detection_report,
    pdj_auc,
    pdj_curve,
)

__all__ = [
    "PDJ_GROUPS",
    "PDJResult",
    "PCKResult",
    "OKSResult",
    "MPJPE2DResult",
    "JointDetectionReport",
    "H3WB_SIGMAS",
    "DSP_CROP_AREA",
    "compute_pdj",
    "compute_pck",
    "compute_oks",
    "compute_mpjpe_2d",
    "joint_detection_report",
    "pdj_auc",
    "pdj_curve",
]
```

- [ ] **Step 5.2: Verificar importação via pacote**

```bash
uv run python3 -c "
from football_orient_pose.evaluation import (
    compute_pck, compute_oks, compute_mpjpe_2d, joint_detection_report
)
print('OK')
"
```
Esperado: `OK`

- [ ] **Step 5.3: Commit**

```bash
git add src/football_orient_pose/evaluation/__init__.py
git commit -m "feat(evaluation): export all new metric symbols from __init__"
```

---

## Task 6: Criar `evaluate.py` — CLI batch

**Files:**
- Create: `src/football_orient_pose/evaluation/evaluate.py`

- [ ] **Step 6.1: Criar o arquivo**

```python
"""CLI de avaliação batch: roda um estimador em um split e calcula todas as métricas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from football_orient_pose.estimators import RTMPoseEstimator
from football_orient_pose.evaluation import (
    compute_mpjpe_2d,
    compute_oks,
    compute_pck,
    compute_pdj,
    joint_detection_report,
)
from football_orient_pose.utils.data_io import load_clip_image, load_keypoints_2d


def _build_estimator(model_name: str, device: str):
    if model_name == "rtmpose":
        return RTMPoseEstimator(device=device)
    raise ValueError(f"Modelo desconhecido: '{model_name}'. Opções: rtmpose")


def _run_inference(estimator, clip_ids: list[str], data_dir: Path):
    predictions, targets = [], []
    for clip_id in clip_ids:
        clip_dir = data_dir / "train" / clip_id
        for frame_idx in range(1, 21):
            image = load_clip_image(clip_dir, frame_idx)
            predictions.append(estimator.predict_h3wb(image))
            targets.append(
                load_keypoints_2d(clip_dir / "posture" / f"{frame_idx:03d}.json")
            )
    return np.asarray(predictions, np.float32), np.asarray(targets, np.float32)


def _print_table(result: dict) -> None:
    w = 50
    print(f"\n{'='*w}")
    print(f"  Model: {result['model']}  |  Split: {result['split']}")
    print(f"  Clips: {result['n_clips']}  |  Frames: {result['n_frames']}")
    print(f"{'='*w}")
    print(f"  PDJ@0.5     {result['pdj']['global']*100:>7.2f}%")
    print(f"  PCK@0.2     {result['pck']['global']*100:>7.2f}%")
    print(f"  OKS         {result['oks']['global_oks']*100:>7.2f}%")
    print(f"  AP50        {result['oks']['ap50']*100:>7.2f}%")
    print(f"  AP75        {result['oks']['ap75']*100:>7.2f}%")
    print(f"  mAP         {result['oks']['ap']*100:>7.2f}%")
    print(f"  MPJPE-2D    {result['mpjpe_2d']['global_px']:>7.2f} px")
    print(f"  F1-macro    {result['f1_macro']*100:>7.2f}%")
    print(f"{'='*w}\n")


def evaluate(
    model_name: str,
    split: str,
    data_dir: Path,
    split_config: Path,
    output_dir: Path,
    device: str = "cuda",
) -> dict:
    split_data = json.loads(split_config.read_text())
    if split not in split_data:
        raise ValueError(f"Split '{split}' não encontrado em {split_config}")
    clip_ids: list[str] = split_data[split]

    estimator = _build_estimator(model_name, device)
    pred, gt = _run_inference(estimator, clip_ids, data_dir)

    pdj = compute_pdj(pred, gt, threshold=0.5)
    pck = compute_pck(pred, gt, threshold=0.2)
    oks = compute_oks(pred, gt)
    mpjpe = compute_mpjpe_2d(pred, gt)
    det = joint_detection_report(pred, gt, threshold=0.5)

    result = {
        "model": model_name,
        "split": split,
        "n_clips": len(clip_ids),
        "n_frames": int(pred.shape[0]),
        "pdj": {"global": round(float(pdj.global_score), 6), "per_group": pdj.per_group},
        "pck": {"global": round(float(pck.global_score), 6), "per_group": pck.per_group},
        "oks": {
            "global_oks": round(float(oks.global_oks), 6),
            "ap": round(float(oks.ap), 6),
            "ap50": round(float(oks.ap50), 6),
            "ap75": round(float(oks.ap75), 6),
        },
        "mpjpe_2d": {
            "global_px": round(float(mpjpe.global_mpjpe), 4),
            "per_group": {k: round(v, 4) for k, v in mpjpe.per_group.items()},
        },
        "f1_macro": round(float(det.f1_macro), 6),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{model_name}_{split}.json"
    out_file.write_text(json.dumps(result, indent=2))
    print(f"Resultados salvos em {out_file}")
    _print_table(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia estimador de pose no 3DSP")
    parser.add_argument("--model", default="rtmpose", choices=["rtmpose"])
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--data-dir", type=Path, default=Path("data/3dsp"))
    parser.add_argument("--split-config", type=Path, default=Path("configs/split.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    evaluate(
        model_name=args.model,
        split=args.split,
        data_dir=args.data_dir,
        split_config=args.split_config,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.2: Verificar sintaxe e importações**

```bash
uv run python3 -c "from football_orient_pose.evaluation.evaluate import evaluate; print('OK')"
```
Esperado: `OK`

- [ ] **Step 6.3: Commit**

```bash
git add src/football_orient_pose/evaluation/evaluate.py
git commit -m "feat(evaluation): add evaluate.py CLI batch with all metrics"
```

---

## Task 7: Criar `training_monitor.py`

**Files:**
- Create: `src/football_orient_pose/evaluation/training_monitor.py`

- [ ] **Step 7.1: Criar o arquivo**

```python
"""Monitoramento de métricas por época para fine-tuning (EPIC 5)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    val_pdj: float | None = None
    val_pck: float | None = None
    val_oks: float | None = None


class TrainingMonitor:
    """Registra métricas por época e gera plots de loss e métricas."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.history: list[EpochRecord] = []

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_pdj: float | None = None,
        val_pck: float | None = None,
        val_oks: float | None = None,
    ) -> None:
        self.history.append(EpochRecord(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_pdj=val_pdj,
            val_pck=val_pck,
            val_oks=val_oks,
        ))

    def save(self, output_dir: str | Path) -> None:
        """Salva history.json e gera loss_curve.png e metrics_curve.png."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "history.json").write_text(
            json.dumps([asdict(r) for r in self.history], indent=2)
        )
        self._plot_loss_curve(out)
        self._plot_metrics_curve(out)

    def _plot_loss_curve(self, out: Path) -> None:
        import matplotlib.pyplot as plt

        epochs = [r.epoch for r in self.history]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, [r.train_loss for r in self.history], label="train")
        ax.plot(epochs, [r.val_loss for r in self.history], label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{self.model_name} — Loss")
        ax.legend()
        fig.savefig(out / "loss_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_metrics_curve(self, out: Path) -> None:
        import matplotlib.pyplot as plt

        epochs = [r.epoch for r in self.history]
        fig, ax = plt.subplots(figsize=(8, 5))
        for attr, label in [("val_pdj", "PDJ"), ("val_pck", "PCK"), ("val_oks", "OKS")]:
            vals = [getattr(r, attr) for r in self.history if getattr(r, attr) is not None]
            if vals:
                ax.plot(epochs[: len(vals)], vals, label=label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_title(f"{self.model_name} — Métricas por Época")
        ax.legend()
        fig.savefig(out / "metrics_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
```

- [ ] **Step 7.2: Verificar importação**

```bash
uv run python3 -c "from football_orient_pose.evaluation.training_monitor import TrainingMonitor; print('OK')"
```
Esperado: `OK`

- [ ] **Step 7.3: Commit**

```bash
git add src/football_orient_pose/evaluation/training_monitor.py
git commit -m "feat(evaluation): add TrainingMonitor for per-epoch loss and metrics (EPIC 5)"
```

---

## Task 8: Atualizar notebook `01_rtmpose_validation.ipynb`

**Files:**
- Modify: `notebooks/01_rtmpose_validation.ipynb`

O notebook tem atualmente 17 células (índices 0–16). As novas células são inseridas após a célula 15 (PDJ validation).

- [ ] **Step 8.1: Adicionar importações das novas métricas**

Adicionar uma célula de código após a célula 15 (antes do critério de aceitação):

```python
from football_orient_pose.evaluation import (
    compute_pck,
    compute_oks,
    compute_mpjpe_2d,
    joint_detection_report,
)
```

- [ ] **Step 8.2: Adicionar célula PCK**

```python
# PCK@0.2
pck = compute_pck(predictions, targets, threshold=0.2)
print(f"PCK@0.2: {pck.global_score * 100:.2f}%  (frames válidos: {pck.valid_frames})")
print("\nPCK@0.2 por grupo:")
for group, score in pck.per_group.items():
    print(f"  {group:<12}: {score * 100:.2f}%")
```

- [ ] **Step 8.3: Adicionar célula OKS / AP**

```python
# OKS e AP (padrão COCO)
oks_result = compute_oks(predictions, targets)
print(f"OKS:     {oks_result.global_oks * 100:.2f}%")
print(f"AP50:    {oks_result.ap50 * 100:.2f}%")
print(f"AP75:    {oks_result.ap75 * 100:.2f}%")
print(f"mAP:     {oks_result.ap * 100:.2f}%")
print("\nmAP por limiar:")
for thr, ap in sorted(oks_result.ap_per_threshold.items()):
    print(f"  @{thr:.2f}  {ap * 100:.2f}%")
```

- [ ] **Step 8.4: Adicionar célula MPJPE-2D**

```python
# MPJPE-2D (erro médio em pixels)
mpjpe = compute_mpjpe_2d(predictions, targets)
print(f"MPJPE-2D global: {mpjpe.global_mpjpe:.2f} px")
print("\nErro por grupo (px):")
for group, err in mpjpe.per_group.items():
    print(f"  {group:<12}: {err:.2f} px")
```

- [ ] **Step 8.5: Adicionar célula F1 por joint**

```python
# Detection Report — F1 por joint
det = joint_detection_report(predictions, targets, threshold=0.5)
print(f"F1-macro: {det.f1_macro * 100:.2f}%")
print("\nF1 por grupo anatômico:")
for group, score in det.per_group.items():
    print(f"  {group:<12}: {score * 100:.2f}%")
```

- [ ] **Step 8.6: Adicionar célula — bar chart por grupo**

```python
import matplotlib.pyplot as plt

groups = list(pdj.per_group.keys())
pdj_vals  = [pdj.per_group[g] * 100  for g in groups]
pck_vals  = [pck.per_group[g] * 100  for g in groups]
f1_vals   = [det.per_group[g] * 100  for g in groups]

x = np.arange(len(groups))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width, pdj_vals, width, label="PDJ@0.5")
ax.bar(x,          pck_vals, width, label="PCK@0.2")
ax.bar(x + width,  f1_vals,  width, label="F1@0.5")
ax.set_xticks(x)
ax.set_xticklabels(groups, rotation=20)
ax.set_ylabel("Score (%)")
ax.set_title("RTMPose — PDJ / PCK / F1 por grupo anatômico")
ax.set_ylim(0, 105)
ax.legend()
fig.tight_layout()
plt.show()
```

- [ ] **Step 8.7: Adicionar célula — heatmap de erro MPJPE por joint**

```python
from football_orient_pose.utils.keypoint_mapping import H3WB17_NAMES

fig, ax = plt.subplots(figsize=(12, 3))
im = ax.imshow(mpjpe.per_joint[np.newaxis, :], cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(17))
ax.set_xticklabels(H3WB17_NAMES, rotation=45, ha="right", fontsize=8)
ax.set_yticks([])
ax.set_title("MPJPE-2D por joint (px) — RTMPose zero-shot")
plt.colorbar(im, ax=ax, label="px")
fig.tight_layout()
plt.show()
```

- [ ] **Step 8.8: Adicionar célula — tabela resumo**

```python
print("\n" + "="*55)
print(f"  {'Métrica':<20}  {'Score':>10}  {'Ref (paper)':>12}")
print("="*55)
print(f"  {'PDJ@0.5':<20}  {pdj.global_score*100:>9.2f}%  {'89.51%':>12}")
print(f"  {'PCK@0.2':<20}  {pck.global_score*100:>9.2f}%  {'—':>12}")
print(f"  {'OKS':<20}  {oks_result.global_oks*100:>9.2f}%  {'—':>12}")
print(f"  {'AP50':<20}  {oks_result.ap50*100:>9.2f}%  {'—':>12}")
print(f"  {'mAP@[.5:.95]':<20}  {oks_result.ap*100:>9.2f}%  {'—':>12}")
print(f"  {'MPJPE-2D':<20}  {mpjpe.global_mpjpe:>9.2f}px  {'—':>12}")
print(f"  {'F1-macro':<20}  {det.f1_macro*100:>9.2f}%  {'—':>12}")
print("="*55)
```

- [ ] **Step 8.9: Commit do notebook (sem outputs)**

```bash
git add notebooks/01_rtmpose_validation.ipynb
git commit -m "feat(notebook): add PCK, OKS, MPJPE-2D, F1 cells to RTMPose validation"
```

---

## Task 9: Fechar issue #11 e rodar smoke test

**Files:** nenhum

- [ ] **Step 9.1: Rodar todos os testes**

```bash
uv run pytest tests/ -v
```
Esperado: todos os testes passam.

- [ ] **Step 9.2: Fechar issue #11**

```bash
gh issue close 11 --comment "RTMPoseEstimator implementado em src/football_orient_pose/estimators/rtmpose.py. 9 testes unitários passando. Todas as métricas (PDJ, PCK, OKS, MPJPE-2D, F1) implementadas e exportadas."
```

- [ ] **Step 9.3: Rodar smoke test no notebook (FULL_RUN=False, 2 clips)**

Abrir `notebooks/01_rtmpose_validation.ipynb` e executar todas as células sequencialmente com `FULL_RUN = False`.

Critérios:
- Células de setup: sem erros
- `kp_coco.shape == (17, 3)` ✓
- `kp_batch.shape == (5, 17, 3)` ✓
- PDJ, PCK, OKS, MPJPE-2D, F1 calculados sem erros
- Bar chart e heatmap renderizados

---

## Task 10: Full run e fechamento das issues

**Files:** nenhum

- [ ] **Step 10.1: Alterar `FULL_RUN = True` na célula 15 do notebook**

Editar a célula 15:
```python
FULL_RUN = True  # era False
MAX_CLIPS = None
```

- [ ] **Step 10.2: Executar o full run (200 clips × 20 frames = 4.000 frames)**

Executar a célula 15 em diante. Aguardar conclusão (CUDA).

Critério de aceitação:
- `PDJ@0.5` dentro de ±2pp de 89.51% (ou desvio documentado na célula 16)

- [ ] **Step 10.3: Rodar CLI de avaliação no split val**

```bash
uv run python3 -m football_orient_pose.evaluation.evaluate \
  --model rtmpose --split val \
  --data-dir data/3dsp \
  --split-config configs/split.json \
  --output-dir results/tables
```

Verificar que `results/tables/rtmpose_val.json` foi criado e a tabela foi impressa.

- [ ] **Step 10.4: Fechar issues #12, #10 e #22, #23, #24**

```bash
gh issue close 12 --comment "PDJ@0.5 validado no full run (4.000 frames). Notebook executável end-to-end com PCK, OKS, MPJPE-2D, F1."
gh issue close 10 --comment "RTMPoseEstimator validado com todas as métricas padrão de 2D pose estimation. PDJ reproduz valor do paper (±2%)."
gh issue close 22 --comment "compute_pck implementado em metrics.py com PCKResult. 3 testes unitários."
gh issue close 23 --comment "compute_oks implementado com OKSResult (global_oks, AP50, AP75, mAP). 3 testes unitários."
gh issue close 24 --comment "evaluate.py CLI batch implementado — roda via python -m football_orient_pose.evaluation.evaluate."
```

- [ ] **Step 10.5: Commit final do notebook com outputs**

```bash
git add notebooks/01_rtmpose_validation.ipynb results/tables/
git commit -m "feat(validation): RTMPose full run — all metrics computed and saved"
```
