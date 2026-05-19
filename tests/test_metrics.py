from __future__ import annotations

import numpy as np
import pytest

from football_orient_pose.evaluation import (
    compute_pdj,
    compute_pck,
    pdj_auc,
    pdj_curve,
    PCKResult,
)


def _make_h3wb_keypoints() -> np.ndarray:
    keypoints = np.zeros((17, 2), dtype=np.float32)
    keypoints[:, 0] = np.arange(17, dtype=np.float32)
    keypoints[:, 1] = np.arange(17, dtype=np.float32) * 2
    keypoints[0] = [0.0, 0.0]
    keypoints[8] = [0.0, 10.0]
    return keypoints


def test_compute_pdj_returns_perfect_score_for_identical_keypoints() -> None:
    target = _make_h3wb_keypoints()

    result = compute_pdj(target, target)

    assert result.threshold == 0.5
    assert result.global_score == 1.0
    assert result.valid_frames == 1
    np.testing.assert_array_equal(result.per_joint, np.ones(17, dtype=np.float32))
    assert result.per_group["head"] == 1.0


def test_compute_pdj_uses_torso_normalized_distance() -> None:
    target = _make_h3wb_keypoints()
    predicted = target.copy()
    predicted[10] = target[10] + [4.9, 0.0]
    predicted[13] = target[13] + [5.1, 0.0]

    result = compute_pdj(predicted, target, threshold=0.5)

    assert result.per_joint[10] == 1.0
    assert result.per_joint[13] == 0.0
    assert result.global_score == pytest.approx(16 / 17)
    assert result.per_group["head"] == 1.0
    assert result.per_group["wrist"] == 0.5


def test_compute_pdj_accepts_batches_and_ignores_invalid_torso_frames() -> None:
    target = np.stack([_make_h3wb_keypoints(), np.zeros((17, 2), dtype=np.float32)])
    predicted = target.copy()

    result = compute_pdj(predicted, target)

    assert result.valid_frames == 1
    assert result.global_score == 1.0


def test_compute_pdj_raises_for_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="shape"):
        compute_pdj(np.zeros((16, 2)), np.zeros((16, 2)))


def test_pdj_curve_and_auc_are_normalized() -> None:
    target = _make_h3wb_keypoints()
    predicted = target.copy()
    thresholds = np.array([0.0, 0.5], dtype=np.float32)

    curve_thresholds, scores = pdj_curve(predicted, target, thresholds=thresholds)
    auc = pdj_auc(predicted, target, thresholds=thresholds)

    np.testing.assert_array_equal(curve_thresholds, thresholds)
    np.testing.assert_array_equal(scores, np.array([0.0, 1.0], dtype=np.float32))
    assert auc == 0.5


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
