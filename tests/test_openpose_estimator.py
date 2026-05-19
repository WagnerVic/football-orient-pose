from __future__ import annotations

import numpy as np
import pytest

from football_orient_pose.estimators import OpenPoseEstimator
from football_orient_pose.estimators.openpose import OPENPOSE_TO_COCO17


class FakeOpenPoseModel:
    """Mock que retorna keypoints e scores fixos sem carregar o modelo Caffe real."""

    def __init__(self, keypoints: np.ndarray, scores: np.ndarray) -> None:
        self.keypoints = keypoints
        self.scores = scores
        self.calls = 0

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.calls += 1
        return self.keypoints, self.scores


def _make_keypoints(offset: float = 0.0) -> np.ndarray:
    kp = np.zeros((17, 2), dtype=np.float32)
    kp[:, 0] = np.arange(17, dtype=np.float32) + offset
    kp[:, 1] = np.arange(100, 117, dtype=np.float32) + offset
    return kp


def test_openpose_estimator_predicts_coco17_with_scores() -> None:
    fake_model = FakeOpenPoseModel(
        keypoints=_make_keypoints(),
        scores=np.full(17, 0.6, dtype=np.float32),
    )
    estimator = OpenPoseEstimator(pose_model=fake_model)
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    keypoints = estimator.predict(image)

    assert estimator.name == "openpose"
    assert keypoints.shape == (17, 3)
    assert keypoints.dtype == np.float32
    np.testing.assert_array_equal(keypoints[:, :2], _make_keypoints())
    np.testing.assert_array_equal(keypoints[:, 2], np.full(17, 0.6, dtype=np.float32))


def test_openpose_estimator_predicts_batch() -> None:
    fake_model = FakeOpenPoseModel(
        keypoints=_make_keypoints(),
        scores=np.full(17, 0.6, dtype=np.float32),
    )
    estimator = OpenPoseEstimator(pose_model=fake_model)
    images = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]

    keypoints = estimator.predict_batch(images)

    assert keypoints.shape == (5, 17, 3)
    assert fake_model.calls == 5


def test_openpose_estimator_returns_empty_batch() -> None:
    estimator = OpenPoseEstimator(
        pose_model=FakeOpenPoseModel(
            keypoints=_make_keypoints(),
            scores=np.full(17, 0.6, dtype=np.float32),
        )
    )

    keypoints = estimator.predict_batch([])

    assert keypoints.shape == (0, 17, 3)
    assert keypoints.dtype == np.float32


def test_openpose_estimator_name() -> None:
    estimator = OpenPoseEstimator(
        pose_model=FakeOpenPoseModel(
            keypoints=_make_keypoints(),
            scores=np.full(17, 0.6, dtype=np.float32),
        )
    )
    assert estimator.name == "openpose"


def test_openpose_estimator_raises_when_paths_empty() -> None:
    with pytest.raises(ValueError, match="prototxt_path"):
        OpenPoseEstimator(prototxt_path="", caffemodel_path="")


def test_openpose_to_coco17_mapping_covers_all_17_joints() -> None:
    assert len(OPENPOSE_TO_COCO17) == 17
    # Todos os índices OpenPose referenciados devem ser 0-17 (excluindo Neck=1)
    for coco_idx, op_idx in enumerate(OPENPOSE_TO_COCO17):
        assert 0 <= op_idx <= 17, f"COCO[{coco_idx}] → OP[{op_idx}] fora do range"
        assert op_idx != 1, f"COCO[{coco_idx}] mapeado para Neck (OP 1) — inválido"


def test_openpose_to_coco17_mapping_nose_is_first() -> None:
    # COCO[0] = Nose deve vir de OP[0] = Nose
    assert OPENPOSE_TO_COCO17[0] == 0


def test_openpose_to_coco17_mapping_no_duplicates() -> None:
    # Cada joint OpenPose deve aparecer no máximo uma vez
    assert len(OPENPOSE_TO_COCO17) == len(set(OPENPOSE_TO_COCO17))


def test_openpose_to_coco17_mapping_specific_joints() -> None:
    # Verifica joints críticos com lateralidade invertida entre formatos
    assert OPENPOSE_TO_COCO17[5] == 5    # COCO  5 LShoulder ← OP  5
    assert OPENPOSE_TO_COCO17[6] == 2    # COCO  6 RShoulder ← OP  2
    assert OPENPOSE_TO_COCO17[11] == 11  # COCO 11 LHip      ← OP 11
    assert OPENPOSE_TO_COCO17[12] == 8   # COCO 12 RHip      ← OP  8
    assert OPENPOSE_TO_COCO17[1] == 15   # COCO  1 LEye      ← OP 15
    assert OPENPOSE_TO_COCO17[2] == 14   # COCO  2 REye      ← OP 14
