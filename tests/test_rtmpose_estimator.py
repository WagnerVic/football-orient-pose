from __future__ import annotations

import builtins

import numpy as np
import pytest

from football_orient_pose.estimators import RTMPoseEstimator


class FakeRTMPoseModel:
    def __init__(self, keypoints: np.ndarray, scores: np.ndarray) -> None:
        self.keypoints = keypoints
        self.scores = scores
        self.calls = 0

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.calls += 1
        return self.keypoints, self.scores


def _make_keypoints(offset: float = 0.0) -> np.ndarray:
    keypoints = np.zeros((17, 2), dtype=np.float32)
    keypoints[:, 0] = np.arange(17, dtype=np.float32) + offset
    keypoints[:, 1] = np.arange(100, 117, dtype=np.float32) + offset
    return keypoints


def test_rtmpose_estimator_predicts_coco17_with_scores() -> None:
    fake_model = FakeRTMPoseModel(
        keypoints=_make_keypoints()[np.newaxis, ...],
        scores=np.full((1, 17), 0.75, dtype=np.float32),
    )
    estimator = RTMPoseEstimator(pose_model=fake_model)
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    keypoints = estimator.predict(image)

    assert estimator.name == "rtmpose-x"
    assert keypoints.shape == (17, 3)
    assert keypoints.dtype == np.float32
    np.testing.assert_array_equal(keypoints[:, :2], _make_keypoints())
    np.testing.assert_array_equal(keypoints[:, 2], np.full(17, 0.75, dtype=np.float32))


def test_rtmpose_estimator_predicts_batch() -> None:
    fake_model = FakeRTMPoseModel(
        keypoints=_make_keypoints()[np.newaxis, ...],
        scores=np.full((1, 17), 0.75, dtype=np.float32),
    )
    estimator = RTMPoseEstimator(pose_model=fake_model)
    images = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]

    keypoints = estimator.predict_batch(images)

    assert keypoints.shape == (3, 17, 3)
    assert fake_model.calls == 3


def test_rtmpose_estimator_returns_empty_batch_for_empty_input() -> None:
    estimator = RTMPoseEstimator(
        pose_model=FakeRTMPoseModel(
            keypoints=_make_keypoints()[np.newaxis, ...],
            scores=np.full((1, 17), 0.75, dtype=np.float32),
        )
    )

    keypoints = estimator.predict_batch([])

    assert keypoints.shape == (0, 17, 3)
    assert keypoints.dtype == np.float32


def test_rtmpose_estimator_selects_instance_closest_to_crop_center() -> None:
    far_keypoints = _make_keypoints(offset=200.0)
    near_keypoints = np.full((17, 2), 50.0, dtype=np.float32)
    fake_model = FakeRTMPoseModel(
        keypoints=np.stack([far_keypoints, near_keypoints]),
        scores=np.stack(
            [
                np.full(17, 0.2, dtype=np.float32),
                np.full(17, 0.9, dtype=np.float32),
            ]
        ),
    )
    estimator = RTMPoseEstimator(pose_model=fake_model)
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    keypoints = estimator.predict(image)

    np.testing.assert_array_equal(keypoints[:, :2], near_keypoints)
    np.testing.assert_array_equal(keypoints[:, 2], np.full(17, 0.9, dtype=np.float32))


def test_rtmpose_estimator_raises_clear_error_without_rtmlib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def import_without_rtmlib(name: str, *args, **kwargs):
        if name == "rtmlib":
            raise ImportError("No module named 'rtmlib'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_rtmlib)

    with pytest.raises(ImportError, match="rtmlib"):
        RTMPoseEstimator()
