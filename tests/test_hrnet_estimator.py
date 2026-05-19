from __future__ import annotations

import builtins

import numpy as np
import pytest

from football_orient_pose.estimators import HRNetEstimator


class FakeHRNetModel:
    """Mock que retorna keypoints e scores fixos sem carregar o modelo ONNX real."""

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


def test_hrnet_estimator_predicts_coco17_with_scores() -> None:
    fake_model = FakeHRNetModel(
        keypoints=_make_keypoints(),
        scores=np.full(17, 0.8, dtype=np.float32),
    )
    estimator = HRNetEstimator(pose_model=fake_model)
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    keypoints = estimator.predict(image)

    assert estimator.name == "hrnet-w48"
    assert keypoints.shape == (17, 3)
    assert keypoints.dtype == np.float32
    np.testing.assert_array_equal(keypoints[:, :2], _make_keypoints())
    np.testing.assert_array_equal(keypoints[:, 2], np.full(17, 0.8, dtype=np.float32))


def test_hrnet_estimator_predicts_batch() -> None:
    fake_model = FakeHRNetModel(
        keypoints=_make_keypoints(),
        scores=np.full(17, 0.8, dtype=np.float32),
    )
    estimator = HRNetEstimator(pose_model=fake_model)
    images = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(4)]

    keypoints = estimator.predict_batch(images)

    assert keypoints.shape == (4, 17, 3)
    assert fake_model.calls == 4


def test_hrnet_estimator_returns_empty_batch() -> None:
    estimator = HRNetEstimator(
        pose_model=FakeHRNetModel(
            keypoints=_make_keypoints(),
            scores=np.full(17, 0.8, dtype=np.float32),
        )
    )

    keypoints = estimator.predict_batch([])

    assert keypoints.shape == (0, 17, 3)
    assert keypoints.dtype == np.float32


def test_hrnet_estimator_name() -> None:
    estimator = HRNetEstimator(
        pose_model=FakeHRNetModel(
            keypoints=_make_keypoints(),
            scores=np.full(17, 0.8, dtype=np.float32),
        )
    )
    assert estimator.name == "hrnet-w48"


def test_hrnet_estimator_raises_clear_error_without_onnxruntime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def import_without_ort(name: str, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("No module named 'onnxruntime'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_ort)

    with pytest.raises(ImportError, match="onnxruntime"):
        HRNetEstimator(model_path="some_model.onnx")


def test_hrnet_estimator_raises_when_model_path_empty() -> None:
    with pytest.raises(ValueError, match="model_path"):
        HRNetEstimator(model_path="")
