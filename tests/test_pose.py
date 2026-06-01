from __future__ import annotations

import numpy as np
import pytest

from football_orient_pose import BasePoseEstimator


class DummyPoseEstimator(BasePoseEstimator):
    @property
    def name(self) -> str:
        return "dummy"

    def predict(self, image: np.ndarray) -> np.ndarray:
        keypoints = np.zeros((17, 3), dtype=np.float32)
        keypoints[:, 0] = np.arange(17, dtype=np.float32)
        keypoints[:, 1] = np.arange(100, 117, dtype=np.float32)
        keypoints[:, 2] = 0.9
        return keypoints

    def predict_batch(self, images: list[np.ndarray]) -> np.ndarray:
        return np.stack([self.predict(image) for image in images])


def test_base_pose_estimator_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        BasePoseEstimator()


def test_subclass_predicts_coco17_keypoints() -> None:
    estimator = DummyPoseEstimator()
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    keypoints = estimator.predict(image)

    assert estimator.name == "dummy"
    assert keypoints.shape == (17, 3)
    assert keypoints.dtype == np.float32
    assert np.all(keypoints[:, 2] == 0.9)


def test_subclass_predicts_coco17_keypoints_in_batch() -> None:
    estimator = DummyPoseEstimator()
    images = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]

    keypoints = estimator.predict_batch(images)

    assert keypoints.shape == (3, 17, 3)


def test_predict_h3wb_converts_coco17_and_discards_confidence() -> None:
    estimator = DummyPoseEstimator()
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    keypoints = estimator.predict_h3wb(image)

    assert keypoints.shape == (17, 2)
    np.testing.assert_array_equal(keypoints[0], np.array([11.5, 111.5], dtype=np.float32))
    np.testing.assert_array_equal(keypoints[10], np.array([0.0, 100.0], dtype=np.float32))
    np.testing.assert_array_equal(keypoints[16], np.array([9.0, 109.0], dtype=np.float32))
