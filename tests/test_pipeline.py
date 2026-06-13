from __future__ import annotations

import numpy as np

from football_orient_pose.detection import Detection, Detector
from football_orient_pose.pipeline import PipelineResult, run_pipeline, select_finisher
from football_orient_pose.pose import BasePoseEstimator


class _TwoBoxDetector(Detector):
    """Mock: devolve duas caixas fixas (um 'finalizador' e um 'outro jogador')."""

    name = "two-box"

    def detect(self, image: np.ndarray) -> list[Detection]:
        far = np.array([1000.0, 100.0, 1040.0, 180.0], dtype=np.float32)  # outro
        finisher = np.array([400.0, 300.0, 460.0, 400.0], dtype=np.float32)
        return [(far, 0.9), (finisher, 0.85)]


class _CenterPose(BasePoseEstimator):
    """Mock: keypoints fixos no centro do crop (sem baixar modelo)."""

    name = "center-pose"

    def predict(self, image: np.ndarray) -> np.ndarray:
        kp = np.zeros((17, 3), dtype=np.float32)
        kp[:, 0] = 50.0
        kp[:, 1] = 50.0
        kp[:, 2] = 0.9
        return kp

    def predict_batch(self, images: list[np.ndarray]) -> np.ndarray:
        return np.stack([self.predict(im) for im in images])


def _frame() -> np.ndarray:
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def test_select_finisher_with_ref_picks_max_iou() -> None:
    boxes = np.array([[1000, 100, 1040, 180], [400, 300, 460, 400]], dtype=np.float32)
    ref = np.array([405, 305, 465, 405], dtype=np.float32)  # perto da 2ª
    assert select_finisher(boxes, ref) == 1


def test_select_finisher_without_ref_picks_largest() -> None:
    boxes = np.array([[0, 0, 10, 10], [0, 0, 100, 100]], dtype=np.float32)
    assert select_finisher(boxes, None) == 1


def test_run_pipeline_chains_and_reprojects() -> None:
    ref = np.array([405, 305, 465, 405], dtype=np.float32)
    res = run_pipeline(_frame(), _TwoBoxDetector(), _CenterPose(), ref_box=ref)
    assert isinstance(res, PipelineResult)
    # escolheu o finalizador (2ª caixa), não o jogador distante
    np.testing.assert_array_equal(res.finisher_box, [400, 300, 460, 400])
    assert res.crop.shape == (100, 100, 3)
    assert res.keypoints_crop.shape == (17, 2) and res.keypoints_frame.shape == (17, 2)
    assert res.all_boxes.shape == (2, 4)
    # o centro do crop (50,50) reprojeta para dentro da caixa do finalizador
    cx, cy = res.keypoints_frame[0]
    assert 400 <= cx <= 460 and 300 <= cy <= 400


def test_run_pipeline_reprojection_matches_crop_to_frame() -> None:
    from football_orient_pose.crop import crop_to_frame

    res = run_pipeline(_frame(), _TwoBoxDetector(), _CenterPose())
    expected = crop_to_frame(res.keypoints_crop, res.crop_params)
    np.testing.assert_allclose(res.keypoints_frame, expected, atol=1e-6)
