from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "evaluation"))

from eval_detectors import (
    evaluate_detector,
    load_predictions,
    predict_clips,
    save_predictions,
)

from football_orient_pose.detection import Detection, Detector


class _FixedDetector(Detector):
    """Mock: devolve sempre uma caixa fixa com conf alta (sem baixar modelo)."""

    name = "fixed"

    def __init__(self, box=(10.0, 10.0, 50.0, 90.0)) -> None:
        self._box = np.asarray(box, dtype=np.float32)

    def detect(self, image: np.ndarray) -> list[Detection]:
        return [(self._box.copy(), 0.95)]


def _make_frames(root: Path, gt: dict[str, dict[int, np.ndarray]]) -> None:
    for clip_id, frames in gt.items():
        for frame_idx in frames:
            d = root / clip_id / "img"
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{frame_idx:03d}.jpg"), np.zeros((720, 1280, 3), dtype=np.uint8))


def _gt() -> dict[str, dict[int, np.ndarray]]:
    return {"example_01": {1: np.array([[10, 10, 50, 90]], np.float32),
                           2: np.array([[10, 10, 50, 90]], np.float32)}}


def test_predict_clips_runs_detector_over_frames(tmp_path: Path) -> None:
    gt = _gt()
    _make_frames(tmp_path, gt)
    preds = predict_clips(_FixedDetector(), tmp_path, gt)
    assert set(preds["example_01"]) == {1, 2}
    boxes, scores = preds["example_01"][1]
    assert boxes.shape == (1, 4) and scores.shape == (1,)


def test_evaluate_detector_perfect_predictions(tmp_path: Path) -> None:
    gt = _gt()
    _make_frames(tmp_path, gt)
    preds = predict_clips(_FixedDetector(), tmp_path, gt)  # caixa == GT
    r = evaluate_detector("fixed", preds, gt, conf=0.3)
    assert r["mAP"] == 1.0 and r["f1"] == 1.0
    assert r["tp"] == 2 and r["fp"] == 0 and r["fn"] == 0
    assert r["n_frames"] == 2 and r["n_gt"] == 2


def test_predictions_cache_roundtrip(tmp_path: Path) -> None:
    gt = _gt()
    _make_frames(tmp_path, gt)
    preds = predict_clips(_FixedDetector(), tmp_path, gt)
    path = tmp_path / "cache.json"
    save_predictions(preds, path)
    back = load_predictions(path)
    np.testing.assert_array_equal(back["example_01"][1][0], preds["example_01"][1][0])
    # métricas idênticas a partir do cache
    assert evaluate_detector("fixed", back, gt, 0.3)["mAP"] == 1.0
