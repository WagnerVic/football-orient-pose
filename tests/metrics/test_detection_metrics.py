from __future__ import annotations

import numpy as np
import pytest

from football_orient_pose.evaluation.detection_metrics import (
    build_coco_dt,
    build_coco_gt,
    cocoeval_stats,
    iou_matrix,
    precision_recall_f1,
)


def test_iou_matrix_known_values() -> None:
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]], dtype=np.float32)
    iou = iou_matrix(a, b)
    assert iou.shape == (1, 3)
    np.testing.assert_allclose(iou[0, 0], 1.0)  # idêntica
    np.testing.assert_allclose(iou[0, 1], 25 / 175, rtol=1e-6)  # inter=25, union=175
    np.testing.assert_allclose(iou[0, 2], 0.0)  # disjunta


def test_iou_matrix_empty() -> None:
    assert iou_matrix(np.empty((0, 4)), np.array([[0, 0, 1, 1]])).shape == (0, 1)


def test_prf1_perfect_match() -> None:
    boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    r = precision_recall_f1(boxes, np.array([0.9, 0.8]), boxes)
    assert r["precision"] == 1.0 and r["recall"] == 1.0 and r["f1"] == 1.0
    assert r["tp"] == 2 and r["fp"] == 0 and r["fn"] == 0


def test_prf1_no_overlap() -> None:
    pred = np.array([[0, 0, 10, 10]], dtype=np.float32)
    gt = np.array([[100, 100, 110, 110]], dtype=np.float32)
    r = precision_recall_f1(pred, np.array([0.9]), gt)
    assert r["tp"] == 0 and r["fp"] == 1 and r["fn"] == 1
    assert r["precision"] == 0.0 and r["recall"] == 0.0 and r["f1"] == 0.0


def test_prf1_partial_and_score_filter() -> None:
    # 2 GT; 1 predição certa (alta conf) + 1 falsa abaixo do score_thr (ignorada)
    gt = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=np.float32)
    pred = np.array([[0, 0, 10, 10], [200, 200, 210, 210]], dtype=np.float32)
    r = precision_recall_f1(pred, np.array([0.9, 0.1]), gt, score_thr=0.5)
    assert r["tp"] == 1 and r["fp"] == 0 and r["fn"] == 1
    assert r["precision"] == 1.0
    assert r["recall"] == pytest.approx(0.5)


def test_prf1_one_gt_no_double_count() -> None:
    # 2 predições sobre o MESMO gt → 1 TP + 1 FP (não conta o gt duas vezes)
    gt = np.array([[0, 0, 10, 10]], dtype=np.float32)
    pred = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32)
    r = precision_recall_f1(pred, np.array([0.9, 0.8]), gt)
    assert r["tp"] == 1 and r["fp"] == 1 and r["fn"] == 0


def _toy_gt() -> dict[str, dict[int, np.ndarray]]:
    return {
        "example_01": {
            1: np.array([[10, 10, 50, 90], [100, 100, 140, 180]], dtype=np.float32),
            2: np.array([[30, 30, 70, 110]], dtype=np.float32),
        }
    }


def test_build_coco_gt_image_id_map() -> None:
    coco, id_map = build_coco_gt(_toy_gt())
    assert id_map == {("example_01", 1): 0, ("example_01", 2): 1}
    assert len(coco.dataset["images"]) == 2
    assert len(coco.dataset["annotations"]) == 3  # 2 + 1 caixas


def test_cocoeval_perfect_prediction_scores_high() -> None:
    gt = _toy_gt()
    coco, id_map = build_coco_gt(gt)
    # predição = GT, conf alta
    preds = {
        clip: {f: (boxes, np.ones(len(boxes), dtype=np.float32)) for f, boxes in frames.items()}
        for clip, frames in gt.items()
    }
    dt = build_coco_dt(preds, id_map)
    stats = cocoeval_stats(coco, dt)
    assert stats["mAP"] == pytest.approx(1.0, abs=1e-6)
    assert stats["AP50"] == pytest.approx(1.0, abs=1e-6)


def test_cocoeval_empty_predictions() -> None:
    coco, id_map = build_coco_gt(_toy_gt())
    stats = cocoeval_stats(coco, build_coco_dt({}, id_map))
    assert stats["mAP"] == 0.0 and stats["AR_100"] == 0.0
