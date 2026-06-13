from __future__ import annotations

import numpy as np

from football_orient_pose.utils.viz import draw_boxes, draw_skeleton


def test_draw_boxes_keeps_shape_and_draws() -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = np.array([[10, 10, 40, 60]], dtype=np.float32)
    out = draw_boxes(img, boxes, color=(0, 255, 0))
    assert out.shape == img.shape and out.dtype == img.dtype
    assert out.sum() > 0  # desenhou algo
    assert img.sum() == 0  # não mexeu na original


def test_draw_boxes_empty_is_noop_copy() -> None:
    img = np.full((20, 20, 3), 7, dtype=np.uint8)
    out = draw_boxes(img, np.empty((0, 4), dtype=np.float32))
    assert np.array_equal(out, img) and out is not img


def test_draw_skeleton_keeps_shape_and_draws() -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    kps = np.random.default_rng(0).uniform(10, 90, size=(17, 2))
    out = draw_skeleton(img, kps)
    assert out.shape == img.shape and out.dtype == img.dtype
    assert out.sum() > 0  # desenhou ossos/juntas
    assert img.sum() == 0  # não mutou a original
