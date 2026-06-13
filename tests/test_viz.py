from __future__ import annotations

import numpy as np
import pytest

from football_orient_pose.utils.viz import draw_boxes, draw_skeleton, frames_to_gif


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


def test_frames_to_gif_writes_animated(tmp_path) -> None:
    from PIL import Image

    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, size=(40, 60, 3), dtype=np.uint8) for _ in range(4)]
    out = frames_to_gif(frames, tmp_path / "anim.gif", fps=5)
    assert out.exists()
    with Image.open(out) as gif:
        assert gif.is_animated and gif.n_frames == 4


def test_frames_to_gif_resizes_width(tmp_path) -> None:
    from PIL import Image

    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
    out = frames_to_gif(frames, tmp_path / "small.gif", target_width=50)
    with Image.open(out) as gif:
        assert gif.size == (50, 50)  # 100x100 → 50x50 (aspecto preservado)


def test_frames_to_gif_empty_raises(tmp_path) -> None:
    with pytest.raises(ValueError):
        frames_to_gif([], tmp_path / "x.gif")
