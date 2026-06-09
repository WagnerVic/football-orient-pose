from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from football_orient_pose.clip_extractor import cut_clip, write_clip
from football_orient_pose.utils.data_io import load_clip_image, load_clip_info

EXAMPLE = Path("data/examples/test_00001.mp4")
pytestmark = pytest.mark.skipif(not EXAMPLE.exists(), reason="data/examples ausente")


def test_cut_clip_returns_exactly_n_frames() -> None:
    frames = cut_clip(EXAMPLE, start_ms=0, end_ms=2000, n=20)
    assert len(frames) == 20
    assert all(f.ndim == 3 for f in frames)


def test_cut_clip_raises_when_interval_too_short() -> None:
    # 0..1ms não rende 20 frames
    with pytest.raises(ValueError, match="intervalo curto"):
        cut_clip(EXAMPLE, start_ms=0, end_ms=1, n=20)


@pytest.mark.parametrize("bad", [(0, 0), (100, 50)])
def test_cut_clip_rejects_invalid_interval(bad: tuple[float, float]) -> None:
    with pytest.raises(ValueError):
        cut_clip(EXAMPLE, start_ms=bad[0], end_ms=bad[1], n=5)


def test_cut_clip_rejects_non_positive_n() -> None:
    with pytest.raises(ValueError, match="n deve ser"):
        cut_clip(EXAMPLE, start_ms=0, end_ms=2000, n=0)


def test_write_clip_creates_structure_and_loaders_read_it(tmp_path: Path) -> None:
    frames = [np.full((720, 1280, 3), i, dtype=np.uint8) for i in range(1, 21)]
    meta = {
        "source_video": str(EXAMPLE), "game": "Teste", "label": "Finalização",
        "start_ms": 0, "end_ms": 2000, "fps": 25, "step_ms": 40,
    }
    clip_dir = write_clip("example_01", frames, meta, root=tmp_path)

    # estrutura
    assert (clip_dir / "info.ini").exists()
    imgs = sorted((clip_dir / "img").glob("*.jpg"))
    assert [p.name for p in imgs] == [f"{i:03d}.jpg" for i in range(1, 21)]

    # loaders existentes leem de volta
    img = load_clip_image(clip_dir, 1)
    assert img.shape == (720, 1280, 3)
    info = load_clip_info(clip_dir)
    assert info["id"] == "example_01"
    assert info["num_frames"] == "20"
    assert info["label"] == "Finalização"
