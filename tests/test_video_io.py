from __future__ import annotations

from pathlib import Path

import pytest

from football_orient_pose.video_io import Frame, extract_frames

EXAMPLE = Path("data/examples/test_00001.mp4")
pytestmark = pytest.mark.skipif(not EXAMPLE.exists(), reason="data/examples ausente")


def test_extract_frames_returns_frames() -> None:
    frames = extract_frames(EXAMPLE, every_n=5)
    assert len(frames) > 0
    assert all(isinstance(f, Frame) for f in frames)
    assert frames[0].image.ndim == 3  # H×W×C


def test_extract_frames_indices_are_increasing_and_stepped() -> None:
    frames = extract_frames(EXAMPLE, every_n=5)
    indices = [f.index for f in frames]
    assert indices == sorted(indices)
    assert all(i % 5 == 0 for i in indices)


def test_extract_frames_all_when_no_sampling() -> None:
    all_frames = extract_frames(EXAMPLE)
    every2 = extract_frames(EXAMPLE, every_n=2)
    assert len(all_frames) >= len(every2) > 0


def test_extract_frames_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        extract_frames("data/examples/nao_existe.mp4")


def test_extract_frames_rejects_both_modes() -> None:
    with pytest.raises(ValueError, match="every_n OU target_fps"):
        extract_frames(EXAMPLE, every_n=5, target_fps=5)


@pytest.mark.parametrize("kwargs", [{"every_n": 0}, {"target_fps": 0}])
def test_extract_frames_rejects_non_positive(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        extract_frames(EXAMPLE, **kwargs)
