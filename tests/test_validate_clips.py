from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from football_orient_pose.clip_extractor import write_clip
from validate_clips import validate_clip

_META = {
    "source_video": "x.mp4", "game": "Teste", "label": "Finalização",
    "start_ms": 0, "end_ms": 2000, "fps": 25, "step_ms": 40,
}


def _make_clip(root: Path, clip_id: str, n_frames: int = 20, height: int = 720):
    frames = [np.zeros((height, 1280, 3), dtype=np.uint8) for _ in range(n_frames)]
    return write_clip(clip_id, frames, dict(_META), root=root)


def test_valid_clip_has_no_errors(tmp_path: Path) -> None:
    clip = _make_clip(tmp_path, "brazil_01")
    assert validate_clip(clip, n=20, min_height=720) == []


def test_wrong_frame_count_is_rejected(tmp_path: Path) -> None:
    clip = _make_clip(tmp_path, "brazil_01", n_frames=19)
    errors = validate_clip(clip, n=20)
    assert any("19 frames" in e for e in errors)


def test_low_resolution_is_rejected(tmp_path: Path) -> None:
    clip = _make_clip(tmp_path, "brazil_01", height=480)
    errors = validate_clip(clip, n=20, min_height=720)
    assert any("altura" in e for e in errors)


def test_missing_info_field_is_rejected(tmp_path: Path) -> None:
    clip = _make_clip(tmp_path, "brazil_01")
    # corrompe o info.ini removendo um campo obrigatório
    info = (clip / "info.ini").read_text(encoding="utf-8")
    (clip / "info.ini").write_text(
        "\n".join(l for l in info.splitlines() if not l.startswith("game")) + "\n",
        encoding="utf-8",
    )
    errors = validate_clip(clip, n=20)
    assert any("info.ini sem campos" in e for e in errors)
