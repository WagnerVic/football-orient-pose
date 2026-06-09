from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "clips"))

from validate_clips import validate_clip

from football_orient_pose.clip_extractor import write_clip

_META = {
    "source_video": "x.mp4", "game": "Teste", "label": "Finalização",
    "start_ms": 0, "end_ms": 2000, "fps": 25, "step_ms": 40,
}


def _make_clip(root: Path, clip_id: str, n_frames: int = 20, height: int = 720):
    frames = [np.zeros((height, 1280, 3), dtype=np.uint8) for _ in range(n_frames)]
    return write_clip(clip_id, frames, dict(_META), root=root)


def test_valid_clip_has_no_errors(tmp_path: Path) -> None:
    clip = _make_clip(tmp_path, "brazil_01")
    assert validate_clip(clip) == []


def test_any_consistent_frame_count_passes(tmp_path: Path) -> None:
    # nº de frames livre: 10 frames + info.ini coerente → válido (sem --n fixo)
    clip = _make_clip(tmp_path, "brazil_10f", n_frames=10)
    assert validate_clip(clip) == []


def test_frame_count_mismatch_is_rejected(tmp_path: Path) -> None:
    # apaga 1 frame depois de escrito: img(19) ≠ num_frames(20) no info.ini
    clip = _make_clip(tmp_path, "brazil_01", n_frames=20)
    (clip / "img" / "020.jpg").unlink()
    errors = validate_clip(clip)
    assert any("num_frames=20" in e for e in errors)


def test_noncontiguous_frames_are_rejected(tmp_path: Path) -> None:
    # mantém a contagem (20) mas renomeia para 011..030 → não contíguo 001..020
    clip = _make_clip(tmp_path, "brazil_01", n_frames=20)
    img = clip / "img"
    for p in sorted(img.glob("*.jpg"), reverse=True):
        p.rename(img / f"{int(p.stem) + 10:03d}.jpg")
    errors = validate_clip(clip)
    assert any("não contígua" in e for e in errors)


def test_low_resolution_is_rejected(tmp_path: Path) -> None:
    clip = _make_clip(tmp_path, "brazil_01", height=480)
    errors = validate_clip(clip, min_height=720)
    assert any("altura" in e for e in errors)


def test_missing_info_field_is_rejected(tmp_path: Path) -> None:
    clip = _make_clip(tmp_path, "brazil_01")
    info = (clip / "info.ini").read_text(encoding="utf-8")
    (clip / "info.ini").write_text(
        "\n".join(ln for ln in info.splitlines() if not ln.startswith("game")) + "\n",
        encoding="utf-8",
    )
    errors = validate_clip(clip)
    assert any("info.ini sem campos" in e for e in errors)
