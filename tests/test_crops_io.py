from __future__ import annotations

import configparser
import json
from pathlib import Path

import numpy as np

from football_orient_pose.crop import CropParams
from football_orient_pose.crops_io import load_crop_params, write_crop_clip


def _sample(n: int = 3):
    crops = [np.full((100, 100, 3), i * 10, dtype=np.uint8) for i in range(1, n + 1)]
    params = [
        CropParams(x0=float(i), y0=2.0, scale=1.25, pad_top=0.0, pad_left=8.0) for i in range(n)
    ]
    bboxes = [np.array([i, i, i + 50, i + 80], dtype=np.float32) for i in range(n)]
    return crops, params, bboxes


def test_write_crop_clip_structure(tmp_path: Path) -> None:
    crops, params, bboxes = _sample(3)
    clip_dir = write_crop_clip(
        tmp_path, "example_01", crops, params, bboxes,
        meta={"fonte": "examples", "crop_mode": "tight", "detector": "yolo26x"},
    )
    assert (clip_dir / "img" / "001.jpg").exists()
    assert (clip_dir / "img" / "003.jpg").exists()
    assert (clip_dir / "crop_params" / "002.json").exists()
    info = configparser.ConfigParser()
    info.read(clip_dir / "info.ini")
    assert info["info"]["num_frames"] == "3"
    assert info["info"]["fonte"] == "examples"


def test_crop_params_roundtrip(tmp_path: Path) -> None:
    crops, params, bboxes = _sample(2)
    clip_dir = write_crop_clip(tmp_path, "c", crops, params, bboxes)
    back = load_crop_params(clip_dir, 1)
    assert back == params[0]


def test_record_has_bbox_and_meta(tmp_path: Path) -> None:
    crops, params, bboxes = _sample(1)
    clip_dir = write_crop_clip(tmp_path, "c", crops, params, bboxes, meta={"crop_mode": "tight"})
    rec = json.loads((clip_dir / "crop_params" / "001.json").read_text())
    assert rec["finisher_bbox_xyxy"] == [0.0, 0.0, 50.0, 80.0]
    assert rec["crop_mode"] == "tight" and rec["source_frame"] == 1


def test_length_mismatch_raises(tmp_path: Path) -> None:
    crops, params, bboxes = _sample(3)
    import pytest

    with pytest.raises(ValueError, match="mesmo comprimento"):
        write_crop_clip(tmp_path, "c", crops, params[:2], bboxes)
