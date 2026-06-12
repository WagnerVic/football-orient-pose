from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from football_orient_pose.utils.roboflow_io import (
    load_clip_bbox_gt,
    load_coco_bboxes,
    load_examples_bbox_gt,
)


def _write_coco(path: Path, frames: list[tuple[str, list[list[float]]]], cls: str = "Jogador"):
    """Escreve um COCO no estilo Roboflow (raiz vazia id 0 + classe real id 1)."""
    images, anns, aid = [], [], 1
    for img_id, (file_name, boxes) in enumerate(frames):
        images.append({"id": img_id, "file_name": file_name, "width": 1280, "height": 720})
        for x, y, w, h in boxes:
            anns.append(
                {"id": aid, "image_id": img_id, "category_id": 1, "bbox": [x, y, w, h]}
            )
            aid += 1
    coco = {
        "categories": [
            {"id": 0, "name": "root", "supercategory": "none"},
            {"id": 1, "name": cls, "supercategory": "root"},
        ],
        "images": images,
        "annotations": anns,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(coco), encoding="utf-8")
    return path


def test_xywh_is_converted_to_xyxy(tmp_path: Path) -> None:
    p = _write_coco(tmp_path / "c.json", [("001_jpg.rf.abc.jpg", [[10, 20, 30, 40]])])
    gt = load_coco_bboxes(p)
    assert set(gt) == {1}
    np.testing.assert_allclose(gt[1], [[10, 20, 40, 60]])  # x2=x+w, y2=y+h
    assert gt[1].dtype == np.float32


def test_frame_index_from_roboflow_filename(tmp_path: Path) -> None:
    p = _write_coco(
        tmp_path / "c.json",
        [("009_jpg.rf.xyz.jpg", [[0, 0, 5, 5]]), ("020_jpg.rf.k.jpg", [[1, 1, 2, 2]])],
    )
    gt = load_coco_bboxes(p)
    assert set(gt) == {9, 20}


def test_other_category_and_empty_frame(tmp_path: Path) -> None:
    p = _write_coco(tmp_path / "c.json", [("003_jpg.rf.a.jpg", [])])  # frame sem caixa
    gt = load_coco_bboxes(p)
    assert gt[3].shape == (0, 4)


def test_missing_class_raises(tmp_path: Path) -> None:
    p = _write_coco(tmp_path / "c.json", [("001_jpg.rf.a.jpg", [[0, 0, 1, 1]])])
    with pytest.raises(ValueError, match="ausente"):
        load_coco_bboxes(p, class_name="Inexistente")


def test_load_clip_finds_nested_json(tmp_path: Path) -> None:
    _write_coco(tmp_path / "example_01" / "train" / "_annotations.coco.json",
                [("001_jpg.rf.a.jpg", [[0, 0, 1, 1]])])
    gt = load_clip_bbox_gt(tmp_path / "example_01")
    assert gt[1].shape == (1, 4)


def test_load_examples_keys_by_folder(tmp_path: Path) -> None:
    for clip in ("example_01", "example_02"):
        _write_coco(tmp_path / clip / "_annotations.coco.json",
                    [("001_jpg.rf.a.jpg", [[0, 0, 1, 1]])])
    allgt = load_examples_bbox_gt(tmp_path)
    assert set(allgt) == {"example_01", "example_02"}


# --- smoke test contra o GT real, se estiver presente ---
_REAL = Path("data/annotations/examples_bbox")


@pytest.mark.skipif(not _REAL.exists(), reason="GT real do Roboflow ausente")
def test_real_examples_gt_shapes() -> None:
    gt = load_examples_bbox_gt(_REAL)
    assert set(gt) == {"example_01", "example_02", "example_03"}
    for clip, frames in gt.items():
        assert set(frames) == set(range(1, 21)), clip  # 20 frames, 1..20
        for arr in frames.values():
            assert arr.ndim == 2 and arr.shape[1] == 4
    # totais conferidos na análise: 188 / 246 / 306
    assert sum(len(a) for a in gt["example_01"].values()) == 188
    assert sum(len(a) for a in gt["example_03"].values()) == 306
