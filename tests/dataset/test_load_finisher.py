from __future__ import annotations

from pathlib import Path

import numpy as np

from football_orient_pose.utils.data_io import load_finisher_boxes


def _make_clip(root: Path, shooter_id: int = 7) -> Path:
    clip = root / "00001"
    (clip / "gt").mkdir(parents=True)
    (clip / "info.ini").write_text(
        f"[info]\nid = 00001\nshooter_tracklet_id = {shooter_id}\n", encoding="utf-8"
    )
    # MOT20: frame, track, x, y, w, h, conf, -1,-1,-1  — dois tracklets (1 e 7)
    lines = [
        "1,1,10,10,20,40,0.9,-1,-1,-1",  # outro jogador
        "1,7,100,200,50,90,0.9,-1,-1,-1",  # finalizador (frame 1)
        "2,7,110,205,50,90,0.9,-1,-1,-1",  # finalizador (frame 2)
        "2,1,12,12,20,40,0.9,-1,-1,-1",
    ]
    (clip / "gt" / "gt.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return clip


def test_load_finisher_boxes_filters_shooter_and_converts_xyxy(tmp_path: Path) -> None:
    clip = _make_clip(tmp_path, shooter_id=7)
    boxes = load_finisher_boxes(clip)
    assert set(boxes) == {1, 2}  # só os frames do finalizador
    # xywh (100,200,50,90) -> xyxy (100,200,150,290)
    np.testing.assert_array_equal(boxes[1], [100, 200, 150, 290])
    np.testing.assert_array_equal(boxes[2], [110, 205, 160, 295])
    assert boxes[1].dtype == np.float32
