"""Utilitários: mapeamento de keypoints, esqueletos, I/O de dados, visualização."""

from football_orient_pose.utils.keypoint_mapping import (
    coco17_to_h3wb17,
    coco17_to_h3wb17_batch,
    COCO17_NAMES,
    H3WB17_NAMES,
    ORIENTATION_KP_IDS,
)
from football_orient_pose.utils.skeleton import (
    COCO17_BONES,
    H3WB_BONES,
    H3WB_JOINTS_LEFT,
    H3WB_JOINTS_RIGHT,
)
from football_orient_pose.utils.data_io import (
    load_keypoints_2d,
    load_clip_image,
    load_full_clip,
    iter_clips,
)

__all__ = [
    "coco17_to_h3wb17",
    "coco17_to_h3wb17_batch",
    "COCO17_NAMES",
    "H3WB17_NAMES",
    "ORIENTATION_KP_IDS",
    "COCO17_BONES",
    "H3WB_BONES",
    "H3WB_JOINTS_LEFT",
    "H3WB_JOINTS_RIGHT",
    "load_keypoints_2d",
    "load_clip_image",
    "load_full_clip",
    "iter_clips",
]
