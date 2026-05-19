"""Utilitários: mapeamento de keypoints, esqueletos, I/O de dados, visualização."""

from football_orient_pose.utils.data_io import (
    iter_clips,
    load_clip_image,
    load_full_clip,
    load_keypoints_2d,
    split_clips,
)
from football_orient_pose.utils.dataset import DSPDataset
from football_orient_pose.utils.keypoint_mapping import (
    COCO17_NAMES,
    H3WB17_NAMES,
    ORIENTATION_KP_IDS,
    coco17_to_h3wb17,
    coco17_to_h3wb17_batch,
)
from football_orient_pose.utils.skeleton import (
    COCO17_BONES,
    H3WB_BONES,
    H3WB_JOINTS_LEFT,
    H3WB_JOINTS_RIGHT,
)

__all__ = [
    "coco17_to_h3wb17",
    "coco17_to_h3wb17_batch",
    "COCO17_NAMES",
    "H3WB17_NAMES",
    "ORIENTATION_KP_IDS",
    "COCO17_BONES",
    "DSPDataset",
    "H3WB_BONES",
    "H3WB_JOINTS_LEFT",
    "H3WB_JOINTS_RIGHT",
    "load_keypoints_2d",
    "load_clip_image",
    "load_full_clip",
    "iter_clips",
    "split_clips",
]
