from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from football_orient_pose.utils import DSPDataset


def _make_clip(data_dir: Path, clip_id: str) -> Path:
    clip_dir = data_dir / "train" / clip_id
    img_dir = clip_dir / "img"
    posture_dir = clip_dir / "posture"
    img_dir.mkdir(parents=True)
    posture_dir.mkdir()

    for frame_idx in range(1, 21):
        image = np.full((100, 100, 3), frame_idx, dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"{frame_idx:03d}.jpg"), image)

        keypoints = {
            str(kp_idx): {
                "name": f"kp_{kp_idx}",
                "x": float(kp_idx + frame_idx),
                "y": float(kp_idx + frame_idx + 1),
            }
            for kp_idx in range(17)
        }
        posture = {
            "bbox": {"x": 10, "y": 20, "w": 30, "h": 40},
            "keypoint_2d": keypoints,
            "keypoint_3d": {},
        }
        (posture_dir / f"{frame_idx:03d}.json").write_text(json.dumps(posture))

    return clip_dir


def test_dsp_dataset_returns_normalized_tensors(tmp_path: Path) -> None:
    clip_dir = _make_clip(tmp_path, "00001")
    dataset = DSPDataset([clip_dir])

    image, keypoints, bbox, clip_id, frame_idx = dataset[0]

    assert len(dataset) == 20
    assert image.shape == (3, 100, 100)
    assert image.dtype == torch.float32
    assert torch.all((image >= 0.0) & (image <= 1.0))
    assert keypoints.shape == (17, 2)
    assert keypoints.dtype == torch.float32
    assert bbox.shape == (4,)
    assert bbox.dtype == torch.float32
    assert torch.equal(bbox, torch.tensor([10.0, 20.0, 40.0, 60.0]))
    assert clip_id == "00001"
    assert frame_idx == 1


def test_dsp_dataset_can_normalize_keypoints_by_crop_size(tmp_path: Path) -> None:
    clip_dir = _make_clip(tmp_path, "00001")
    dataset = DSPDataset([clip_dir], normalize_keypoints=True)

    _, keypoints, _, _, _ = dataset[0]

    assert torch.equal(keypoints[0], torch.tensor([0.01, 0.02]))
    assert torch.equal(keypoints[16], torch.tensor([0.17, 0.18]))


def test_dsp_dataset_maps_global_index_to_clip_and_frame(tmp_path: Path) -> None:
    first_clip = _make_clip(tmp_path, "00001")
    second_clip = _make_clip(tmp_path, "00002")
    dataset = DSPDataset([first_clip, second_clip])

    _, _, _, clip_id, frame_idx = dataset[20]

    assert len(dataset) == 40
    assert clip_id == "00002"
    assert frame_idx == 1


def test_dsp_dataset_works_with_torch_dataloader(tmp_path: Path) -> None:
    clip_dir = _make_clip(tmp_path, "00001")
    dataset = DSPDataset([clip_dir])
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    images, keypoints, bboxes, clip_ids, frame_idxs = next(iter(loader))

    assert images.shape == (16, 3, 100, 100)
    assert keypoints.shape == (16, 17, 2)
    assert bboxes.shape == (16, 4)
    assert len(clip_ids) == 16
    assert frame_idxs.shape == (16,)


def test_dsp_dataset_flip_augmentation_swaps_h3wb_lateral_keypoints(
    tmp_path: Path,
    monkeypatch,
) -> None:
    clip_dir = _make_clip(tmp_path, "00001")
    dataset = DSPDataset([clip_dir], augment=True)
    monkeypatch.setattr("football_orient_pose.utils.dataset.random.random", lambda: 1.0)

    _, keypoints, _, _, _ = dataset[0]

    assert torch.equal(keypoints[1], torch.tensor([95.0, 6.0]))
    assert torch.equal(keypoints[4], torch.tensor([98.0, 3.0]))
    assert torch.equal(keypoints[14], torch.tensor([88.0, 13.0]))
    assert torch.equal(keypoints[11], torch.tensor([85.0, 16.0]))


def test_dsp_dataset_from_split_reads_clip_names(tmp_path: Path) -> None:
    _make_clip(tmp_path, "00001")
    _make_clip(tmp_path, "00002")
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "train": ["00001"],
                "val": ["00002"],
                "seed": 42,
                "train_ratio": 0.5,
                "n_train": 1,
                "n_val": 1,
            }
        )
    )

    dataset = DSPDataset.from_split(tmp_path, split="val", split_path=split_path)

    assert len(dataset) == 20
    assert dataset.clip_dirs == [tmp_path / "train" / "00002"]


def test_official_train_split_has_expected_dataset_length() -> None:
    dataset = DSPDataset.from_split("data/3dsp", split="train")

    assert len(dataset) == 3200
