from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from football_orient_pose.utils import split_clips


def _make_clip_dirs(data_dir: Path, n_clips: int) -> list[str]:
    train_dir = data_dir / "train"
    train_dir.mkdir(parents=True)

    names = [f"{idx:05d}" for idx in range(1, n_clips + 1)]
    for name in names:
        (train_dir / name).mkdir()

    return names


def test_split_clips_uses_clip_level_80_20_split(tmp_path: Path) -> None:
    _make_clip_dirs(tmp_path, 200)

    train_clips, val_clips = split_clips(tmp_path, train_ratio=0.8, seed=42)

    assert len(train_clips) == 160
    assert len(val_clips) == 40
    assert set(train_clips).isdisjoint(val_clips)
    assert all(isinstance(clip, Path) for clip in train_clips + val_clips)


def test_split_clips_is_reproducible_for_same_seed(tmp_path: Path) -> None:
    _make_clip_dirs(tmp_path, 200)

    first_split = split_clips(tmp_path, train_ratio=0.8, seed=42)
    second_split = split_clips(tmp_path, train_ratio=0.8, seed=42)

    assert first_split == second_split


def test_split_clips_changes_with_different_seed(tmp_path: Path) -> None:
    _make_clip_dirs(tmp_path, 200)

    first_split = split_clips(tmp_path, train_ratio=0.8, seed=42)
    second_split = split_clips(tmp_path, train_ratio=0.8, seed=7)

    assert first_split != second_split


def test_split_clips_handles_non_200_dataset_sizes(tmp_path: Path) -> None:
    _make_clip_dirs(tmp_path, 7)

    train_clips, val_clips = split_clips(tmp_path, train_ratio=0.8, seed=42)

    assert len(train_clips) == 5
    assert len(val_clips) == 2
    assert set(train_clips).isdisjoint(val_clips)


def test_split_clips_rejects_invalid_train_ratio(tmp_path: Path) -> None:
    _make_clip_dirs(tmp_path, 2)

    with pytest.raises(ValueError, match="train_ratio"):
        split_clips(tmp_path, train_ratio=1.0)


def test_committed_split_json_matches_seed_42_algorithm() -> None:
    split_path = Path("configs/split.json")
    split = json.loads(split_path.read_text())

    clip_names = [f"{idx:05d}" for idx in range(1, 201)]
    rng = random.Random(split["seed"])
    rng.shuffle(clip_names)

    assert split["train_ratio"] == 0.8
    assert split["n_train"] == 160
    assert split["n_val"] == 40
    assert split["train"] == clip_names[:160]
    assert split["val"] == clip_names[160:]
    assert set(split["train"]).isdisjoint(split["val"])
