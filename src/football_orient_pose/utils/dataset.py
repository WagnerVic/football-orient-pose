"""Dataset PyTorch para o 3DSP."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from football_orient_pose.utils.data_io import load_bbox, load_clip_image, load_keypoints_2d
from football_orient_pose.utils.skeleton import H3WB_SWAP_PAIRS


class DSPDataset(Dataset):
    """Dataset lazy para imagens e keypoints 2D do 3DSP.

    Cada item representa um frame de um clip e retorna tensores prontos para
    treinamento ou avaliação com ``torch.utils.data.DataLoader``.
    """

    def __init__(
        self,
        clip_dirs: list[Path],
        augment: bool = False,
        normalize_keypoints: bool = False,
    ) -> None:
        """Inicializa o dataset com diretórios de clips do 3DSP.

        Parameters
        ----------
        clip_dirs : list[Path]
            Diretórios dos clips, por exemplo ``data/3dsp/train/00001``.
        augment : bool
            Se True, aplica flip horizontal aleatório com swap de lateralidade.
        normalize_keypoints : bool
            Se True, normaliza keypoints 2D pelo tamanho do crop para [0, 1].
        """
        self.clip_dirs = [Path(clip_dir) for clip_dir in clip_dirs]
        self.augment = augment
        self.normalize_keypoints = normalize_keypoints
        self.num_frames = 20
        self.crop_size = 100.0

    @classmethod
    def from_split(
        cls,
        data_dir: str | Path,
        split: str,
        split_path: str | Path = "configs/split.json",
        augment: bool = False,
        normalize_keypoints: bool = False,
    ) -> "DSPDataset":
        """Cria um dataset usando os nomes de clips persistidos em split.json."""
        split_data = json.loads(Path(split_path).read_text())
        if split not in split_data:
            raise ValueError(f"Split inválido: {split}")

        clip_dirs = [Path(data_dir) / "train" / clip_id for clip_id in split_data[split]]
        return cls(
            clip_dirs=clip_dirs,
            augment=augment,
            normalize_keypoints=normalize_keypoints,
        )

    def __len__(self) -> int:
        """Retorna o total de frames do dataset."""
        return len(self.clip_dirs) * self.num_frames

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, int]:
        """Carrega um frame e retorna imagem, keypoints, bbox, clip_id e frame_idx."""
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        clip_idx = idx // self.num_frames
        frame_idx = (idx % self.num_frames) + 1
        clip_dir = self.clip_dirs[clip_idx]
        posture_path = clip_dir / "posture" / f"{frame_idx:03d}.json"

        image = self._load_image_tensor(clip_dir, frame_idx)
        keypoints = torch.from_numpy(load_keypoints_2d(posture_path)).float()
        bbox = self._load_bbox_tensor(posture_path)

        if self.augment and random.random() > 0.5:
            image, keypoints = self._horizontal_flip(image, keypoints)

        if self.normalize_keypoints:
            keypoints = keypoints / self.crop_size

        return image, keypoints, bbox, clip_dir.name, frame_idx

    def _load_image_tensor(self, clip_dir: Path, frame_idx: int) -> torch.Tensor:
        image = load_clip_image(clip_dir, frame_idx)
        return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

    def _load_bbox_tensor(self, posture_path: Path) -> torch.Tensor:
        bbox = load_bbox(posture_path)
        x1 = float(bbox["x"])
        y1 = float(bbox["y"])
        x2 = x1 + float(bbox["w"])
        y2 = y1 + float(bbox["h"])
        return torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

    def _horizontal_flip(
        self,
        image: torch.Tensor,
        keypoints: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image = image.flip(-1)
        keypoints = keypoints.clone()
        keypoints[:, 0] = self.crop_size - keypoints[:, 0]

        for left, right in H3WB_SWAP_PAIRS:
            left_keypoint = keypoints[left].clone()
            keypoints[left] = keypoints[right]
            keypoints[right] = left_keypoint

        return image, keypoints
