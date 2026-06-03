"""DSP3Dataset — adapter entre o 3DSP e o protocolo MMPose TopdownPoseEstimator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mmengine.dataset import BaseDataset
from mmpose.registry import DATASETS

from football_orient_pose.utils.data_io import load_bbox, load_keypoints_2d

# IDs H3WB que são keypoints calculados (médias), não anotações manuais.
# O MMPose os excluirá da SimCC loss quando visibility=0.
_DERIVED_KEYPOINT_IDS = {0, 7, 8, 9}

# Metainfo H3WB-17 para o MMPose.
# flip_indices[i] = j significa que, após flip horizontal, o joint i assume a
# posição do joint j (swapa pares L/R; joints da linha central ficam no lugar).
_H3WB_METAINFO = dict(
    dataset_name="3dsp",
    num_keypoints=17,
    keypoint_info={
        0: dict(name="center_hips",      id=0,  color=[128, 128, 128], type="lower",  swap=""),
        1: dict(name="left_hip",         id=1,  color=[0, 255, 0],     type="lower",  swap="right_hip"),
        2: dict(name="left_knee",        id=2,  color=[0, 255, 0],     type="lower",  swap="right_knee"),
        3: dict(name="left_ankle",       id=3,  color=[0, 255, 0],     type="lower",  swap="right_ankle"),
        4: dict(name="right_hip",        id=4,  color=[255, 128, 0],   type="lower",  swap="left_hip"),
        5: dict(name="right_knee",       id=5,  color=[255, 128, 0],   type="lower",  swap="left_knee"),
        6: dict(name="right_ankle",      id=6,  color=[255, 128, 0],   type="lower",  swap="left_ankle"),
        7: dict(name="center_body",      id=7,  color=[128, 128, 128], type="upper",  swap=""),
        8: dict(name="center_shoulder",  id=8,  color=[128, 128, 128], type="upper",  swap=""),
        9: dict(name="neck",             id=9,  color=[128, 128, 128], type="upper",  swap=""),
        10: dict(name="head",            id=10, color=[255, 255, 0],   type="upper",  swap=""),
        11: dict(name="right_shoulder",  id=11, color=[255, 128, 0],   type="upper",  swap="left_shoulder"),
        12: dict(name="right_elbow",     id=12, color=[255, 128, 0],   type="upper",  swap="left_elbow"),
        13: dict(name="right_wrist",     id=13, color=[255, 128, 0],   type="upper",  swap="left_wrist"),
        14: dict(name="left_shoulder",   id=14, color=[0, 255, 0],     type="upper",  swap="right_shoulder"),
        15: dict(name="left_elbow",      id=15, color=[0, 255, 0],     type="upper",  swap="right_elbow"),
        16: dict(name="left_wrist",      id=16, color=[0, 255, 0],     type="upper",  swap="right_wrist"),
    },
    skeleton_info={
        0:  dict(link=("left_hip",        "right_hip"),       id=0,  color=[128, 128, 128]),
        1:  dict(link=("left_hip",        "left_knee"),       id=1,  color=[0, 255, 0]),
        2:  dict(link=("left_knee",       "left_ankle"),      id=2,  color=[0, 255, 0]),
        3:  dict(link=("right_hip",       "right_knee"),      id=3,  color=[255, 128, 0]),
        4:  dict(link=("right_knee",      "right_ankle"),     id=4,  color=[255, 128, 0]),
        5:  dict(link=("center_hips",     "center_body"),     id=5,  color=[128, 128, 128]),
        6:  dict(link=("center_body",     "center_shoulder"), id=6,  color=[128, 128, 128]),
        7:  dict(link=("center_shoulder", "neck"),            id=7,  color=[128, 128, 128]),
        8:  dict(link=("neck",            "head"),            id=8,  color=[128, 128, 128]),
        9:  dict(link=("center_shoulder", "right_shoulder"),  id=9,  color=[255, 128, 0]),
        10: dict(link=("right_shoulder",  "right_elbow"),     id=10, color=[255, 128, 0]),
        11: dict(link=("right_elbow",     "right_wrist"),     id=11, color=[255, 128, 0]),
        12: dict(link=("center_shoulder", "left_shoulder"),   id=12, color=[0, 255, 0]),
        13: dict(link=("left_shoulder",   "left_elbow"),      id=13, color=[0, 255, 0]),
        14: dict(link=("left_elbow",      "left_wrist"),      id=14, color=[0, 255, 0]),
    },
    # flip_indices[i] = j: após flip horizontal, joint i → posição do joint j
    flip_indices=[0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13],
    joint_weights=[1.] * 17,
    # Sigmas H3WB para OKS (do arquivo metrics.py do projeto)
    sigmas=[
        0.107, 0.107, 0.087, 0.089,  # 0-3
        0.107, 0.087, 0.089,          # 4-6
        0.093, 0.079, 0.052, 0.026,   # 7-10
        0.079, 0.072, 0.062,          # 11-13
        0.079, 0.072, 0.062,          # 14-16
    ],
)


@DATASETS.register_module()
class DSP3Dataset(BaseDataset):
    """Dataset MMPose para o 3DSP (3D Shot Posture Dataset).

    Converte a estrutura de diretórios do 3DSP para o protocolo TopdownPoseEstimator
    do MMPose, sem precisar de arquivos intermediários em disco.

    Cada item representa um frame: uma imagem 100×100 com um único jogador e 17
    keypoints H3WB. Os 4 keypoints derivados (IDs 0, 7, 8, 9) recebem
    ``visibility=0`` para serem ignorados na SimCC loss.

    Parameters
    ----------
    data_root : str
        Raiz do dataset 3DSP (diretório que contém a pasta ``train/``).
    split_file : str
        Path para o split.json com chaves ``"train"`` e ``"val"``.
    split : str
        ``"train"`` ou ``"val"``.
    """

    METAINFO = _H3WB_METAINFO

    def __init__(
        self,
        data_root: str,
        split_file: str,
        split: str,
        **kwargs,
    ) -> None:
        self._data_root = Path(data_root)
        self._split_file = Path(split_file)
        self._split = split
        super().__init__(ann_file="", **kwargs)

    def load_data_list(self) -> list[dict]:
        split_data = json.loads(self._split_file.read_text())
        if self._split not in split_data:
            raise ValueError(
                f"Split '{self._split}' não encontrado em {self._split_file}. "
                f"Opções: {list(split_data.keys())}"
            )
        clip_ids: list[str] = split_data[self._split]

        data_list = []
        item_id = 0
        for clip_id in clip_ids:
            clip_dir = self._data_root / "train" / clip_id
            for frame_idx in range(1, 21):
                posture_path = clip_dir / "posture" / f"{frame_idx:03d}.json"
                img_path = str(clip_dir / "img" / f"{frame_idx:03d}.jpg")

                keypoints_2d = load_keypoints_2d(posture_path)  # (17, 2) em coords 0-100
                bbox_dict = load_bbox(posture_path)

                # Visibilidade: 0 para keypoints derivados, 1 para os demais
                visibility = np.ones(17, dtype=np.float32)
                for kid in _DERIVED_KEYPOINT_IDS:
                    visibility[kid] = 0.0

                # MMPose espera arrays (num_instances, K, 2) e (num_instances, K)
                keypoints = keypoints_2d[np.newaxis]     # (1, 17, 2)
                keypoints_visible = visibility[np.newaxis]  # (1, 17)

                # bbox xyxy
                x, y, w, h = (
                    float(bbox_dict["x"]),
                    float(bbox_dict["y"]),
                    float(bbox_dict["w"]),
                    float(bbox_dict["h"]),
                )
                bbox = np.array([[x, y, x + w, y + h]], dtype=np.float32)  # (1, 4)
                bbox_score = np.ones(1, dtype=np.float32)                   # (1,)

                data_list.append(dict(
                    img_path=img_path,
                    img_id=item_id,
                    id=item_id,
                    width=100,
                    height=100,
                    bbox=bbox,
                    bbox_score=bbox_score,
                    keypoints=keypoints,
                    keypoints_visible=keypoints_visible,
                    num_keypoints=int(visibility.sum()),
                    category_id=1,
                    iscrowd=0,
                    segmentation=[],
                ))
                item_id += 1

        return data_list
