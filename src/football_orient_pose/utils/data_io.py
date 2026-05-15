"""Data I/O para o dataset 3DSP (3D Shot Posture).

Funções para carregar imagens, keypoints 2D/3D e metadados do dataset 3DSP.
Lida com as particularidades do formato JSON (H3WB) e do info.ini.

Referência:
    - Dataset: calvinyeungck/3D-Shot-Posture-Dataset
    - Paper: Yeung, Ide, Fujii — AutoSoccerPose (CVPRW 2024)
    - Licença: Apache 2.0
"""

from __future__ import annotations

import configparser
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def load_keypoints_2d(json_path: str | Path) -> np.ndarray:
    """Carrega keypoints 2D de um JSON do 3DSP.

    Parameters
    ----------
    json_path : str | Path
        Caminho para o arquivo JSON (e.g. ``data/train/00001/posture/001.json``).

    Returns
    -------
    np.ndarray
        Array de shape ``(17, 2)`` com coordenadas (x, y) no formato H3WB.
        Coordenadas são relativas ao crop (0-100).
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    kp_dict = data["keypoint_2d"]
    keypoints = np.zeros((17, 2), dtype=np.float32)

    for kid, kv in kp_dict.items():
        idx = int(kid)
        keypoints[idx] = [kv["x"], kv["y"]]

    return keypoints


def load_keypoints_3d(json_path: str | Path) -> np.ndarray:
    """Carrega keypoints 3D estimados de um JSON do 3DSP.

    .. note::
        Os keypoints 3D são estimativas do MotionAGFormer,
        **não** ground truth manual.

    Parameters
    ----------
    json_path : str | Path
        Caminho para o arquivo JSON.

    Returns
    -------
    np.ndarray
        Array de shape ``(17, 3)`` com coordenadas (x, y, z) no formato H3WB.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    kp_dict = data["keypoint_3d"]
    keypoints = np.zeros((17, 3), dtype=np.float32)

    for kid, kv in kp_dict.items():
        idx = int(kid)
        keypoints[idx] = [kv["x"], kv["y"], kv["z"]]

    return keypoints


def load_bbox(json_path: str | Path) -> dict[str, int]:
    """Carrega bounding box de um JSON do 3DSP.

    Returns
    -------
    dict
        Dicionário com chaves ``x``, ``y``, ``w``, ``h`` (formato xywh).
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    return data["bbox"]


def load_clip_image(clip_dir: str | Path, frame_idx: int) -> np.ndarray:
    """Carrega uma imagem cropada de um clip do 3DSP.

    Parameters
    ----------
    clip_dir : str | Path
        Diretório do clip (e.g. ``data/train/00001``).
    frame_idx : int
        Índice do frame (1-indexed, como no dataset: 1 a 20).

    Returns
    -------
    np.ndarray
        Imagem BGR de shape ``(100, 100, 3)``.
    """
    img_path = Path(clip_dir) / "img" / f"{frame_idx:03d}.jpg"
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {img_path}")
    return img


def load_clip_info(clip_dir: str | Path) -> dict[str, str]:
    """Carrega metadados do info.ini de um clip do 3DSP.

    Returns
    -------
    dict
        Dicionário com metadados: id, label, team, game, gametime,
        visibility, half, etc.
    """
    ini_path = Path(clip_dir) / "info.ini"
    config = configparser.ConfigParser()
    config.read(str(ini_path))
    return dict(config["info"])


def load_clip_gt(clip_dir: str | Path) -> np.ndarray:
    """Carrega tracklets MOT20 (gt.txt) de um clip do 3DSP.

    Returns
    -------
    np.ndarray
        Array de shape ``(N, 10)`` no formato MOT20:
        ``[frame_id, track_id, x, y, w, h, confidence, -1, -1, -1]``
    """
    gt_path = Path(clip_dir) / "gt" / "gt.txt"
    return np.loadtxt(str(gt_path), delimiter=",")


def load_full_clip(
    clip_dir: str | Path,
    num_frames: int = 20,
    load_3d: bool = False,
) -> dict[str, Any]:
    """Carrega todos os dados de um clip do 3DSP.

    Parameters
    ----------
    clip_dir : str | Path
        Diretório do clip.
    num_frames : int
        Número de frames a carregar (default: 20).
    load_3d : bool
        Se True, também carrega keypoints 3D.

    Returns
    -------
    dict
        Dicionário com:
        - ``images``: lista de imagens (100×100 BGR)
        - ``keypoints_2d``: array (N, 17, 2) se posture/ existir, senão None
        - ``keypoints_3d``: array (N, 17, 3) se load_3d e posture/ existir
        - ``bboxes``: lista de dicts com x, y, w, h
        - ``info``: metadados do info.ini
    """
    clip_dir = Path(clip_dir)
    posture_dir = clip_dir / "posture"
    has_posture = posture_dir.exists()

    images = []
    keypoints_2d_list = []
    keypoints_3d_list = []
    bboxes = []

    for i in range(1, num_frames + 1):
        # Imagem
        images.append(load_clip_image(clip_dir, i))

        # Keypoints (apenas se existir posture/)
        if has_posture:
            json_path = posture_dir / f"{i:03d}.json"
            keypoints_2d_list.append(load_keypoints_2d(json_path))
            bboxes.append(load_bbox(json_path))
            if load_3d:
                keypoints_3d_list.append(load_keypoints_3d(json_path))

    result: dict[str, Any] = {
        "images": images,
        "keypoints_2d": np.array(keypoints_2d_list) if has_posture else None,
        "bboxes": bboxes if has_posture else None,
        "info": load_clip_info(clip_dir),
    }

    if load_3d and has_posture:
        result["keypoints_3d"] = np.array(keypoints_3d_list)

    return result


def iter_clips(
    data_dir: str | Path,
    split: str = "train",
) -> list[Path]:
    """Lista todos os diretórios de clips de um split.

    Parameters
    ----------
    data_dir : str | Path
        Diretório raiz dos dados (e.g. ``data/``).
    split : str
        ``"train"`` ou ``"test"``.

    Returns
    -------
    list[Path]
        Lista de Paths para cada clip, ordenados por ID.
    """
    split_dir = Path(data_dir) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {split_dir}")

    clips = sorted(
        [d for d in split_dir.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )
    return clips
