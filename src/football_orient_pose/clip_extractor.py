"""Extração de clips estruturados a partir de vídeo bruto.

Dado um vídeo e um intervalo (ms), ``cut_clip`` devolve exatamente ``n`` frames
e ``write_clip`` grava a estrutura de clip do 3DSP (``img/{:03d}.jpg`` +
``info.ini``), consumível por ``utils.data_io.load_clip_image`` /
``load_clip_info`` sem adaptação.
"""

from __future__ import annotations

import configparser
from pathlib import Path

import cv2
import numpy as np

from football_orient_pose.video_io import extract_frames


def cut_clip(
    video_path: str | Path,
    start_ms: float,
    end_ms: float,
    n: int = 20,
) -> list[np.ndarray]:
    """Corta um intervalo do vídeo e devolve exatamente ``n`` frames.

    Os frames cujo timestamp cai em ``[start_ms, end_ms]`` são selecionados; se
    houver mais que ``n``, amostra ``n`` uniformemente (``np.linspace``).

    Parameters
    ----------
    video_path : str | Path
        Caminho do vídeo de origem.
    start_ms, end_ms : float
        Limites do intervalo em milissegundos (inclusivos).
    n : int
        Número de frames de saída. Default 20 (igual ao 3DSP).

    Returns
    -------
    list[np.ndarray]
        ``n`` imagens BGR.

    Raises
    ------
    ValueError
        Se ``n <= 0``, ``end_ms <= start_ms`` ou o intervalo render < ``n`` frames.
    """
    if n <= 0:
        raise ValueError(f"n deve ser > 0, recebido {n}")
    if end_ms <= start_ms:
        raise ValueError(f"intervalo inválido: end_ms ({end_ms}) <= start_ms ({start_ms})")

    frames = extract_frames(video_path)
    selected = [f for f in frames if start_ms <= f.timestamp_ms <= end_ms]
    if len(selected) < n:
        raise ValueError(
            f"intervalo curto: {len(selected)} frames em [{start_ms}, {end_ms}] ms < n={n}"
        )

    idx = np.linspace(0, len(selected) - 1, n).round().astype(int)
    return [selected[i].image for i in idx]


def write_clip(
    clip_id: str,
    frames: list[np.ndarray],
    meta: dict,
    root: str | Path = "data/clips",
) -> Path:
    """Grava um clip na estrutura do 3DSP: ``<root>/<clip_id>/img/{:03d}.jpg`` + ``info.ini``.

    Parameters
    ----------
    clip_id : str
        Identificador do clip (ex.: ``"brazil_01"``). Vira o nome da pasta.
    frames : list[np.ndarray]
        Imagens BGR (1-indexed na escrita: ``001.jpg`` ... ).
    meta : dict
        Campos extras do ``[info]`` (ex.: ``source_video, game, label, start_ms,
        end_ms, fps, step_ms, notes``). ``id`` e ``num_frames`` são preenchidos
        automaticamente.
    root : str | Path
        Raiz onde o clip é criado. Default ``data/clips``.

    Returns
    -------
    Path
        Diretório do clip criado.
    """
    clip_dir = Path(root) / clip_id
    img_dir = clip_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    for i, image in enumerate(frames, start=1):  # 1-indexed, igual ao 3DSP
        cv2.imwrite(str(img_dir / f"{i:03d}.jpg"), image)

    config = configparser.ConfigParser()
    info = {"id": clip_id, "num_frames": str(len(frames))}
    info.update({k: str(v) for k, v in meta.items()})
    config["info"] = info
    with open(clip_dir / "info.ini", "w", encoding="utf-8") as fh:
        config.write(fh)

    return clip_dir
