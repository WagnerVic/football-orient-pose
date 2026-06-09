"""Leitura de vídeo: abre um MP4 e devolve frames amostrados.

Este módulo é o primeiro elo do pipeline de visão — transforma um arquivo de
vídeo numa lista de frames em memória. É reutilizado pela extração de clips
(Épico 13), pela detecção (Épico 10) e pelo pipeline end-to-end (Épico 12).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Frame:
    """Um frame extraído de um vídeo.

    Attributes
    ----------
    image : np.ndarray
        Imagem do frame em ``H x W x C`` (BGR, padrão OpenCV).
    index : int
        Índice original do frame no vídeo (0-indexed).
    timestamp_ms : float
        Posição temporal do frame em milissegundos.
    """

    image: np.ndarray
    index: int
    timestamp_ms: float


def extract_frames(
    video_path: str | Path,
    every_n: int | None = None,
    target_fps: float | None = None,
    start_ms: float | None = None,
    end_ms: float | None = None,
) -> list[Frame]:
    """Extrai frames de um vídeo MP4 a uma taxa configurável.

    Use ``every_n`` (pega 1 a cada N frames) **ou** ``target_fps`` (reamostra
    para uma taxa-alvo) — não os dois. Sem nenhum, devolve todos os frames.

    A janela ``[start_ms, end_ms]`` limita a extração a um trecho do vídeo:
    frames antes de ``start_ms`` são descartados (não ficam em memória) e a
    leitura **para** assim que passa de ``end_ms``. Essencial para vídeos longos
    — sem a janela, todos os frames do vídeo iriam para a RAM.

    Parameters
    ----------
    video_path : str | Path
        Caminho para o arquivo de vídeo.
    every_n : int | None
        Passo de amostragem por frames. ``every_n=5`` → frames 0, 5, 10, ...
    target_fps : float | None
        Taxa-alvo em quadros por segundo. O passo é ``round(fps_vídeo / target_fps)``.
    start_ms, end_ms : float | None
        Limites temporais (ms, inclusivos) do trecho a extrair. ``None`` = sem limite.

    Returns
    -------
    list[Frame]
        Frames amostrados (dentro da janela), em ordem crescente de ``index``.

    Raises
    ------
    FileNotFoundError
        Se o vídeo não existir ou não puder ser aberto.
    ValueError
        Se ``every_n`` e ``target_fps`` forem informados juntos, ou se forem <= 0.
    """
    if every_n is not None and target_fps is not None:
        raise ValueError("informe every_n OU target_fps, não os dois")
    if every_n is not None and every_n <= 0:
        raise ValueError(f"every_n deve ser > 0, recebido {every_n}")
    if target_fps is not None and target_fps <= 0:
        raise ValueError(f"target_fps deve ser > 0, recebido {target_fps}")

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"vídeo não encontrado: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise FileNotFoundError(f"não foi possível abrir o vídeo: {path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if target_fps is not None:
        if src_fps <= 0:
            raise ValueError(
                f"o vídeo não reporta FPS; use every_n em vez de target_fps para {path}"
            )
        step = max(1, round(src_fps / target_fps))
    else:
        step = every_n or 1

    frames: list[Frame] = []
    index = 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            # timestamp por índice/fps é robusto; CAP_PROP_POS_MSEC é 0/incorreto
            # em vários streams (ex.: vídeo-only do YouTube).
            if src_fps > 0:
                timestamp_ms = index / src_fps * 1000.0
            else:
                timestamp_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
            if end_ms is not None and timestamp_ms > end_ms:
                break  # passou da janela — para de decodifar o resto
            if start_ms is None or timestamp_ms >= start_ms:
                if index % step == 0:
                    frames.append(Frame(image=image, index=index, timestamp_ms=timestamp_ms))
            index += 1
    finally:
        cap.release()  # nunca vazar o handle do VideoCapture

    return frames
