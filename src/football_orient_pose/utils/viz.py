"""Funções de visualização: esqueletos, bounding boxes, vetores de orientação."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from football_orient_pose.utils.skeleton import (
    H3WB_BONES,
    H3WB_COLORS,
    H3WB_JOINTS_LEFT,
    H3WB_JOINTS_RIGHT,
)


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    labels: list[str] | None = None,
) -> np.ndarray:
    """Desenha caixas ``xyxy`` (pixel de frame) sobre uma cópia da imagem BGR.

    Usado para comparar GT × predição de detectores (ex.: GT verde, predição vermelha). Não altera
    a imagem original. ``labels`` opcional escreve um texto acima de cada caixa.
    """
    out = image.copy()
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        p1, p2 = (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))
        cv2.rectangle(out, p1, p2, color, thickness)
        if labels is not None and i < len(labels):
            cv2.putText(
                out, labels[i], (p1[0], max(0, p1[1] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )
    return out


def draw_skeleton(
    image: np.ndarray,
    keypoints: np.ndarray,
    bones: list[tuple[int, int]] = H3WB_BONES,
    joints_left: list[int] = H3WB_JOINTS_LEFT,
    joints_right: list[int] = H3WB_JOINTS_RIGHT,
    point_radius: int = 2,
    thickness: int = 2,
) -> np.ndarray:
    """Desenha o esqueleto H3WB-17 (ossos + juntas) sobre uma cópia da imagem BGR.

    ``keypoints`` é ``(17, 2)`` (mesmas coordenadas da imagem — crop ou frame). Cada osso/junta é
    colorido por lateralidade (esquerda/direita/centro) via ``H3WB_COLORS``. Não muta a original.
    """
    out = image.copy()
    kps = np.asarray(keypoints, dtype=np.float64).reshape(-1, 2)

    def _color(j: int) -> tuple[int, int, int]:
        if j in joints_left:
            return H3WB_COLORS["left"]
        if j in joints_right:
            return H3WB_COLORS["right"]
        return H3WB_COLORS["center"]

    def _pt(j: int) -> tuple[int, int]:
        return int(round(kps[j, 0])), int(round(kps[j, 1]))

    for a, b in bones:
        cv2.line(out, _pt(a), _pt(b), _color(b), thickness, cv2.LINE_AA)
    for j in range(len(kps)):
        cv2.circle(out, _pt(j), point_radius, _color(j), -1, cv2.LINE_AA)
    return out


def frames_to_gif(
    frames: list[np.ndarray],
    out_path: str | Path,
    fps: float = 5.0,
    target_width: int | None = None,
    loop: int = 0,
) -> Path:
    """Monta um GIF animado a partir de uma lista de frames BGR (np arrays).

    Pós-processa as imagens do showcase numa animação leve. Se ``target_width`` for dado,
    redimensiona cada quadro preservando o aspecto — **downscale** com ``INTER_AREA`` (frame de
    broadcast) e **upscale** com ``INTER_NEAREST`` (crops 100×100, p/ não borrar o esqueleto fino).
    Quantiza cada quadro em palette adaptativa (256 cores) e salva com Pillow otimizado.

    Parameters
    ----------
    frames : list[np.ndarray]
        Quadros ``H×W×3`` em BGR (ordem já correta).
    out_path : str | Path
        Caminho do ``.gif`` de saída (pastas criadas se necessário).
    fps : float
        Quadros por segundo (vira ``duration = 1000/fps`` ms por quadro).
    target_width : int | None
        Largura-alvo; ``None`` mantém o tamanho original.
    loop : int
        ``0`` = loop infinito.

    Returns
    -------
    Path
        O caminho do GIF gravado.
    """
    from PIL import Image

    if not frames:
        raise ValueError("frames_to_gif: lista de frames vazia")

    pil_frames: list[Image.Image] = []
    for frame in frames:
        img = np.asarray(frame)
        if target_width is not None and img.shape[1] != target_width:
            h, w = img.shape[:2]
            target_height = max(1, round(h * target_width / w))
            interp = cv2.INTER_AREA if target_width < w else cv2.INTER_NEAREST
            img = cv2.resize(img, (target_width, target_height), interpolation=interp)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_frames.append(Image.fromarray(rgb).quantize(colors=256, method=Image.MEDIANCUT))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pil_frames[0].save(
        out,
        save_all=True,
        append_images=pil_frames[1:],
        duration=round(1000 / fps),
        loop=loop,
        optimize=True,
        disposal=2,
    )
    return out
