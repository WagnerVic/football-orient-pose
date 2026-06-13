"""Funções de visualização: esqueletos, bounding boxes, vetores de orientação."""

from __future__ import annotations

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
