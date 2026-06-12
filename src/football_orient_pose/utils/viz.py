"""Funções de visualização: esqueletos, bounding boxes, vetores de orientação."""

from __future__ import annotations

import cv2
import numpy as np


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
