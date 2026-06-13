"""Recorte do jogador (crop 100×100) a partir de uma caixa de detecção (Épico #119, US #101).

Gera dois enquadramentos da **mesma** caixa — **justo** (tight, letterbox preservando aspecto,
estilo treino) e **frouxo** (loose, quadrado expandido que emula o crop do 3DSP) — e oferece
transformadas determinísticas crop↔frame. Assim o GT de keypoint anotado uma vez serve aos dois
crops (única variável = enquadramento): projeta-se um ponto de um crop ao outro via coord. de frame.

A saída é ``size×size`` (default 100, para casar com o baseline 3DSP / modelo do Épico 2); ``size``
é parâmetro, então resoluções maiores são triviais. O letterbox usa escala uniforme + padding preto
(**nunca stretch**), espelhando o pré-proc de treino (100×100 → 288×384).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

# Margem extra do crop frouxo (lado = max(w, h, size) * LOOSE_FACTOR), emulando o 3DSP.
LOOSE_FACTOR = 1.4


@dataclass(frozen=True)
class CropParams:
    """Geometria do recorte, suficiente para inverter coordenadas crop↔frame.

    ``x0, y0``: origem (canto sup-esq) do recorte no frame. ``scale``: fator do letterbox
    (px_crop / px_frame). ``pad_top, pad_left``: padding do letterbox dentro do crop. ``size``: lado
    do crop de saída.
    """

    x0: float
    y0: float
    scale: float
    pad_top: float
    pad_left: float
    size: int = 100

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> CropParams:
        return cls(**d)


def _letterbox(
    region: np.ndarray, x0: float, y0: float, size: int
) -> tuple[np.ndarray, CropParams]:
    """Redimensiona ``region`` para caber em ``size×size`` preservando aspecto + padding preto."""
    rh, rw = region.shape[:2]
    scale = size / max(rh, rw)
    new_w, new_h = max(1, round(rw * scale)), max(1, round(rh * scale))
    resized = cv2.resize(region, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    out = np.zeros((size, size, 3), dtype=np.uint8)
    pad_top = (size - new_h) // 2
    pad_left = (size - new_w) // 2
    out[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
    return out, CropParams(x0=x0, y0=y0, scale=scale, pad_top=pad_top, pad_left=pad_left, size=size)


def make_crop(
    frame: np.ndarray,
    bbox: np.ndarray | tuple[float, float, float, float],
    mode: str = "tight",
    margin: float = 0.12,
    size: int = 100,
) -> tuple[np.ndarray, CropParams]:
    """Recorta um patch ``size×size`` do jogador a partir de ``bbox`` (xyxy, coords de frame).

    ``mode="tight"``: caixa + ``margin`` em cada lado (enquadramento justo). ``mode="loose"``:
    quadrado ``max(w, h, size) * LOOSE_FACTOR`` centrado na caixa (emula o frouxo do 3DSP). Em
    ambos, a região é recortada (clampada aos limites do frame) e passada por ``_letterbox``.

    Retorna ``(crop, CropParams)``; o ``CropParams`` permite projetar pontos crop↔frame.
    """
    x1, y1, x2, y2 = (float(v) for v in bbox)
    w, h = x2 - x1, y2 - y1

    if mode == "tight":
        x1 -= w * margin
        x2 += w * margin
        y1 -= h * margin
        y2 += h * margin
    elif mode == "loose":
        side = max(w, h, size) * LOOSE_FACTOR
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        x1, x2 = cx - side / 2, cx + side / 2
        y1, y2 = cy - side / 2, cy + side / 2
    else:
        raise ValueError(f"mode inválido: {mode!r} (use 'tight' ou 'loose')")

    fh, fw = frame.shape[:2]
    # clampa aos limites do frame (top-left por max; bottom-right pelo min)
    xa, ya = max(int(round(x1)), 0), max(int(round(y1)), 0)
    xb, yb = min(int(round(x2)), fw), min(int(round(y2)), fh)
    if xb <= xa or yb <= ya:
        raise ValueError(f"caixa fora do frame ou degenerada: {bbox}")

    region = frame[ya:yb, xa:xb]
    return _letterbox(region, x0=xa, y0=ya, size=size)


def crop_to_frame(pt_xy: np.ndarray | tuple[float, float], p: CropParams) -> np.ndarray:
    """Ponto(s) no crop (0..size) → coordenada(s) no frame. Aceita ``(2,)`` ou ``(N, 2)``."""
    pt = np.asarray(pt_xy, dtype=np.float64)
    return (pt - np.array([p.pad_left, p.pad_top])) / p.scale + np.array([p.x0, p.y0])


def frame_to_crop(pt_xy: np.ndarray | tuple[float, float], p: CropParams) -> np.ndarray:
    """Ponto(s) no frame → coordenada(s) no crop (0..size). Aceita ``(2,)`` ou ``(N, 2)``."""
    pt = np.asarray(pt_xy, dtype=np.float64)
    return (pt - np.array([p.x0, p.y0])) * p.scale + np.array([p.pad_left, p.pad_top])


def project_between_crops(
    pt_xy: np.ndarray | tuple[float, float], p_src: CropParams, p_dst: CropParams
) -> np.ndarray:
    """Projeta ponto(s) do crop ``p_src`` para o crop ``p_dst`` via coordenada de frame.

    Uso no Épico #119: o keypoint GT anotado no crop justo vale no frouxo (e vice-versa) sem
    re-anotar — a única diferença é o enquadramento.
    """
    return frame_to_crop(crop_to_frame(pt_xy, p_src), p_dst)
