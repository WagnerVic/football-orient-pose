"""Ingestão das anotações de bbox exportadas do Roboflow (COCO) → GT interno.

Cada example foi anotado num projeto Roboflow separado e exportado em COCO
(``data/annotations/examples_bbox/<clip_id>/_annotations.coco.json``). O nome da
pasta é o ``clip_id`` (ex.: ``example_01``) e o ``file_name`` de cada imagem
preserva o índice do frame (``009_jpg.rf.<hash>.jpg`` → frame 9), casando com
``data/clips/examples/<clip_id>/img/{idx:03d}.jpg``.

As caixas são devolvidas em **xyxy** (``float32``, coordenada de frame), igual ao
contrato ``Detection`` em ``detection.py``, para o benchmark dos detectores
(Épico #113) comparar GT × predição no mesmo formato. O COCO guarda ``[x,y,w,h]``;
a conversão para ``[x1,y1,x2,y2]`` é feita aqui.

Cobre as issues #108 (ingestão de bbox).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

# O Roboflow exporta um "supercategory" raiz vazio (id 0) + a classe real.
# A classe de pessoa anotada neste projeto chama-se "Jogador".
DEFAULT_CLASS = "Jogador"

# file_name começa com o índice do frame original ("009_jpg.rf.<hash>.jpg" → 9)
_FRAME_RE = re.compile(r"^(\d+)")


def _frame_index(file_name: str) -> int:
    match = _FRAME_RE.match(file_name)
    if not match:
        raise ValueError(f"file_name sem índice numérico no início: {file_name!r}")
    return int(match.group(1))


def load_coco_bboxes(
    coco_path: str | Path, class_name: str = DEFAULT_CLASS
) -> dict[int, np.ndarray]:
    """Lê um COCO do Roboflow → ``{frame_idx (1-based): (N, 4) xyxy float32}``.

    Todos os frames presentes no arquivo entram no dicionário; frames sem caixa
    da classe alvo recebem um array vazio ``(0, 4)`` (útil para o eval contar
    falsos positivos onde não há GT).
    """
    coco_path = Path(coco_path)
    data = json.loads(coco_path.read_text(encoding="utf-8"))

    cat_ids = {c["id"] for c in data["categories"] if c["name"] == class_name}
    if not cat_ids:
        nomes = [c["name"] for c in data["categories"]]
        raise ValueError(f"classe {class_name!r} ausente em {coco_path} (categorias: {nomes})")

    frame_of = {img["id"]: _frame_index(img["file_name"]) for img in data["images"]}
    boxes: dict[int, list[list[float]]] = {fi: [] for fi in frame_of.values()}
    for ann in data["annotations"]:
        if ann["category_id"] not in cat_ids:
            continue
        x, y, w, h = ann["bbox"]
        boxes[frame_of[ann["image_id"]]].append([x, y, x + w, y + h])

    return {
        fi: np.asarray(bs, dtype="float32").reshape(-1, 4) for fi, bs in sorted(boxes.items())
    }


def load_clip_bbox_gt(
    clip_dir: str | Path, class_name: str = DEFAULT_CLASS
) -> dict[int, np.ndarray]:
    """Acha o ``_annotations.coco.json`` em ``clip_dir`` (em qualquer subpasta) e o carrega."""
    clip_dir = Path(clip_dir)
    matches = sorted(clip_dir.rglob("_annotations.coco.json"))
    if not matches:
        raise FileNotFoundError(f"sem _annotations.coco.json em {clip_dir}")
    if len(matches) > 1:
        raise ValueError(f"múltiplos COCO em {clip_dir}: {matches}")
    return load_coco_bboxes(matches[0], class_name=class_name)


def load_examples_bbox_gt(
    root: str | Path = "data/annotations/examples_bbox", class_name: str = DEFAULT_CLASS
) -> dict[str, dict[int, np.ndarray]]:
    """Carrega o GT de bbox de todos os clips sob ``root``.

    Retorna ``{clip_id: {frame_idx: (N, 4) xyxy}}``, onde ``clip_id`` é o nome da
    pasta (ex.: ``example_01``) — casa com ``data/clips/examples/<clip_id>``.
    """
    root = Path(root)
    clips = sorted(p for p in root.iterdir() if p.is_dir())
    if not clips:
        raise FileNotFoundError(f"nenhum clip em {root}")
    return {p.name: load_clip_bbox_gt(p, class_name=class_name) for p in clips}
