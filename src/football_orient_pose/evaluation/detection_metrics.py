"""Métricas de detecção de pessoas (Épico #113): COCOeval + PR/F1 no ponto de operação.

O GT humano já está em COCO (export Roboflow, ver ``roboflow_io``), então o motor principal é o
``pycocotools.COCOeval`` — padrão-ouro para mAP@[.5:.95], AP50/AP75 e **recall por tamanho**
(AR small/medium/large, resolve a #116). Soma-se uma função self-contained de precision/recall/F1
no ponto de operação (conf+IoU fixos), que o COCOeval não expõe diretamente.

Convenção de caixas: ``xyxy`` (``[x1, y1, x2, y2]``, pixel de frame) — igual à saída do
``detection.py`` e ao GT do ``roboflow_io``. A conversão para o ``xywh`` do COCO é feita aqui.
"""

from __future__ import annotations

import contextlib
import io
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pycocotools.coco import COCO

PERSON_CAT_ID = 1
PERSON_CAT = {"id": PERSON_CAT_ID, "name": "person", "supercategory": "person"}

# Índices de COCOeval.stats (bbox): a ordem é fixa na lib.
_STAT_KEYS = [
    "mAP",  # 0: AP @[.5:.95]
    "AP50",  # 1: AP @.50
    "AP75",  # 2: AP @.75
    "AP_small",  # 3
    "AP_medium",  # 4
    "AP_large",  # 5
    "AR_1",  # 6: AR dado 1 det/imagem
    "AR_10",  # 7
    "AR_100",  # 8
    "AR_small",  # 9: recall por tamanho (#116)
    "AR_medium",  # 10
    "AR_large",  # 11
]


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU par-a-par entre dois conjuntos de caixas xyxy. Retorna ``(len(a), len(b))``."""
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)

    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def precision_recall_f1(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_thr: float = 0.5,
    score_thr: float = 0.5,
) -> dict[str, float]:
    """Precision/Recall/F1 no ponto de operação (``score_thr``, ``iou_thr``).

    Matching guloso: predições acima de ``score_thr`` ordenadas por confiança decrescente;
    cada uma casa com o GT de maior IoU ainda livre (≥ ``iou_thr``). Casadas = TP; predições
    sobrando = FP; GT sem casar = FN.
    """
    pred_boxes = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
    pred_scores = np.asarray(pred_scores, dtype=np.float64).reshape(-1)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)

    keep = pred_scores >= score_thr
    pred_boxes, pred_scores = pred_boxes[keep], pred_scores[keep]
    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]

    n_gt = len(gt_boxes)
    matched_gt = np.zeros(n_gt, dtype=bool)
    tp = 0
    ious = iou_matrix(pred_boxes, gt_boxes)
    for i in range(len(pred_boxes)):
        if n_gt == 0:
            break
        cand = ious[i].copy()
        cand[matched_gt] = -1.0
        j = int(np.argmax(cand))
        if cand[j] >= iou_thr:
            matched_gt[j] = True
            tp += 1
    fp = len(pred_boxes) - tp
    fn = n_gt - tp

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "iou_thr": iou_thr,
        "score_thr": score_thr,
    }


def _xyxy_to_xywh(box: np.ndarray) -> list[float]:
    x1, y1, x2, y2 = (float(v) for v in box)
    return [x1, y1, x2 - x1, y2 - y1]


def build_coco_gt(
    gt: dict[str, dict[int, np.ndarray]],
) -> tuple[COCO, dict[tuple[str, int], int]]:
    """Converte ``{clip_id: {frame_idx: (N,4) xyxy}}`` num objeto ``COCO`` em memória.

    Retorna ``(coco, image_id_map)`` onde ``image_id_map[(clip_id, frame_idx)] = image_id`` único —
    o mesmo mapa deve ser usado em ``build_coco_dt`` para alinhar predições e GT.
    """
    from pycocotools.coco import COCO

    images: list[dict] = []
    annotations: list[dict] = []
    image_id_map: dict[tuple[str, int], int] = {}
    img_id = 0
    ann_id = 1
    for clip_id in sorted(gt):
        for frame_idx in sorted(gt[clip_id]):
            image_id_map[(clip_id, frame_idx)] = img_id
            images.append({"id": img_id, "file_name": f"{clip_id}/{frame_idx:03d}.jpg"})
            for box in gt[clip_id][frame_idx]:
                xywh = _xyxy_to_xywh(box)
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": PERSON_CAT_ID,
                        "bbox": xywh,
                        "area": xywh[2] * xywh[3],
                        "iscrowd": 0,
                    }
                )
                ann_id += 1
            img_id += 1

    coco = COCO()
    coco.dataset = {"images": images, "annotations": annotations, "categories": [PERSON_CAT]}
    with contextlib.redirect_stdout(io.StringIO()):  # createIndex é verboso
        coco.createIndex()
    return coco, image_id_map


def build_coco_dt(
    preds: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]],
    image_id_map: dict[tuple[str, int], int],
) -> list[dict]:
    """Resultados COCO a partir de ``{clip_id: {frame_idx: (boxes (N,4), scores (N,))}}``."""
    results: list[dict] = []
    for clip_id in preds:
        for frame_idx, (boxes, scores) in preds[clip_id].items():
            image_id = image_id_map.get((clip_id, frame_idx))
            if image_id is None:
                continue
            boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
            scores = np.asarray(scores, dtype=np.float64).reshape(-1)
            for box, score in zip(boxes, scores):
                results.append(
                    {
                        "image_id": image_id,
                        "category_id": PERSON_CAT_ID,
                        "bbox": _xyxy_to_xywh(box),
                        "score": float(score),
                    }
                )
    return results


def cocoeval_stats(coco_gt: COCO, coco_dt_results: list[dict]) -> dict[str, float]:
    """Roda COCOeval (bbox) e devolve os 12 stats nomeados (ver ``_STAT_KEYS``)."""
    from pycocotools.cocoeval import COCOeval

    if not coco_dt_results:  # sem predições: COCOeval quebra com results vazio
        return {k: 0.0 for k in _STAT_KEYS}

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(coco_dt_results)
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    stats = [round(float(v), 6) for v in ev.stats]
    return dict(zip(_STAT_KEYS, stats))
