"""Pipeline ponta-a-ponta: frame → detecção → finalizador → crop → pose (Épico #126).

Costura as peças do diferencial numa única chamada: detecta pessoas (YOLO26x), seleciona o
finalizador, recorta justo (``crop.py``), estima a pose e reprojeta os keypoints para o frame. É
agnóstico ao detector/estimador (recebe qualquer ``Detector``/``BasePoseEstimator``), então roda
local com modelo zero-shot ou com o fine-tunado.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from football_orient_pose.crop import CropParams, crop_to_frame, make_crop
from football_orient_pose.detection import Detector, detections_to_arrays
from football_orient_pose.evaluation.detection_metrics import iou_matrix
from football_orient_pose.pose import BasePoseEstimator


@dataclass
class PipelineResult:
    """Saída do pipeline para um frame."""

    finisher_box: np.ndarray  # (4,) xyxy no frame
    crop: np.ndarray  # (size, size, 3) crop justo do finalizador
    crop_params: CropParams
    keypoints_crop: np.ndarray  # (17, 2) H3WB, espaço do crop
    keypoints_frame: np.ndarray  # (17, 2) H3WB, reprojetado para o frame
    all_boxes: np.ndarray  # (N, 4) todas as detecções (contexto/viz)


def select_finisher(boxes: np.ndarray, ref_box: np.ndarray | None = None) -> int:
    """Índice da caixa do finalizador entre ``boxes`` (N,4).

    Com ``ref_box`` (caixa de referência do 3DSP), escolhe a detecção de **maior IoU** — robusto a
    pequeno offset de frame (jogadores são separados). Sem ela, cai numa heurística (maior área).
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if len(boxes) == 0:
        raise ValueError("nenhuma detecção para selecionar o finalizador")
    if ref_box is not None:
        ious = iou_matrix(np.asarray(ref_box, dtype=np.float64).reshape(1, 4), boxes)[0]
        return int(np.argmax(ious))
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return int(np.argmax(areas))


def _largest_area_idx(boxes: np.ndarray) -> int:
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return int(np.argmax(areas))


def track_finisher(
    boxes_per_frame: list[np.ndarray], ref_boxes: list[np.ndarray | None]
) -> list[int | None]:
    """Rastreia o finalizador ao longo de um clip → índice da caixa escolhida por frame.

    O matching per-frame com a referência do 3DSP é frágil em aglomerado (jogadores colados → IoU
    ambíguo). Em vez disso: **ancora** no frame de **maior IoU** com a referência (identificação
    mais confiável) e **propaga** para frente e para trás seguindo a caixa anterior por IoU — segue
    o *mesmo* jogador, robusto a desalinhamento/ambiguidade do índice de frame.

    Sem referência em nenhum frame, cai na heurística de maior área. ``None`` em frame sem detecção.
    """
    n = len(boxes_per_frame)
    best_iou, anchor_i, anchor_j = -1.0, None, None
    for i in range(n):
        boxes, ref = boxes_per_frame[i], ref_boxes[i]
        if ref is None or len(boxes) == 0:
            continue
        ious = iou_matrix(np.asarray(ref, dtype=np.float64).reshape(1, 4), boxes)[0]
        j = int(np.argmax(ious))
        if ious[j] > best_iou:
            best_iou, anchor_i, anchor_j = float(ious[j]), i, j

    picks: list[int | None] = [None] * n
    if anchor_i is None:  # sem referência em nenhum frame
        for i, boxes in enumerate(boxes_per_frame):
            picks[i] = _largest_area_idx(boxes) if len(boxes) else None
        return picks

    picks[anchor_i] = anchor_j
    # propaga seguindo a caixa do frame anterior (frente e trás a partir da âncora)
    for direction in (range(anchor_i + 1, n), range(anchor_i - 1, -1, -1)):
        prev = boxes_per_frame[anchor_i][anchor_j]
        for i in direction:
            boxes = boxes_per_frame[i]
            if len(boxes) == 0:
                picks[i] = None
                continue
            ious = iou_matrix(np.asarray(prev, dtype=np.float64).reshape(1, 4), boxes)[0]
            j = int(np.argmax(ious))
            picks[i] = j
            prev = boxes[j]
    return picks


def crop_and_pose(
    frame: np.ndarray,
    finisher_box: np.ndarray,
    pose_estimator: BasePoseEstimator,
    all_boxes: np.ndarray | None = None,
    crop_mode: str = "tight",
) -> PipelineResult:
    """Crop justo da caixa + pose + reprojeção (dado o finalizador já selecionado)."""
    finisher_box = np.asarray(finisher_box, dtype=np.float32).reshape(4)
    crop, params = make_crop(frame, finisher_box, mode=crop_mode)
    keypoints_crop = pose_estimator.predict_h3wb(crop)  # (17, 2) H3WB no espaço do crop
    keypoints_frame = crop_to_frame(keypoints_crop, params)
    return PipelineResult(
        finisher_box=finisher_box,
        crop=crop,
        crop_params=params,
        keypoints_crop=np.asarray(keypoints_crop, dtype=np.float64),
        keypoints_frame=np.asarray(keypoints_frame, dtype=np.float64),
        all_boxes=np.empty((0, 4)) if all_boxes is None else np.asarray(all_boxes),
    )


def pose_all(
    frame: np.ndarray,
    boxes: np.ndarray,
    pose_estimator: BasePoseEstimator,
    crop_mode: str = "tight",
    min_box_height: float = 0.0,
) -> list[PipelineResult]:
    """Estima a pose de **todos** os jogadores detectados (replica o baseline Reis et al.).

    Para cada caixa em ``boxes`` (N,4 xyxy), recorta justo, estima a pose e reprojeta os keypoints
    para o frame — um ``PipelineResult`` por jogador. Diferente do caminho do finalizador, aqui não
    há seleção/rastreamento: poseia todo mundo (mesma ideia da Fig. 5 do Reis, que re-cola todos os
    esqueletos no frame).

    ``min_box_height`` descarta caixas mais baixas que esse limiar (jogador distante → crop ruim →
    esqueleto embolado). Os ``crop`` ficam nos resultados em memória; cabe ao chamador descartá-los
    (não são persistidos no showcase).
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    heights = boxes[:, 3] - boxes[:, 1]
    kept = boxes[heights >= min_box_height]
    return [
        crop_and_pose(frame, box, pose_estimator, all_boxes=boxes, crop_mode=crop_mode)
        for box in kept
    ]


def run_pipeline(
    frame: np.ndarray,
    detector: Detector,
    pose_estimator: BasePoseEstimator,
    ref_box: np.ndarray | None = None,
    crop_mode: str = "tight",
) -> PipelineResult:
    """Roda detecção → seleção do finalizador → crop → pose → reprojeção, para um frame.

    Para um clip inteiro (rastreamento robusto), use ``track_finisher`` + ``crop_and_pose``.
    """
    boxes, _scores = detections_to_arrays(detector.detect(frame))
    if len(boxes) == 0:
        raise ValueError("nenhuma pessoa detectada no frame")
    finisher_box = boxes[select_finisher(boxes, ref_box)]
    return crop_and_pose(frame, finisher_box, pose_estimator, all_boxes=boxes, crop_mode=crop_mode)
