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


def run_pipeline(
    frame: np.ndarray,
    detector: Detector,
    pose_estimator: BasePoseEstimator,
    ref_box: np.ndarray | None = None,
    crop_mode: str = "tight",
) -> PipelineResult:
    """Roda detecção → seleção do finalizador → crop → pose → reprojeção, para um frame."""
    boxes, _scores = detections_to_arrays(detector.detect(frame))
    if len(boxes) == 0:
        raise ValueError("nenhuma pessoa detectada no frame")

    idx = select_finisher(boxes, ref_box)
    finisher_box = boxes[idx]

    crop, params = make_crop(frame, finisher_box, mode=crop_mode)
    keypoints_crop = pose_estimator.predict_h3wb(crop)  # (17, 2) H3WB no espaço do crop
    keypoints_frame = crop_to_frame(keypoints_crop, params)

    return PipelineResult(
        finisher_box=finisher_box,
        crop=crop,
        crop_params=params,
        keypoints_crop=np.asarray(keypoints_crop, dtype=np.float64),
        keypoints_frame=np.asarray(keypoints_frame, dtype=np.float64),
        all_boxes=boxes,
    )
