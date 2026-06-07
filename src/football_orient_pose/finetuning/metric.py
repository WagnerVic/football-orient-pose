"""Metric MMPose que avalia o PCK@0.2 ESTRITO (ref. ombros/quadris) no val loop.

É a mesma métrica do `evaluate.py` (`compute_pck`), então o `save_best='PCK'`
seleciona o checkpoint pela régua do artigo — e não pelo PCK leniente
(`norm_item='bbox'`) que satura em ~20px. PCK é razão (dist/ref), invariante à
escala, então funciona no espaço interno do MMPose (288×384).
"""

from __future__ import annotations

import numpy as np

from mmengine.evaluator import BaseMetric
from mmpose.registry import METRICS

from football_orient_pose.evaluation import compute_pck, compute_pdj
from football_orient_pose.utils.keypoint_mapping import derive_h3wb_centers


@METRICS.register_module()
class StrictPCKMetric(BaseMetric):
    """PCK@0.2 estrito + PDJ@0.5, consistentes com o evaluate.py.

    Deriva os keypoints [0,7,8,9] das predições antes de medir (igual ao GT e ao
    baseline), senão o modelo seria penalizado em joints que treina com visibility=0.
    """

    def __init__(self, pck_thr: float = 0.2, pdj_thr: float = 0.5,
                 collect_device: str = "cpu", prefix: str | None = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.pck_thr = pck_thr
        self.pdj_thr = pdj_thr

    def process(self, data_batch, data_samples) -> None:
        for ds in data_samples:
            pred = np.asarray(ds["pred_instances"]["keypoints"], dtype=np.float32)
            gt = np.asarray(ds["gt_instances"]["keypoints"], dtype=np.float32)
            self.results.append((pred[0], gt[0]))  # (17, 2) cada

    def compute_metrics(self, results) -> dict:
        preds = derive_h3wb_centers(np.stack([p for p, _ in results]))  # (M, 17, 2)
        gts = np.stack([g for _, g in results])
        pck = compute_pck(preds, gts, threshold=self.pck_thr)
        pdj = compute_pdj(preds, gts, threshold=self.pdj_thr)
        return {"PCK": float(pck.global_score), "PDJ": float(pdj.global_score)}
