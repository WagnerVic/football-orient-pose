"""Subpacote de avaliação: métricas e comparação com baseline."""

from football_orient_pose.evaluation.metrics import (
    PDJ_GROUPS,
    PDJResult,
    PCK_LEFT_HIP_ID,
    PCK_LEFT_SHOULDER_ID,
    PCK_RIGHT_HIP_ID,
    PCK_RIGHT_SHOULDER_ID,
    PCKResult,
    compute_pdj,
    compute_pck,
    pdj_auc,
    pdj_curve,
)

__all__ = [
    "PDJ_GROUPS",
    "PDJResult",
    "PCK_LEFT_HIP_ID",
    "PCK_LEFT_SHOULDER_ID",
    "PCK_RIGHT_HIP_ID",
    "PCK_RIGHT_SHOULDER_ID",
    "PCKResult",
    "compute_pdj",
    "compute_pck",
    "pdj_auc",
    "pdj_curve",
]
