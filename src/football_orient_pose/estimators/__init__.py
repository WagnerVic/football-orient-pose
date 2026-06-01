"""Wrappers concretos de estimadores de pose.

Importações são lazy para que importar um único estimador não force as
dependências opcionais dos outros (p.ex., rtmlib não requer cv2, e vice-versa).
"""

from __future__ import annotations

__all__ = ["HRNetEstimator", "OpenPoseEstimator", "RTMPoseEstimator"]


def __getattr__(name: str):
    if name == "RTMPoseEstimator":
        from football_orient_pose.estimators.rtmpose import RTMPoseEstimator
        return RTMPoseEstimator
    if name == "HRNetEstimator":
        from football_orient_pose.estimators.hrnet import HRNetEstimator
        return HRNetEstimator
    if name == "OpenPoseEstimator":
        from football_orient_pose.estimators.openpose import OpenPoseEstimator
        return OpenPoseEstimator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
