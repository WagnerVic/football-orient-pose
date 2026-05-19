"""Wrappers concretos de estimadores de pose."""

from football_orient_pose.estimators.hrnet import HRNetEstimator
from football_orient_pose.estimators.openpose import OpenPoseEstimator
from football_orient_pose.estimators.rtmpose import RTMPoseEstimator

__all__ = ["HRNetEstimator", "OpenPoseEstimator", "RTMPoseEstimator"]
