"""Métricas de avaliação para keypoints 2D."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PDJ_GROUPS: dict[str, list[int]] = {
    "head": [10],
    "shoulder": [8, 11, 14],
    "elbow": [12, 15],
    "wrist": [13, 16],
    "hip": [0, 1, 4],
    "knee": [2, 5],
    "ankle": [3, 6],
}

PCK_LEFT_SHOULDER_ID = 14
PCK_RIGHT_SHOULDER_ID = 11
PCK_LEFT_HIP_ID = 1
PCK_RIGHT_HIP_ID = 4

H3WB_SIGMAS: np.ndarray = np.array([
    0.107,  # 0:  Center of Hips
    0.107,  # 1:  Left Hip
    0.087,  # 2:  Left Knee
    0.089,  # 3:  Left Ankle
    0.107,  # 4:  Right Hip
    0.087,  # 5:  Right Knee
    0.089,  # 6:  Right Ankle
    0.093,  # 7:  Center of Body
    0.079,  # 8:  Center of Shoulder
    0.052,  # 9:  Neck
    0.026,  # 10: Head
    0.079,  # 11: Right Shoulder
    0.072,  # 12: Right Elbow
    0.062,  # 13: Right Wrist
    0.079,  # 14: Left Shoulder
    0.072,  # 15: Left Elbow
    0.062,  # 16: Left Wrist
], dtype=np.float32)

DSP_CROP_AREA: float = 10_000.0  # 100×100 px²


@dataclass(frozen=True)
class PDJResult:
    """Resultado agregado de PDJ."""

    threshold: float
    global_score: float
    per_joint: np.ndarray
    per_group: dict[str, float]
    valid_frames: int


def compute_pdj(
    predicted: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
    torso_ids: tuple[int, int] = (0, 8),
) -> PDJResult:
    """Calcula Percent of Detected Joints (PDJ) para keypoints H3WB-17.

    Um keypoint é considerado correto quando a distância Euclidiana entre
    predição e ground truth, normalizada pela distância entre Center of Hips
    (ID 0) e Center of Shoulder (ID 8), é menor que ``threshold``.

    Parameters
    ----------
    predicted : np.ndarray
        Keypoints preditos em shape ``(17, 2)`` ou ``(N, 17, 2)``.
    target : np.ndarray
        Keypoints ground truth em shape ``(17, 2)`` ou ``(N, 17, 2)``.
    threshold : float
        Limiar de correção no espaço normalizado. O padrão do 3DSP é ``0.5``.
    torso_ids : tuple[int, int]
        IDs H3WB usados para normalização. O padrão é ``(0, 8)``.

    Returns
    -------
    PDJResult
        Score global, score por keypoint, score por grupo anatômico e número
        de frames válidos.
    """
    pred = _ensure_keypoint_batch(predicted, "predicted")
    gt = _ensure_keypoint_batch(target, "target")
    if pred.shape != gt.shape:
        raise ValueError(f"predicted e target devem ter o mesmo shape: {pred.shape} != {gt.shape}")

    normalizer = np.linalg.norm(gt[:, torso_ids[0]] - gt[:, torso_ids[1]], axis=1)
    valid = normalizer > 0
    if not np.any(valid):
        raise ValueError("Nenhum frame possui distância torso válida para normalizar o PDJ")

    distances = np.linalg.norm(pred[valid] - gt[valid], axis=2)
    normalized_distances = distances / normalizer[valid, np.newaxis]
    correct = normalized_distances < threshold

    per_joint = correct.mean(axis=0)
    per_group = {
        group_name: float(per_joint[joint_ids].mean())
        for group_name, joint_ids in PDJ_GROUPS.items()
    }
    return PDJResult(
        threshold=threshold,
        global_score=float(correct.mean()),
        per_joint=per_joint.astype(np.float32),
        per_group=per_group,
        valid_frames=int(valid.sum()),
    )


@dataclass(frozen=True)
class PCKResult:
    """Resultado agregado de PCK."""

    threshold: float
    global_score: float
    per_joint: np.ndarray
    per_group: dict[str, float]
    valid_frames: int


def compute_pck(
    predicted: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.2,
) -> PCKResult:
    """Calcula PCK (Percentage of Correct Keypoints) para keypoints H3WB-17.

    Referência = max(dist(LeftShoulder[14], RightShoulder[11]),
                     dist(LeftHip[1], RightHip[4])).
    Threshold padrão 0.2 (PCK@0.2 é o mais comum na literatura).
    """
    pred = _ensure_keypoint_batch(predicted, "predicted")
    gt = _ensure_keypoint_batch(target, "target")
    if pred.shape != gt.shape:
        raise ValueError(
            f"predicted e target devem ter o mesmo shape: {pred.shape} != {gt.shape}"
        )

    shoulder_width = np.linalg.norm(
        gt[:, PCK_LEFT_SHOULDER_ID] - gt[:, PCK_RIGHT_SHOULDER_ID], axis=1
    )
    hip_width = np.linalg.norm(
        gt[:, PCK_LEFT_HIP_ID] - gt[:, PCK_RIGHT_HIP_ID], axis=1
    )
    ref_size = np.maximum(shoulder_width, hip_width)
    valid = ref_size > 0

    if not np.any(valid):
        raise ValueError("Nenhum frame possui referência de tamanho válida para PCK")

    distances = np.linalg.norm(pred[valid] - gt[valid], axis=2)
    normalized = distances / ref_size[valid, np.newaxis]
    correct = normalized < threshold

    per_joint = correct.mean(axis=0)
    per_group = {
        group_name: float(per_joint[joint_ids].mean())
        for group_name, joint_ids in PDJ_GROUPS.items()
    }
    return PCKResult(
        threshold=threshold,
        global_score=float(correct.mean()),
        per_joint=per_joint.astype(np.float32),
        per_group=per_group,
        valid_frames=int(valid.sum()),
    )


@dataclass(frozen=True)
class OKSResult:
    """Resultado OKS com AP calculado em limiares COCO."""

    global_oks: float
    per_joint_oks: np.ndarray
    ap: float
    ap50: float
    ap75: float
    ap_per_threshold: dict[float, float]
    valid_frames: int


def compute_oks(
    predicted: np.ndarray,
    target: np.ndarray,
    area: float = DSP_CROP_AREA,
    sigmas: np.ndarray | None = None,
) -> OKSResult:
    """Calcula OKS e AP para keypoints H3WB-17.

    OKS por frame: mean(exp(-d_j² / (2 * s² * sigma_j²)))
    AP@t: fração de frames com OKS >= t (avaliação single-instance).
    mAP: média de AP@t para t em [0.50, 0.55, ..., 0.95].
    """
    pred = _ensure_keypoint_batch(predicted, "predicted")
    gt = _ensure_keypoint_batch(target, "target")
    if pred.shape != gt.shape:
        raise ValueError(
            f"predicted e target devem ter o mesmo shape: {pred.shape} != {gt.shape}"
        )

    if sigmas is None:
        sigmas = H3WB_SIGMAS

    s = float(np.sqrt(area))
    d = np.linalg.norm(pred - gt, axis=2)                     # (N, 17)
    oks_per_joint = np.exp(-d**2 / (2.0 * s**2 * sigmas**2))  # (N, 17)
    oks_per_frame = oks_per_joint.mean(axis=1)                 # (N,)

    thresholds = [round(0.5 + i * 0.05, 2) for i in range(10)]
    ap_per_threshold = {
        t: float((oks_per_frame >= t).mean()) for t in thresholds
    }

    return OKSResult(
        global_oks=float(oks_per_frame.mean()),
        per_joint_oks=oks_per_joint.mean(axis=0).astype(np.float32),
        ap=float(np.mean(list(ap_per_threshold.values()))),
        ap50=ap_per_threshold[0.5],
        ap75=ap_per_threshold[0.75],
        ap_per_threshold=ap_per_threshold,
        valid_frames=len(pred),
    )


def pdj_curve(
    predicted: np.ndarray,
    target: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula a curva PDJ para uma sequência de limiares."""
    if thresholds is None:
        thresholds = np.linspace(0.0, 0.5, 51, dtype=np.float32)

    scores = np.array(
        [
            compute_pdj(predicted, target, threshold=float(threshold)).global_score
            for threshold in thresholds
        ],
        dtype=np.float32,
    )
    return thresholds.astype(np.float32), scores


def pdj_auc(
    predicted: np.ndarray,
    target: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> float:
    """Calcula AUC normalizada da curva PDJ."""
    curve_thresholds, scores = pdj_curve(predicted, target, thresholds)
    max_threshold = float(curve_thresholds[-1])
    if max_threshold <= 0:
        raise ValueError("O maior threshold da curva PDJ deve ser positivo")
    return float(np.trapezoid(scores, curve_thresholds) / max_threshold)


def _ensure_keypoint_batch(keypoints: np.ndarray, name: str) -> np.ndarray:
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.shape == (17, 2):
        return keypoints[np.newaxis, ...]
    if keypoints.ndim == 3 and keypoints.shape[1:] == (17, 2):
        return keypoints
    raise ValueError(f"{name} deve ter shape (17, 2) ou (N, 17, 2), recebido {keypoints.shape}")
