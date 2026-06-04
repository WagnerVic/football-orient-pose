"""Mapeamento de keypoints entre formatos COCO-17 e H3WB-17 (3DSP dataset).

O dataset 3DSP usa formato H3WB/MotionAGFormer com 17 keypoints que diferem
completamente do formato COCO-17 padrão usado por modelos como RTMPose, HRNet
e OpenPose. Este módulo implementa a conversão bidirecional.

Referência:
    - Código original: calvinyeungck/3D-Shot-Posture-Dataset (demo.py, L247-274)
    - Licença: Apache 2.0 — Yeung, Ide, Fujii (CVPRW 2024)
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Constantes — nomes dos keypoints em cada formato
# ---------------------------------------------------------------------------

COCO17_NAMES: list[str] = [
    "nose",              # 0
    "left_eye",          # 1
    "right_eye",         # 2
    "left_ear",          # 3
    "right_ear",         # 4
    "left_shoulder",     # 5
    "right_shoulder",    # 6
    "left_elbow",        # 7
    "right_elbow",       # 8
    "left_wrist",        # 9
    "right_wrist",       # 10
    "left_hip",          # 11
    "right_hip",         # 12
    "left_knee",         # 13
    "right_knee",        # 14
    "left_ankle",        # 15
    "right_ankle",       # 16
]

H3WB17_NAMES: list[str] = [
    "Center of Hips",       # 0  ← calculado: (COCO 11 + 12) / 2
    "Left Hip",             # 1  ← COCO 11
    "Left Knee",            # 2  ← COCO 13
    "Left Ankle",           # 3  ← COCO 15
    "Right Hip",            # 4  ← COCO 12
    "Right Knee",           # 5  ← COCO 14
    "Right Ankle",          # 6  ← COCO 16
    "Center of Body",       # 7  ← calculado: (H3WB 0 + H3WB 8) / 2
    "Center of Shoulder",   # 8  ← calculado: (COCO 5 + 6) / 2
    "Neck",                 # 9  ← calculado: (COCO 0 + H3WB 8) / 2
    "Head",                 # 10 ← COCO 0 (nose)
    "Right Shoulder",       # 11 ← COCO 6
    "Right Elbow",          # 12 ← COCO 8
    "Right Wrist",          # 13 ← COCO 10
    "Left Shoulder",        # 14 ← COCO 5
    "Left Elbow",           # 15 ← COCO 7
    "Left Wrist",           # 16 ← COCO 9
]

# IDs dos keypoints H3WB relevantes para orientação corporal
ORIENTATION_KP_IDS = {
    "left_shoulder": 14,
    "right_shoulder": 11,
    "left_hip": 1,
    "right_hip": 4,
    "center_hips": 0,
    "center_shoulder": 8,
    "center_body": 7,
}


def coco17_to_h3wb17(keypoints_coco: np.ndarray) -> np.ndarray:
    """Converte keypoints COCO-17 para formato H3WB-17 (3DSP).

    4 dos 17 keypoints H3WB são **calculados** (médias de keypoints COCO),
    não mapeados diretamente. Os 5 keypoints faciais do COCO (eyes, ears)
    não existem no H3WB e são descartados.

    Quando ``D == 3`` (x, y, confidence), a média dos joints calculados
    ignora joints com ``confidence == 0`` — se apenas um lado for detectado,
    usa-o diretamente em vez de puxar o resultado para (0, 0).

    Parameters
    ----------
    keypoints_coco : np.ndarray
        Array de shape ``(17, D)`` onde D é 2 (x, y) ou 3 (x, y, confidence).
        Índices seguem o formato COCO-17 padrão.

    Returns
    -------
    np.ndarray
        Array de shape ``(17, 2)`` no formato H3WB-17 (3DSP), apenas x e y.
    """
    kp = keypoints_coco
    has_conf = kp.shape[-1] >= 3

    def _det(i: int) -> bool:
        return bool(kp[i, 2] > 0) if has_conf else True

    def _avg(i: int, j: int) -> np.ndarray:
        """Média de dois joints COCO; ignora joint não detectado quando possível."""
        di, dj = _det(i), _det(j)
        if di and dj:
            return (kp[i, :2] + kp[j, :2]) / 2
        if di:
            return kp[i, :2].copy()
        if dj:
            return kp[j, :2].copy()
        return (kp[i, :2] + kp[j, :2]) / 2  # ambos não detectados: melhor esforço

    def _avg_point_computed(pt_idx: int, computed: np.ndarray) -> np.ndarray:
        """Média entre um joint COCO e um ponto já calculado."""
        if has_conf and not _det(pt_idx):
            return computed.copy()
        return (kp[pt_idx, :2] + computed) / 2

    center_hips = _avg(11, 12)
    center_shoulder = _avg(5, 6)
    center_body = (center_hips + center_shoulder) / 2
    neck = _avg_point_computed(0, center_shoulder)  # nose + center_shoulder

    h3wb = np.array([
        center_hips,         # 0:  Center of Hips
        kp[11, :2],          # 1:  Left Hip
        kp[13, :2],          # 2:  Left Knee
        kp[15, :2],          # 3:  Left Ankle
        kp[12, :2],          # 4:  Right Hip
        kp[14, :2],          # 5:  Right Knee
        kp[16, :2],          # 6:  Right Ankle
        center_body,         # 7:  Center of Body
        center_shoulder,     # 8:  Center of Shoulder
        neck,                # 9:  Neck
        kp[0, :2],           # 10: Head (nose)
        kp[6, :2],           # 11: Right Shoulder
        kp[8, :2],           # 12: Right Elbow
        kp[10, :2],          # 13: Right Wrist
        kp[5, :2],           # 14: Left Shoulder
        kp[7, :2],           # 15: Left Elbow
        kp[9, :2],           # 16: Left Wrist
    ], dtype=np.float32)

    return h3wb


def coco17_to_h3wb17_batch(keypoints_batch: np.ndarray) -> np.ndarray:
    """Versão batch de :func:`coco17_to_h3wb17`.

    Parameters
    ----------
    keypoints_batch : np.ndarray
        Array de shape ``(N, 17, D)`` no formato COCO-17, D=2 ou D=3.

    Returns
    -------
    np.ndarray
        Array de shape ``(N, 17, 2)`` no formato H3WB-17.
    """
    return np.array([coco17_to_h3wb17(kp) for kp in keypoints_batch])


# Mapeamento direto COCO ID → H3WB ID (apenas para keypoints não-calculados)
COCO_TO_H3WB_DIRECT: dict[int, int] = {
    0: 10,   # nose → Head
    5: 14,   # left_shoulder → Left Shoulder
    6: 11,   # right_shoulder → Right Shoulder
    7: 15,   # left_elbow → Left Elbow
    8: 12,   # right_elbow → Right Elbow
    9: 16,   # left_wrist → Left Wrist
    10: 13,  # right_wrist → Right Wrist
    11: 1,   # left_hip → Left Hip
    12: 4,   # right_hip → Right Hip
    13: 2,   # left_knee → Left Knee
    14: 5,   # right_knee → Right Knee
    15: 3,   # left_ankle → Left Ankle
    16: 6,   # right_ankle → Right Ankle
}

# Keypoints COCO sem correspondência direta no H3WB (faciais)
COCO_DISCARDED_IDS: list[int] = [1, 2, 3, 4]  # left/right eye/ear

# Keypoints H3WB que são calculados (não existem no COCO)
H3WB_COMPUTED_IDS: list[int] = [0, 7, 8, 9]  # center_hips, center_body, center_shoulder, neck


def derive_h3wb_centers(keypoints: np.ndarray) -> np.ndarray:
    """Preenche os 4 keypoints derivados H3WB (IDs 0,7,8,9) como médias dos
    anotados, mesma definição usada no GT e em :func:`coco17_to_h3wb17`.

    Esses 4 são treinados com ``visibility=0`` (não supervisionados), então o
    modelo não os prediz; derivá-los das predições torna a avaliação justa vs.
    o baseline (que também os calcula como médias).

    Parameters
    ----------
    keypoints : np.ndarray
        Shape ``(..., 17, 2)`` em ordem H3WB. Não é modificado in-place.
    """
    k = keypoints.copy()
    k[..., 0, :] = (k[..., 1, :] + k[..., 4, :]) / 2      # center_hips = avg(L_hip, R_hip)
    k[..., 8, :] = (k[..., 11, :] + k[..., 14, :]) / 2    # center_shoulder = avg(R_sh, L_sh)
    k[..., 7, :] = (k[..., 0, :] + k[..., 8, :]) / 2      # center_body = avg(hips, shoulders)
    k[..., 9, :] = (k[..., 10, :] + k[..., 8, :]) / 2     # neck ~ avg(head, center_shoulder)
    return k
