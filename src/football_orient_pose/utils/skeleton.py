"""Definições de esqueleto para formatos COCO-17 e H3WB-17.

Contém as conexões entre joints (bones), cores de visualização, e grupos
de lateralidade para ambos os formatos de keypoints usados no projeto.

Referência:
    - COCO17: calvinyeungck/3D-Shot-Posture-Dataset (rtmlib/visualization/skeleton/coco17.py)
    - H3WB:   calvinyeungck/3D-Shot-Posture-Dataset (pre_process.py, L149-150)
    - Licença: Apache 2.0 — Yeung, Ide, Fujii (CVPRW 2024)
"""

from __future__ import annotations

import numpy as np

# ===========================================================================
# H3WB-17 Skeleton (formato do dataset 3DSP)
# ===========================================================================

# Conexões do esqueleto H3WB: pares (parent, child)
# Extraído de pre_process.py L149-150
H3WB_SKELETON_I = np.array([0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
H3WB_SKELETON_J = np.array([1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

H3WB_BONES: list[tuple[int, int]] = list(zip(H3WB_SKELETON_I.tolist(), H3WB_SKELETON_J.tolist()))
"""16 conexões (bones) do esqueleto H3WB-17.

::

    Bone 0:  Center of Hips(0)     → Left Hip(1)
    Bone 1:  Center of Hips(0)     → Right Hip(4)
    Bone 2:  Left Hip(1)           → Left Knee(2)
    Bone 3:  Right Hip(4)          → Right Knee(5)
    Bone 4:  Left Knee(2)          → Left Ankle(3)
    Bone 5:  Right Knee(5)         → Right Ankle(6)
    Bone 6:  Center of Hips(0)     → Center of Body(7)
    Bone 7:  Center of Body(7)     → Center of Shoulder(8)
    Bone 8:  Center of Shoulder(8) → Left Shoulder(14)
    Bone 9:  Center of Shoulder(8) → Right Shoulder(11)
    Bone 10: Left Shoulder(14)     → Left Elbow(15)
    Bone 11: Left Elbow(15)        → Left Wrist(16)
    Bone 12: Right Shoulder(11)    → Right Elbow(12)
    Bone 13: Right Elbow(12)       → Right Wrist(13)
    Bone 14: Center of Shoulder(8) → Neck(9)
    Bone 15: Neck(9)               → Head(10)
"""

# Lateralidade H3WB (perspectiva anatômica do jogador)
H3WB_JOINTS_LEFT: list[int] = [1, 2, 3, 14, 15, 16]
"""Left Hip, Left Knee, Left Ankle, Left Shoulder, Left Elbow, Left Wrist"""

H3WB_JOINTS_RIGHT: list[int] = [4, 5, 6, 11, 12, 13]
"""Right Hip, Right Knee, Right Ankle, Right Shoulder, Right Elbow, Right Wrist"""

H3WB_JOINTS_CENTER: list[int] = [0, 7, 8, 9, 10]
"""Center of Hips, Center of Body, Center of Shoulder, Neck, Head"""

# Cores para visualização H3WB (BGR — convenção OpenCV)
H3WB_COLORS = {
    "left": (0, 255, 0),       # verde — lado esquerdo
    "right": (0, 128, 255),    # laranja — lado direito
    "center": (255, 153, 51),  # azul — centro/cabeça
}


# ===========================================================================
# COCO-17 Skeleton (formato de output dos modelos)
# ===========================================================================

COCO17_BONES: list[tuple[int, int]] = [
    (15, 13),   # left_ankle → left_knee
    (13, 11),   # left_knee → left_hip
    (16, 14),   # right_ankle → right_knee
    (14, 12),   # right_knee → right_hip
    (11, 12),   # left_hip → right_hip
    (5, 11),    # left_shoulder → left_hip
    (6, 12),    # right_shoulder → right_hip
    (5, 6),     # left_shoulder → right_shoulder
    (5, 7),     # left_shoulder → left_elbow
    (6, 8),     # right_shoulder → right_elbow
    (7, 9),     # left_elbow → left_wrist
    (8, 10),    # right_elbow → right_wrist
    (1, 2),     # left_eye → right_eye
    (0, 1),     # nose → left_eye
    (0, 2),     # nose → right_eye
    (1, 3),     # left_eye → left_ear
    (2, 4),     # right_eye → right_ear
    (3, 5),     # left_ear → left_shoulder
    (4, 6),     # right_ear → right_shoulder
]

# Lateralidade COCO
COCO17_JOINTS_LEFT: list[int] = [1, 3, 5, 7, 9, 11, 13, 15]
"""left_eye, left_ear, left_shoulder, left_elbow, left_wrist, left_hip,
left_knee, left_ankle"""

COCO17_JOINTS_RIGHT: list[int] = [2, 4, 6, 8, 10, 12, 14, 16]
"""right_eye, right_ear, right_shoulder, right_elbow, right_wrist, right_hip,
right_knee, right_ankle"""

# Cores para visualização COCO (BGR)
COCO17_COLORS = {
    "left": (0, 255, 0),       # verde
    "right": (0, 128, 255),    # laranja
    "face": (255, 153, 51),    # azul — nariz/olhos/orelhas
}

# Keypoints COCO com informação de swap (para data augmentation com flip)
COCO17_SWAP_PAIRS: list[tuple[int, int]] = [
    (1, 2),    # left_eye ↔ right_eye
    (3, 4),    # left_ear ↔ right_ear
    (5, 6),    # left_shoulder ↔ right_shoulder
    (7, 8),    # left_elbow ↔ right_elbow
    (9, 10),   # left_wrist ↔ right_wrist
    (11, 12),  # left_hip ↔ right_hip
    (13, 14),  # left_knee ↔ right_knee
    (15, 16),  # left_ankle ↔ right_ankle
]

# Swap pairs para H3WB
H3WB_SWAP_PAIRS: list[tuple[int, int]] = [
    (1, 4),    # Left Hip ↔ Right Hip
    (2, 5),    # Left Knee ↔ Right Knee
    (3, 6),    # Left Ankle ↔ Right Ankle
    (14, 11),  # Left Shoulder ↔ Right Shoulder
    (15, 12),  # Left Elbow ↔ Right Elbow
    (16, 13),  # Left Wrist ↔ Right Wrist
]
