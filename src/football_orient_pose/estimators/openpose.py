"""Wrapper OpenPose para inferência de keypoints COCO-17."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from football_orient_pose.pose import BasePoseEstimator

OPENPOSE_PROTOTXT_URL = (
    "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/"
    "openpose/master/models/pose/coco/pose_deploy_linevec.prototxt"
)
OPENPOSE_CAFFEMODEL_URL = (
    "http://posefs1.perception.cs.cmu.edu/OpenPose/models/"
    "pose/coco/pose_iter_440000.caffemodel"
)

# Mapeamento OpenPose-18 → COCO-17.
# OPENPOSE_TO_COCO17[i] = índice OpenPose que corresponde ao i-ésimo joint COCO.
# O OpenPose index 1 (Neck) não existe no COCO-17 e é descartado.
#
# OpenPose-18:  0=Nose 1=Neck 2=RShldr 3=RElbow 4=RWrist 5=LShldr 6=LElbow
#               7=LWrist 8=RHip 9=RKnee 10=RAnkle 11=LHip 12=LKnee 13=LAnkle
#               14=REye 15=LEye 16=REar 17=LEar
# COCO-17:      0=Nose 1=LEye 2=REye 3=LEar 4=REar 5=LShldr 6=RShldr 7=LElbow
#               8=RElbow 9=LWrist 10=RWrist 11=LHip 12=RHip 13=LKnee 14=RKnee
#               15=LAnkle 16=RAnkle
OPENPOSE_TO_COCO17: list[int] = [
    0,   # COCO  0 Nose      ← OP  0
    15,  # COCO  1 LEye      ← OP 15
    14,  # COCO  2 REye      ← OP 14
    17,  # COCO  3 LEar      ← OP 17
    16,  # COCO  4 REar      ← OP 16
    5,   # COCO  5 LShoulder ← OP  5
    2,   # COCO  6 RShoulder ← OP  2
    6,   # COCO  7 LElbow    ← OP  6
    3,   # COCO  8 RElbow    ← OP  3
    7,   # COCO  9 LWrist    ← OP  7
    4,   # COCO 10 RWrist    ← OP  4
    11,  # COCO 11 LHip      ← OP 11
    8,   # COCO 12 RHip      ← OP  8
    12,  # COCO 13 LKnee     ← OP 12
    9,   # COCO 14 RKnee     ← OP  9
    13,  # COCO 15 LAnkle    ← OP 13
    10,  # COCO 16 RAnkle    ← OP 10
]

_CONFIDENCE_THRESHOLD = 0.1


class OpenPoseEstimator(BasePoseEstimator):
    """Estimador OpenPose com saída padronizada em COCO-17.

    Usa ``cv2.dnn`` para carregar o modelo Caffe do OpenPose COCO (18 partes).
    O OpenPose é bottom-up: detecta todos os keypoints da imagem de uma vez
    via Part Affinity Fields. Para crops 100×100, extrai o pico de cada
    heatmap e remapeia OpenPose-18 → COCO-17 descartando o joint Neck.

    Parameters
    ----------
    prototxt_path : str
        Caminho local para o arquivo ``.prototxt`` da rede.
    caffemodel_path : str
        Caminho local para os pesos ``.caffemodel``.
    input_size : tuple[int, int]
        Tamanho de entrada da rede em ``(width, height)``, padrão ``(368, 368)``.
    confidence_threshold : float
        Limiar mínimo de confiança; joints abaixo disso ficam em ``(0, 0, 0)``.
    device : str
        ``"cpu"`` ou ``"cuda"`` (requer OpenCV compilado com suporte CUDA).
    pose_model : Any | None
        Callable ``(image: np.ndarray) -> (keypoints (17,2), scores (17,))``
        injetado para testes. Quando informado, o modelo Caffe não é carregado.

    Notes
    -----
    Os pesos do modelo podem ser baixados em:
    ``https://github.com/CMU-Perceptual-Computing-Lab/openpose`` (ver README).
    """

    def __init__(
        self,
        prototxt_path: str = "",
        caffemodel_path: str = "",
        input_size: tuple[int, int] = (368, 368),
        confidence_threshold: float = _CONFIDENCE_THRESHOLD,
        device: str = "cpu",
        pose_model: Any | None = None,
    ) -> None:
        self.prototxt_path = prototxt_path
        self.caffemodel_path = caffemodel_path
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._pose_model = pose_model if pose_model is not None else self._load_pose_model()

    @property
    def name(self) -> str:
        """Nome identificável do estimador."""
        return "openpose"

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Retorna keypoints COCO-17 para uma imagem.

        Parameters
        ----------
        image : np.ndarray
            Crop do jogador em formato ``H × W × C`` BGR.

        Returns
        -------
        np.ndarray
            Array ``(17, 3)`` com ``x``, ``y`` e ``confidence``.
        """
        keypoints, scores = self._pose_model(image)
        return np.column_stack([
            np.asarray(keypoints, dtype=np.float32),
            np.asarray(scores, dtype=np.float32),
        ])

    def predict_batch(self, images: list[np.ndarray]) -> np.ndarray:
        """Retorna keypoints COCO-17 para um lote de imagens."""
        if not images:
            return np.empty((0, 17, 3), dtype=np.float32)
        return np.stack([self.predict(img) for img in images])

    def _load_pose_model(self) -> Any:
        if not self.prototxt_path or not self.caffemodel_path:
            raise ValueError(
                "OpenPoseEstimator: 'prototxt_path' e 'caffemodel_path' são "
                "obrigatórios. Baixe os arquivos do OpenPose COCO e forneça os "
                "caminhos, ou injete 'pose_model' no construtor para testes."
            )

        net = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.caffemodel_path)
        if self.device == "cuda":
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        input_w, input_h = self.input_size
        threshold = self.confidence_threshold

        def run(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            orig_h, orig_w = image.shape[:2]

            blob = cv2.dnn.blobFromImage(
                image, 1.0 / 255.0, (input_w, input_h), (0, 0, 0), swapRB=False
            )
            net.setInput(blob)
            output = net.forward()  # (1, 57, hm_h, hm_w) — 19 heatmaps + 38 PAFs

            hm_h, hm_w = output.shape[2], output.shape[3]

            # Extrair os 18 heatmaps de partes corporais (primeiros 18 canais)
            openpose_kps = np.zeros((17, 2), dtype=np.float32)
            openpose_scores = np.zeros(17, dtype=np.float32)

            for coco_idx, op_idx in enumerate(OPENPOSE_TO_COCO17):
                heatmap = output[0, op_idx]
                _, conf, _, peak_loc = cv2.minMaxLoc(heatmap)
                if conf >= threshold:
                    x_img = peak_loc[0] * orig_w / hm_w
                    y_img = peak_loc[1] * orig_h / hm_h
                    openpose_kps[coco_idx] = [x_img, y_img]
                    openpose_scores[coco_idx] = conf

            return openpose_kps, openpose_scores

        return run
