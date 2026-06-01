"""Wrapper HRNet-W48 para inferência de keypoints COCO-17."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from football_orient_pose.pose import BasePoseEstimator

# Caminho padrão após rodar: uv run python scripts/export_hrnet_onnx.py
HRNET_W48_ONNX_URL = "models/weights/hrnet_w48_coco_256x192.onnx"

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class HRNetEstimator(BasePoseEstimator):
    """Estimador HRNet-W48 com saída padronizada em COCO-17.

    O HRNet gera heatmaps ``(17, 64, 48)`` para localizar keypoints.
    Em crops 100×100 do 3DSP, a quantização heatmap→coordenada introduz
    erro de ~2 px por joint, resultando em PDJ esperado de ~56% (Yeung et al., 2024).

    Parameters
    ----------
    model_path : str
        Caminho local ou URL para o modelo ONNX do HRNet-W48 (256×192).
    model_input_size : tuple[int, int]
        Tamanho de entrada em ordem ``(width, height)``, padrão ``(192, 256)``.
    device : str
        Dispositivo de inferência, ``"cpu"`` ou ``"cuda"``.
    pose_model : Any | None
        Callable ``(image: np.ndarray) -> (keypoints (17,2), scores (17,))``
        injetado para testes. Quando informado, o ONNX não é carregado.

    Notes
    -----
    Instale a dependência opcional antes de usar o estimador real:
    ``pip install onnxruntime``  (ou ``onnxruntime-gpu`` para CUDA).
    """

    def __init__(
        self,
        model_path: str = HRNET_W48_ONNX_URL,
        model_input_size: tuple[int, int] = (192, 256),
        device: str = "cpu",
        pose_model: Any | None = None,
    ) -> None:
        self.model_path = model_path
        self.model_input_size = model_input_size
        self.device = device
        self._pose_model = pose_model if pose_model is not None else self._load_pose_model()

    @property
    def name(self) -> str:
        """Nome identificável do estimador."""
        return "hrnet-w48"

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
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "HRNetEstimator requer a dependência opcional 'onnxruntime'. "
                "Instale com 'pip install onnxruntime' ou injete "
                "'pose_model' no construtor."
            ) from exc

        if not self.model_path:
            raise ValueError(
                "HRNetEstimator: 'model_path' não foi configurado. "
                "Forneça o caminho para o arquivo ONNX exportado do HRNet-W48."
            )
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"HRNetEstimator: arquivo ONNX não encontrado em '{self.model_path}'. "
                "Execute 'bash scripts/download_models.sh' para baixar os pesos."
            )

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device == "cuda"
            else ["CPUExecutionProvider"]
        )
        session = ort.InferenceSession(self.model_path, providers=providers)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        input_w, input_h = self.model_input_size

        def run(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            orig_h, orig_w = image.shape[:2]

            resized = cv2.resize(image, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            normalized = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
            blob = normalized.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, H, W)

            heatmaps = session.run([output_name], {input_name: blob})[0][0]  # (17, 64, 48)

            hm_h, hm_w = heatmaps.shape[1], heatmaps.shape[2]
            flat = heatmaps.reshape(17, -1)
            flat_idx = flat.argmax(axis=1)
            confidence = flat.max(axis=1)

            y_hm = (flat_idx // hm_w).astype(np.float32)
            x_hm = (flat_idx % hm_w).astype(np.float32)

            x_img = x_hm * (orig_w / hm_w)
            y_img = y_hm * (orig_h / hm_h)

            return np.column_stack([x_img, y_img]).astype(np.float32), confidence.astype(np.float32)

        return run
