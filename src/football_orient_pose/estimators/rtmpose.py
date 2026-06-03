"""Wrappers RTMPose para inferência de keypoints.

RTMPoseEstimator  — zero-shot via rtmlib/ONNX, saída COCO-17.
RTMPoseFinetunedEstimator — modelo fine-tunado via MMPose, saída H3WB-17.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from football_orient_pose.pose import BasePoseEstimator

RTMPOSE_X_ONNX_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip"
)


class RTMPoseEstimator(BasePoseEstimator):
    """Estimador RTMPose-X com saída padronizada em COCO-17.

    Este wrapper usa a API ``rtmlib.RTMPose`` diretamente sobre o crop do
    jogador. Para evitar depender de um detector em crops 100x100 do 3DSP, a
    inferência é feita com a bounding box padrão da própria rtmlib, que cobre a
    imagem inteira.

    Parameters
    ----------
    model_path : str
        Caminho local ou URL para o modelo ONNX/ZIP do RTMPose.
    model_input_size : tuple[int, int]
        Tamanho de entrada esperado pela rtmlib em ordem ``(width, height)``.
    backend : str
        Backend de inferência da rtmlib, como ``"onnxruntime"`` ou ``"opencv"``.
    device : str
        Dispositivo de inferência, como ``"cpu"`` ou ``"cuda"``.
    pose_model : Any | None
        Modelo já instanciado, útil para testes ou para injetar uma configuração
        externa. Quando informado, ``rtmlib`` não é importado.

    Notes
    -----
    Instale a dependência opcional antes de usar o estimador real:
    ``pip install rtmlib onnxruntime``.
    """

    def __init__(
        self,
        model_path: str = RTMPOSE_X_ONNX_URL,
        model_input_size: tuple[int, int] = (288, 384),
        backend: str = "onnxruntime",
        device: str = "cpu",
        pose_model: Any | None = None,
    ) -> None:
        self.model_path = model_path
        self.model_input_size = model_input_size
        self.backend = backend
        self.device = device
        self._pose_model = pose_model if pose_model is not None else self._load_pose_model()

    @property
    def name(self) -> str:
        """Nome identificável do estimador."""
        return "rtmpose-x"

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Retorna keypoints COCO-17 para uma imagem.

        Parameters
        ----------
        image : np.ndarray
            Crop do jogador em formato ``H x W x C``.

        Returns
        -------
        np.ndarray
            Array ``(17, 3)`` com ``x``, ``y`` e ``confidence``.
        """
        keypoints, scores = self._pose_model(image)
        keypoints = self._ensure_instance_axis(np.asarray(keypoints, dtype=np.float32))
        scores = self._ensure_score_axis(np.asarray(scores, dtype=np.float32))

        instance_index = self._select_best_instance(keypoints, image.shape[:2])
        coco_keypoints = np.column_stack((keypoints[instance_index], scores[instance_index]))
        return coco_keypoints.astype(np.float32)

    def predict_batch(self, images: list[np.ndarray]) -> np.ndarray:
        """Retorna keypoints COCO-17 para um lote de imagens."""
        if not images:
            return np.empty((0, 17, 3), dtype=np.float32)
        return np.stack([self.predict(image) for image in images])

    def _load_pose_model(self) -> Any:
        try:
            from rtmlib import RTMPose
        except ImportError as exc:
            raise ImportError(
                "RTMPoseEstimator requer a dependência opcional 'rtmlib'. "
                "Instale com 'pip install rtmlib onnxruntime' ou injete "
                "'pose_model' no construtor."
            ) from exc

        return RTMPose(
            self.model_path,
            model_input_size=self.model_input_size,
            backend=self.backend,
            device=self.device,
        )

    @staticmethod
    def _ensure_instance_axis(keypoints: np.ndarray) -> np.ndarray:
        if keypoints.shape == (17, 2):
            return keypoints[np.newaxis, ...]
        if keypoints.ndim == 3 and keypoints.shape[1:] == (17, 2):
            return keypoints
        raise ValueError(f"RTMPose retornou keypoints com shape inesperado: {keypoints.shape}")

    @staticmethod
    def _ensure_score_axis(scores: np.ndarray) -> np.ndarray:
        if scores.shape == (17,):
            return scores[np.newaxis, ...]
        if scores.ndim == 2 and scores.shape[1] == 17:
            return scores
        raise ValueError(f"RTMPose retornou scores com shape inesperado: {scores.shape}")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config_path: str | None = None,
        device: str = "cuda",
    ) -> "RTMPoseFinetunedEstimator":
        """Carrega modelo fine-tunado via MMPose.

        Retorna um ``RTMPoseFinetunedEstimator`` com a mesma interface de
        ``predict()`` / ``predict_h3wb()``, mas com saída direta em H3WB-17
        (sem conversão COCO→H3WB).

        Parameters
        ----------
        checkpoint_path : str
            Path para o .pth gerado pelo train.py.
        config_path : str | None
            Path para o config MMPose. Se None, infere pelo nome do checkpoint.
        device : str
            ``"cuda"`` ou ``"cpu"``.
        """
        if config_path is None:
            config_path = _infer_config_from_checkpoint(checkpoint_path)
        return RTMPoseFinetunedEstimator(checkpoint_path, config_path, device)

    @staticmethod
    def _select_best_instance(keypoints: np.ndarray, image_shape: tuple[int, int]) -> int:
        if keypoints.shape[0] == 1:
            return 0

        image_center = np.array([image_shape[1] / 2, image_shape[0] / 2], dtype=np.float32)
        center_hips = (keypoints[:, 11] + keypoints[:, 12]) / 2
        center_shoulders = (keypoints[:, 5] + keypoints[:, 6]) / 2
        center_body = (center_hips + center_shoulders) / 2
        distances = np.linalg.norm(center_body - image_center, axis=1)
        return int(np.argmin(distances))


def _infer_config_from_checkpoint(checkpoint_path: str) -> str:
    p = Path(checkpoint_path)
    for part in p.parts:
        for cenario in ("a", "b", "c", "d"):
            if f"cenario_{cenario}" in part.lower():
                candidate = Path("configs") / f"cenario_{cenario}.py"
                if candidate.exists():
                    return str(candidate)
    raise ValueError(
        f"Não foi possível inferir o config a partir de '{checkpoint_path}'. "
        "Passe config_path explicitamente."
    )


class RTMPoseFinetunedEstimator:
    """RTMPose-X fine-tunado via MMPose.

    Saída em H3WB-17 — não realiza conversão COCO→H3WB, ao contrário do
    ``RTMPoseEstimator`` zero-shot. Compatível com as funções de métricas do
    projeto (que esperam coordenadas H3WB).

    Parameters
    ----------
    checkpoint_path : str
        Path para o .pth gerado pelo train.py.
    config_path : str
        Path para o config MMPose do cenário treinado.
    device : str
        ``"cuda"`` ou ``"cpu"``.
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        device: str = "cuda",
    ) -> None:
        try:
            from mmpose.apis import init_model
        except ImportError as exc:
            raise ImportError(
                "RTMPoseFinetunedEstimator requer MMPose. "
                "Instale com 'mim install mmpose' ou use o Docker de fine-tuning."
            ) from exc

        self._model = init_model(config_path, checkpoint_path, device=device)
        self._model.eval()
        self._checkpoint_path = checkpoint_path

    @property
    def name(self) -> str:
        return f"rtmpose-x-finetuned({Path(self._checkpoint_path).parent.name})"

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Retorna keypoints H3WB-17 para uma imagem.

        Parameters
        ----------
        image : np.ndarray
            Crop do jogador em formato H×W×C (BGR, 100×100).

        Returns
        -------
        np.ndarray
            Array ``(17, 3)`` com x, y (coordenadas do crop) e confidence.
        """
        from mmpose.apis import inference_topdown

        H, W = image.shape[:2]
        results = inference_topdown(
            self._model,
            image,
            bboxes=np.array([[0.0, 0.0, float(W), float(H)]]),
            bbox_format="xyxy",
        )
        if not results or len(results[0].pred_instances.keypoints) == 0:
            return np.zeros((17, 3), dtype=np.float32)

        kps = np.asarray(results[0].pred_instances.keypoints[0], dtype=np.float32)      # (17, 2)
        scores = np.asarray(results[0].pred_instances.keypoint_scores[0], dtype=np.float32)  # (17,)
        return np.column_stack([kps, scores]).astype(np.float32)  # (17, 3)

    def predict_h3wb(self, image: np.ndarray) -> np.ndarray:
        """Retorna keypoints H3WB-17 sem a coluna de confiança.

        Returns
        -------
        np.ndarray
            Array ``(17, 2)`` com x, y no espaço do crop.
        """
        return self.predict(image)[:, :2]

    def predict_batch(self, images: list[np.ndarray]) -> np.ndarray:
        if not images:
            return np.empty((0, 17, 3), dtype=np.float32)
        return np.stack([self.predict(img) for img in images])
