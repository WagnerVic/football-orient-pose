"""Detecção de pessoas: ABC Detector + wrappers YOLO26, RetinaNet, Faster R-CNN, Cascade R-CNN."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

Detection = tuple[np.ndarray, float]  # (xyxy float32 shape (4,), conf)

COCO_PERSON_LABEL = 1  # torchvision/COCO: label==1 é person


def _auto_device(device: str | None) -> str:
    """Resolve o device: respeita o explícito; senão usa CUDA quando disponível, senão CPU.

    Evita o crash de instanciar um detector com ``device="cuda"`` numa máquina sem GPU.
    """
    if device is not None:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def detections_to_arrays(detections: list[Detection]) -> tuple[np.ndarray, np.ndarray]:
    """Converte ``list[(xyxy, conf)]`` em ``(boxes (N,4) float32, scores (N,) float32)``.

    Ponte entre a saída dos detectores e o GT ``(N,4)`` do ``roboflow_io`` — o benchmark
    (Épico #113) compara os dois em forma de array, sem repetir ``zip``/``stack``.
    """
    if not detections:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
    boxes = np.stack([b for b, _ in detections]).astype(np.float32)
    scores = np.asarray([s for _, s in detections], dtype=np.float32)
    return boxes, scores


class Detector(ABC):
    """Interface comum para detectores de pessoas em frames de vídeo.

    Detectores concretos devem implementar `detect` e expor `name`.
    O contrato espelha `BasePoseEstimator` em `pose.py`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Nome identificável do detector."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detecta pessoas num frame.

        Parameters
        ----------
        image : np.ndarray
            Frame BGR (OpenCV), shape ``H x W x 3``.

        Returns
        -------
        list[Detection]
            Lista de ``(xyxy, conf)`` onde ``xyxy`` é ``float32 (4,)`` em
            coordenadas do frame e ``conf`` é a confiança da detecção.
            Somente caixas de pessoa são retornadas.
        """


class _TorchvisionDetector(Detector):
    """Base reutilizável para detectores torchvision pré-treinados COCO."""

    def __init__(
        self, build_fn, weights, conf_threshold: float = 0.3, device: str | None = None
    ) -> None:
        import torch

        self._torch = torch
        self._device = _auto_device(device)
        self._model = build_fn(weights=weights).eval().to(self._device)
        self._tf = weights.transforms()
        self.conf_threshold = conf_threshold

    def detect(self, image: np.ndarray) -> list[Detection]:
        torch = self._torch
        # BGR (OpenCV) → RGB tensor
        rgb = torch.from_numpy(image[..., ::-1].copy()).permute(2, 0, 1)
        x = self._tf(rgb).to(self._device)
        with torch.inference_mode():
            out = self._model([x])[0]
        keep = (out["labels"] == COCO_PERSON_LABEL) & (out["scores"] >= self.conf_threshold)
        boxes = out["boxes"][keep].cpu().numpy().astype("float32")
        scores = out["scores"][keep].cpu().numpy()
        return [(b, float(s)) for b, s in zip(boxes, scores)]


class FasterRCNNDetector(_TorchvisionDetector):
    """Faster R-CNN ResNet50-FPN pré-treinado COCO (torchvision)."""

    name = "faster-rcnn"

    def __init__(self, conf_threshold: float = 0.3, device: str | None = None) -> None:
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_Weights,
            fasterrcnn_resnet50_fpn,
        )

        super().__init__(
            fasterrcnn_resnet50_fpn,
            FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
            conf_threshold,
            device,
        )


class YOLO26Detector(Detector):
    """YOLO26 (ultralytics) pré-treinado COCO.

    Ultralytics usa ``cls==0`` para person (diferente do torchvision).
    """

    name = "yolo26"

    def __init__(
        self,
        weights: str = "yolo26n.pt",
        conf_threshold: float = 0.3,
        device: str | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "YOLO26Detector requer 'ultralytics' (pip install ultralytics)"
            ) from exc
        self._model = YOLO(weights)
        self.conf_threshold = conf_threshold
        self._device = _auto_device(device)

    def detect(self, image: np.ndarray) -> list[Detection]:
        res = self._model.predict(
            image,
            classes=[0],
            conf=self.conf_threshold,
            device=self._device,
            verbose=False,
        )[0]
        return [
            (box.xyxy[0].cpu().numpy().astype("float32"), float(box.conf[0]))
            for box in res.boxes
        ]


class RetinaNetDetector(_TorchvisionDetector):
    """RetinaNet ResNet50-FPN pré-treinado COCO (torchvision)."""

    name = "retinanet"

    def __init__(self, conf_threshold: float = 0.3, device: str | None = None) -> None:
        from torchvision.models.detection import (
            RetinaNet_ResNet50_FPN_Weights,
            retinanet_resnet50_fpn,
        )

        super().__init__(
            retinanet_resnet50_fpn,
            RetinaNet_ResNet50_FPN_Weights.COCO_V1,
            conf_threshold,
            device,
        )


class CascadeRCNNDetector(Detector):
    """Cascade R-CNN pré-treinado COCO via mmdet.

    Requer config e checkpoint do model zoo do mmdet.
    Person = ``labels==0`` no mmdet (diferente do torchvision que usa 1).
    """

    name = "cascade-rcnn"

    def __init__(
        self,
        config: str,
        checkpoint: str,
        conf_threshold: float = 0.3,
        device: str | None = None,
    ) -> None:
        try:
            from mmdet.apis import init_detector
        except ImportError as exc:
            raise ImportError(
                "CascadeRCNNDetector requer mmdet (mim install mmdet)"
            ) from exc
        self._device = _auto_device(device)
        self._model = init_detector(config, checkpoint, device=self._device)
        self.conf_threshold = conf_threshold

    def detect(self, image: np.ndarray) -> list[Detection]:
        from mmdet.apis import inference_detector

        res = inference_detector(self._model, image)
        inst = res.pred_instances
        m = (inst.labels == 0) & (inst.scores >= self.conf_threshold)
        boxes = inst.bboxes[m].cpu().numpy().astype("float32")
        scores = inst.scores[m].cpu().numpy()
        return [(b, float(s)) for b, s in zip(boxes, scores)]
