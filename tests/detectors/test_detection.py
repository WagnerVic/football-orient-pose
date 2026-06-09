from __future__ import annotations

import numpy as np
import pytest

from football_orient_pose.detection import (
    CascadeRCNNDetector,
    Detection,
    Detector,
    FasterRCNNDetector,
    RetinaNetDetector,
    YOLO26Detector,
)


class DummyDetector(Detector):
    @property
    def name(self) -> str:
        return "dummy"

    def detect(self, image: np.ndarray) -> list[Detection]:
        box1 = np.array([10.0, 20.0, 100.0, 200.0], dtype=np.float32)
        box2 = np.array([50.0, 60.0, 150.0, 300.0], dtype=np.float32)
        return [(box1, 0.9), (box2, 0.75)]


def test_detector_abc_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        Detector()


def test_dummy_detector_name() -> None:
    assert DummyDetector().name == "dummy"


def test_dummy_detector_returns_list_of_detections() -> None:
    detector = DummyDetector()
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = detector.detect(image)

    assert isinstance(detections, list)
    assert len(detections) == 2
    for box, conf in detections:
        assert isinstance(box, np.ndarray)
        assert box.shape == (4,)
        assert box.dtype == np.float32
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0


def test_dummy_detector_returns_correct_values() -> None:
    detector = DummyDetector()
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = detector.detect(image)

    box, conf = detections[0]
    np.testing.assert_array_equal(box, np.array([10.0, 20.0, 100.0, 200.0], dtype=np.float32))
    assert conf == pytest.approx(0.9)


def test_all_concrete_detectors_subclass_detector() -> None:
    for cls in (YOLO26Detector, RetinaNetDetector, FasterRCNNDetector, CascadeRCNNDetector):
        assert issubclass(cls, Detector)


def test_torchvision_detector_names() -> None:
    assert FasterRCNNDetector.name == "faster-rcnn"
    assert RetinaNetDetector.name == "retinanet"


def test_cascade_rcnn_requires_mmdet(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name.startswith("mmdet"):
            raise ImportError("mmdet not available")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(ImportError, match="mmdet"):
        CascadeRCNNDetector(config="cfg.py", checkpoint="ckpt.pth")


@pytest.mark.integration
def test_faster_rcnn_detects_on_cpu() -> None:
    detector = FasterRCNNDetector(conf_threshold=0.3, device="cpu")
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = detector.detect(image)

    assert isinstance(detections, list)
    for box, conf in detections:
        assert box.shape == (4,)
        assert box.dtype == np.float32
        assert 0.0 <= conf <= 1.0


@pytest.mark.integration
def test_retinanet_detects_on_cpu() -> None:
    detector = RetinaNetDetector(conf_threshold=0.3, device="cpu")
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = detector.detect(image)

    assert isinstance(detections, list)
    for box, conf in detections:
        assert box.shape == (4,)
        assert box.dtype == np.float32
        assert 0.0 <= conf <= 1.0
