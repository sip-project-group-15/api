import numpy as np

from app import config, detector


def test_falls_back_to_mock_when_no_model(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "missing.onnx")
    detector.reset_detector()

    assert detector.get_detector().is_mock


def test_falls_back_to_mock_when_model_is_unloadable(tmp_path, monkeypatch):
    """A corrupt weights file must not take the service down.

    The file exists, so the presence check passes and loading is attempted --
    onnxruntime then rejects it. Serving mock alerts beats a boot loop.
    """
    broken = tmp_path / "best.onnx"
    broken.write_bytes(b"not an onnx graph")
    monkeypatch.setattr(config, "MODEL_PATH", broken)
    detector.reset_detector()

    assert detector.get_detector().is_mock


def test_detector_is_cached():
    detector.reset_detector()

    assert detector.get_detector() is detector.get_detector()


def test_mock_reproduces_the_original_schedule():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    mock = detector.MockDetector()

    def confidence(frame_number: int) -> float:
        return mock.predict(frame, frame_number)[0]["confidence"]

    assert confidence(1) == 0.85
    assert confidence(45) == 0.85
    assert confidence(20) == 0.65
    assert confidence(7) == 0.25


def test_mock_emits_a_threat_class():
    """Alerts only fire on threat classes, so the mock must produce one."""
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    detections = detector.MockDetector().predict(frame, 1)

    assert detections[0]["label"] in config.THREAT_CLASSES
