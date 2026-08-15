import numpy as np
import pytest

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


# ── Output layout ────────────────────────────────────────────────────────────
# Two ONNX layouts reach the detector and they are not distinguishable by
# anything except shape. Picking wrong does not raise, it silently produces
# nonsense boxes, so these guard the discriminator itself.


def test_end_to_end_layout_is_recognised():
    """YOLO26-style: (detections, 6) with NMS already applied in-graph."""
    assert detector.is_end_to_end(np.zeros((300, 6), np.float32), 80)


def test_classic_layout_is_recognised():
    """YOLOv8/11-style: (4 + num_classes, boxes), still needs NMS here."""
    assert not detector.is_end_to_end(np.zeros((84, 8400), np.float32), 80)
    assert not detector.is_end_to_end(np.zeros((7, 8400), np.float32), 3)


def test_a_two_class_head_is_not_mistaken_for_end_to_end():
    """4 + 2 classes = 6 rows, which is the one genuinely ambiguous width."""
    assert not detector.is_end_to_end(np.zeros((6, 8400), np.float32), 2)


def test_parse_end_to_end_drops_the_padding_rows():
    """The graph emits a fixed count, tail-padded with zero-score rows."""
    rows = np.zeros((300, 6), np.float32)
    rows[0] = [10, 20, 50, 80, 0.9, 5]
    rows[1] = [30, 40, 70, 90, 0.8, 0]

    boxes, confidences, class_ids = detector.parse_end_to_end(rows)

    assert len(boxes) == 2
    assert boxes[0].tolist() == [10, 20, 50, 80]
    assert confidences.tolist() == pytest.approx([0.9, 0.8])
    assert class_ids.tolist() == [5, 0]


def test_parse_end_to_end_applies_the_confidence_floor(monkeypatch):
    monkeypatch.setattr(config, "MODEL_CONF_THRESHOLD", 0.5)
    rows = np.zeros((10, 6), np.float32)
    rows[0] = [10, 20, 50, 80, 0.9, 1]
    rows[1] = [10, 20, 50, 80, 0.4, 1]

    boxes, _, _ = detector.parse_end_to_end(rows)

    assert len(boxes) == 1


def test_parse_raw_suppresses_duplicate_boxes():
    """Three near-identical boxes for one object must collapse to one."""
    predictions = np.zeros((6, 3), np.float32)  # 4 coords + 2 classes
    for index, confidence in enumerate((0.9, 0.85, 0.8)):
        predictions[:4, index] = [100 + index, 100, 40, 40]  # centre xywh
        predictions[4, index] = confidence

    boxes, confidences, _ = detector.parse_raw(predictions)

    assert len(boxes) == 1
    assert confidences[0] == pytest.approx(0.9)


def test_parse_raw_returns_corners():
    predictions = np.zeros((6, 1), np.float32)
    predictions[:4, 0] = [100, 100, 40, 60]  # centre x, centre y, w, h
    predictions[4, 0] = 0.9

    boxes, _, _ = detector.parse_raw(predictions)

    assert boxes[0].tolist() == [80, 70, 120, 130]


def test_to_detections_undoes_the_letterbox():
    boxes = np.array([[30.0, 40.0, 130.0, 240.0]])

    detections = detector.to_detections(
        boxes, np.array([0.9]), np.array([1]), ["rhino", "person"],
        scale=0.5, pad_x=10, pad_y=20, frame_shape=(1000, 1000, 3),
    )

    # (30 - 10) / 0.5 = 40, (40 - 20) / 0.5 = 40, then 200x400 at that scale.
    assert detections[0]["box"] == [40, 40, 200, 400]
    assert detections[0]["label"] == "person"


def test_to_detections_clamps_to_the_frame():
    """A box running off-frame would read as standing closer to the camera."""
    boxes = np.array([[-50.0, -50.0, 500.0, 500.0]])

    detections = detector.to_detections(
        boxes, np.array([0.9]), np.array([0]), ["rhino"],
        scale=1.0, pad_x=0, pad_y=0, frame_shape=(200, 300, 3),
    )

    x, y, w, h = detections[0]["box"]
    assert (x, y) == (0, 0)
    assert x + w <= 300 and y + h <= 200


def test_to_detections_applies_label_aliases(monkeypatch):
    """Lets stock COCO weights stand in for the trained model in a smoke test."""
    monkeypatch.setattr(config, "LABEL_ALIASES", {"truck": "vehicle"})
    boxes = np.array([[0.0, 0.0, 10.0, 10.0]])

    detections = detector.to_detections(
        boxes, np.array([0.9]), np.array([0]), ["truck"],
        scale=1.0, pad_x=0, pad_y=0, frame_shape=(100, 100, 3),
    )

    assert detections[0]["label"] == "vehicle"
