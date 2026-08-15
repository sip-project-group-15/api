"""Frame-level detection, with a mock fallback.

Selection is by file presence: if config.MODEL_PATH exists and loads, the ONNX
model is used; otherwise the mock keeps the API fully functional. This lets the
service deploy and be integrated against before any weights are trained.

Serving ONNX rather than PyTorch is deliberate — the deployment box has 6 shared
cores and no GPU, where onnxruntime keeps the image ~250MB instead of ~2GB.
"""

import ast
import logging
from typing import Protocol

import numpy as np

from app import config

logger = logging.getLogger(__name__)


class Detector(Protocol):
    """Common surface so callers never branch on which backend is active."""

    name: str
    is_mock: bool

    def predict(self, frame: np.ndarray, frame_number: int) -> list[dict]: ...


class MockDetector:
    """Deterministic stand-in used until real weights are available.

    Reproduces the original mock_predict schedule so existing behaviour and
    demos are unchanged, but emits detection dicts like the real detector.
    """

    name = "mock"
    is_mock = True

    def predict(self, frame: np.ndarray, frame_number: int) -> list[dict]:
        if frame_number == 1 or frame_number % 45 == 0:
            confidence = 0.85
        elif frame_number % 20 == 0:
            confidence = 0.65
        else:
            confidence = 0.25

        height, width = frame.shape[:2]
        return [
            {
                "label": "person",
                "confidence": round(confidence, 4),
                # Centre box, sized off the frame so it is plausible to draw.
                "box": [width // 4, height // 4, width // 2, height // 2],
            }
        ]


class YoloOnnxDetector:
    """YOLO detection head served through onnxruntime."""

    name = "yolo-onnx"
    is_mock = False

    def __init__(self, model_path, image_size: int):
        import onnxruntime as ort

        self.image_size = image_size
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.class_names = self._read_class_names()

    def _read_class_names(self) -> list[str]:
        """Prefer names baked into the graph over the config fallback.

        Ultralytics writes them during export, so the served labels always match
        the weights even if config.CLASS_NAMES drifts.
        """
        metadata = self.session.get_modelmeta().custom_metadata_map or {}
        raw = metadata.get("names")
        if raw:
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, dict):
                    return [parsed[key] for key in sorted(parsed)]
            except (ValueError, SyntaxError, KeyError):
                logger.warning("Could not parse class names from model metadata")
        return list(config.CLASS_NAMES)

    def _letterbox(self, frame: np.ndarray):
        """Resize preserving aspect ratio, padding to a square input.

        Plain resize would distort the image and shift box coordinates, so the
        scale and padding are returned to undo the transform afterwards.
        """
        import cv2

        height, width = frame.shape[:2]
        scale = min(self.image_size / height, self.image_size / width)
        resized_w, resized_h = int(round(width * scale)), int(round(height * scale))

        resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

        pad_x = (self.image_size - resized_w) // 2
        pad_y = (self.image_size - resized_h) // 2
        canvas = np.full((self.image_size, self.image_size, 3), 114, dtype=np.uint8)
        canvas[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized

        return canvas, scale, pad_x, pad_y

    def predict(self, frame: np.ndarray, frame_number: int) -> list[dict]:
        import cv2

        canvas, scale, pad_x, pad_y = self._letterbox(frame)

        # BGR->RGB, HWC->CHW, 0-1 normalised, batched.
        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)

        outputs = self.session.run(None, {self.input_name: blob})[0]

        # (1, 4 + num_classes, num_boxes) -> (num_boxes, 4 + num_classes)
        predictions = np.squeeze(outputs, axis=0).T
        if predictions.shape[1] < 5:
            return []

        class_scores = predictions[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        confidences = class_scores.max(axis=1)

        keep = confidences >= config.MODEL_CONF_THRESHOLD
        if not keep.any():
            return []

        boxes_xywh = predictions[keep, :4]
        class_ids = class_ids[keep]
        confidences = confidences[keep]

        # Model emits centre-x/centre-y/w/h in letterboxed space; convert to
        # top-left xywh in original-frame coordinates.
        centre_x, centre_y = boxes_xywh[:, 0], boxes_xywh[:, 1]
        box_w, box_h = boxes_xywh[:, 2], boxes_xywh[:, 3]
        left = (centre_x - box_w / 2 - pad_x) / scale
        top = (centre_y - box_h / 2 - pad_y) / scale
        width = box_w / scale
        height = box_h / scale

        boxes = np.stack([left, top, width, height], axis=1)

        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            confidences.tolist(),
            config.MODEL_CONF_THRESHOLD,
            config.MODEL_IOU_THRESHOLD,
        )
        if len(indices) == 0:
            return []

        frame_h, frame_w = frame.shape[:2]
        detections = []
        for index in np.array(indices).flatten():
            x, y, w, h = boxes[index]
            class_id = int(class_ids[index])
            label = (
                self.class_names[class_id]
                if class_id < len(self.class_names)
                else str(class_id)
            )
            detections.append(
                {
                    "label": label,
                    "confidence": round(float(confidences[index]), 4),
                    "box": [
                        max(0, int(x)),
                        max(0, int(y)),
                        min(frame_w, int(w)),
                        min(frame_h, int(h)),
                    ],
                }
            )
        return detections


_detector: Detector | None = None


def load_detector() -> Detector:
    """Pick a backend based on whether weights are present and loadable.

    A model that exists but fails to load falls back to the mock rather than
    taking the service down — an unservable file should not turn a deploy into
    an outage.
    """
    if not config.MODEL_PATH.is_file():
        logger.warning(
            "No model at %s — using mock detector. Alerts are simulated.",
            config.MODEL_PATH,
        )
        return MockDetector()

    try:
        detector = YoloOnnxDetector(config.MODEL_PATH, config.MODEL_IMAGE_SIZE)
    except Exception:
        logger.exception("Failed to load %s — falling back to mock", config.MODEL_PATH)
        return MockDetector()

    logger.info(
        "Loaded model %s with classes %s", config.MODEL_PATH, detector.class_names
    )
    return detector


def get_detector() -> Detector:
    """Process-wide singleton; the ONNX session is expensive to build."""
    global _detector
    if _detector is None:
        _detector = load_detector()
    return _detector


def reset_detector() -> None:
    """Drop the cached detector so the next call re-resolves. Used by tests."""
    global _detector
    _detector = None
