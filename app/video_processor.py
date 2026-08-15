from pathlib import Path

from app import config
from app.detector import get_detector


def frame_stride(fps: float) -> int:
    """How many frames to skip between inferences.

    Sampling is what keeps analysis inside the proxy's timeout: at 30fps a
    5-minute clip is 9,000 frames, but ~2 inferences per second covers the same
    footage in a few hundred.
    """
    if config.FRAME_SAMPLE_FPS <= 0:
        return 1
    return max(1, round(fps / config.FRAME_SAMPLE_FPS))


def assess(detections: list[dict]) -> tuple[float, str | None]:
    """Turn per-frame detections into an alert probability and severity.

    The rule lives here rather than in the model so it stays explainable: a
    ranger can be told "a person was seen near a rhino", which is more
    actionable than a bare confidence score.
    """
    threats = [d for d in detections if d["label"] in config.THREAT_CLASSES]
    if not threats:
        return 0.0, None

    probability = max(d["confidence"] for d in threats)
    assets_present = any(d["label"] in config.ASSET_CLASSES for d in detections)
    return probability, "high" if assets_present else "medium"


def process_video(video_path: Path, threshold: float, alert_frames_folder: Path) -> dict:
    import cv2

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError("Video could not be opened")

    detector = get_detector()
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    stride = frame_stride(fps)

    processed_frames = 0
    analyzed_frames = 0
    alerts = []

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            processed_frames += 1

            # Frame 1 is always analysed, then every stride-th frame after it.
            if (processed_frames - 1) % stride:
                continue

            analyzed_frames += 1
            detections = detector.predict(frame, processed_frames)
            probability, severity = assess(detections)

            if probability >= threshold and severity is not None:
                timestamp_seconds = round((processed_frames - 1) / fps, 2)
                snapshot_path = alert_frames_folder / f"frame_{processed_frames:06d}.jpg"
                cv2.imwrite(str(snapshot_path), frame)
                alerts.append(
                    {
                        "frame_number": processed_frames,
                        "timestamp_seconds": timestamp_seconds,
                        "probability": probability,
                        "label": "possible_poaching",
                        "severity": severity,
                        "detections": detections,
                        "location": None,
                        "sms_sent": False,
                        "snapshot_path": str(snapshot_path),
                        "message": "Possible poaching activity detected",
                    }
                )
    finally:
        capture.release()

    return {
        "processed_frames": processed_frames,
        "analyzed_frames": analyzed_frames,
        "frame_stride": stride,
        "detector": detector.name,
        "alerts": alerts,
    }
