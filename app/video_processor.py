from pathlib import Path

from app import config
from app.detector import get_detector
from app.threat import ThreatMonitor


def frame_stride(fps: float) -> int:
    """How many frames to skip between inferences.

    Sampling is what keeps analysis inside the proxy's timeout: at 30fps a
    5-minute clip is 9,000 frames, but ~2 inferences per second covers the same
    footage in a few hundred.
    """
    if config.FRAME_SAMPLE_FPS <= 0:
        return 1
    return max(1, round(fps / config.FRAME_SAMPLE_FPS))


def process_video(video_path: Path, threshold: float, alert_frames_folder: Path) -> dict:
    import cv2

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError("Video could not be opened")

    detector = get_detector()
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    stride = frame_stride(fps)

    # Per-video, because track ids and distance histories from one clip mean
    # nothing in the next.
    monitor = ThreatMonitor()

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
            timestamp_seconds = round((processed_frames - 1) / fps, 2)
            detections = detector.predict(frame, processed_frames)

            # The monitor is fed every analysed frame, not only alerting ones:
            # it is building the tracks and distance histories that make
            # "approaching" measurable, and a gap in that record is a lost
            # trend.
            assessment = monitor.update(detections, timestamp_seconds)

            if assessment is None or assessment["score"] < threshold:
                continue

            snapshot_path = alert_frames_folder / f"frame_{processed_frames:06d}.jpg"
            cv2.imwrite(str(snapshot_path), frame)
            alerts.append(
                {
                    "frame_number": processed_frames,
                    "timestamp_seconds": timestamp_seconds,
                    # Kept under the old key so existing consumers and the SMS
                    # template keep working, but this is now the composite
                    # threat score, not a bare detection confidence — that is
                    # reported separately as detection_confidence.
                    "probability": assessment["score"],
                    "label": "possible_poaching",
                    "detections": detections,
                    "location": None,
                    "sms_sent": False,
                    "snapshot_path": str(snapshot_path),
                    "message": assessment["reason"],
                    **{
                        key: value
                        for key, value in assessment.items()
                        if key not in {"score", "label", "reason"}
                    },
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
