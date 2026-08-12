from pathlib import Path


def mock_predict(frame, frame_number: int) -> float:
    if frame_number == 1 or frame_number % 45 == 0:
        return 0.85

    if frame_number % 20 == 0:
        return 0.65

    return 0.25


def process_video(video_path: Path, threshold: float, alert_frames_folder: Path) -> dict:
    import cv2

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError("Video could not be opened")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    processed_frames = 0
    alerts = []

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            processed_frames += 1
            probability = mock_predict(frame, processed_frames)

            if probability >= threshold:
                timestamp_seconds = round((processed_frames - 1) / fps, 2)
                snapshot_path = alert_frames_folder / f"frame_{processed_frames:06d}.jpg"
                cv2.imwrite(str(snapshot_path), frame)
                alerts.append(
                    {
                        "frame_number": processed_frames,
                        "timestamp_seconds": timestamp_seconds,
                        "probability": probability,
                        "label": "possible_poaching",
                        "location": None,
                        "sms_sent": False,
                        "snapshot_path": str(snapshot_path),
                        "message": "Possible poaching activity detected",
                    }
                )
    finally:
        capture.release()

    return {"processed_frames": processed_frames, "alerts": alerts}
