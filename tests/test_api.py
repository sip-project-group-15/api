from fastapi.testclient import TestClient

from app import detector
from app.main import app

client = TestClient(app)


def test_health_reports_the_active_detector():
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    # Whichever backend is live, health must say which -- this is how a deploy
    # is verified without uploading a video.
    assert body["detector"] in {"mock", "yolo-onnx"}
    assert body["model_loaded"] is (not detector.get_detector().is_mock)


def test_rejects_non_mp4():
    response = client.post(
        "/videos/analyze",
        files={"video": ("clip.avi", b"data", "video/x-msvideo")},
    )

    assert response.status_code == 400
    assert "mp4" in response.json()["detail"].lower()


def test_rejects_out_of_range_threshold():
    response = client.post(
        "/videos/analyze",
        files={"video": ("clip.mp4", b"data", "video/mp4")},
        data={"threshold": "1.5"},
    )

    assert response.status_code == 400


def test_rejects_undecodable_video():
    """An .mp4 name is not enough -- OpenCV must actually open it."""
    response = client.post(
        "/videos/analyze",
        files={"video": ("clip.mp4", b"not really a video", "video/mp4")},
    )

    assert response.status_code == 400
    assert "could not be opened" in response.json()["detail"].lower()
