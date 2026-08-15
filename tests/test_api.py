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


# ── Serving alert frames ─────────────────────────────────────────────────────
# The compose volume keeps snapshots across restarts, but persistence is not
# reachability — without this route the frontend has a path it cannot use.

import re
from pathlib import Path

from app.alert_store import snapshot_url


def test_snapshot_url_is_a_fetchable_route():
    url = snapshot_url("a" * 32, "uploads/aaa/alert_frames/frame_000016.jpg")

    assert url == f"/uploads/{'a' * 32}/alert_frames/frame_000016.jpg"


def test_snapshot_url_is_none_without_a_snapshot():
    assert snapshot_url("a" * 32, None) is None


def test_a_snapshot_can_be_fetched(tmp_path, monkeypatch):
    import app.main as main

    upload_id = "0123456789abcdef0123456789abcdef"
    frames = tmp_path / upload_id / "alert_frames"
    frames.mkdir(parents=True)
    (frames / "frame_000016.jpg").write_bytes(b"\xff\xd8\xff\xe0jpegbytes")
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path)

    response = client.get(f"/uploads/{upload_id}/alert_frames/frame_000016.jpg")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content.startswith(b"\xff\xd8")


def test_a_missing_snapshot_is_404(tmp_path, monkeypatch):
    import app.main as main
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path)

    upload_id = "0123456789abcdef0123456789abcdef"
    assert client.get(f"/uploads/{upload_id}/alert_frames/frame_000001.jpg").status_code == 404


def test_path_traversal_cannot_escape_the_uploads_directory(tmp_path, monkeypatch):
    """Both segments reach the filesystem, so both are allowlisted, not sanitised."""
    import app.main as main
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path)
    secret = tmp_path.parent / "secret.txt"
    secret.write_text("do not serve me")

    upload_id = "0123456789abcdef0123456789abcdef"
    for attack in ("../../secret.txt", "..%2f..%2fsecret.txt", "frame_000001.jpg.txt"):
        assert client.get(f"/uploads/{upload_id}/alert_frames/{attack}").status_code == 404
    for bad_id in ("../..", "NOTHEX" + "0" * 26, "0" * 31):
        assert client.get(f"/uploads/{bad_id}/alert_frames/frame_000001.jpg").status_code == 404
