import os
import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.alert_store import read_alerts, save_alerts
from app import config
from app.detector import get_detector
from app.sms_service import send_alert_sms
from app.video_processor import process_video

DEFAULT_CORS_ALLOW_ORIGINS = (
    "http://localhost:5173,"
    "http://127.0.0.1:5173,"
    "http://localhost:3000,"
    "http://127.0.0.1:3000,"
    "https://kifaru.site,"
    "https://www.kifaru.site"
)


def get_cors_allow_origins() -> list[str]:
    raw_origins = os.getenv("CORS_ALLOW_ORIGINS", DEFAULT_CORS_ALLOW_ORIGINS)
    return [origin.strip().rstrip("/") for origin in raw_origins.split(",") if origin.strip()]


logging.basicConfig(level=logging.INFO)

UPLOAD_DIR = Path("uploads")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Resolve the detector at boot so a missing or broken model shows up in the
    # logs immediately, rather than on the first upload.
    get_detector()
    yield


app = FastAPI(title="Rhino Conservation API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_allow_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Reports which backend is live, so a deploy is verifiable without an upload."""
    detector = get_detector()
    return {
        "status": "ok",
        "detector": detector.name,
        "model_loaded": not detector.is_mock,
    }


@app.get("/alerts")
def get_alerts():
    alerts = list(reversed(read_alerts()))
    return {"count": len(alerts), "alerts": alerts}


# Both segments are matched against a strict pattern rather than sanitised.
# An allowlist cannot be talked into resolving somewhere else, which is the
# whole risk when a path segment reaches the filesystem.
UPLOAD_ID = re.compile(r"[0-9a-f]{32}")
FRAME_NAME = re.compile(r"frame_\d{6}\.jpg")


@app.get("/uploads/{upload_id}/alert_frames/{filename}")
def get_alert_frame(upload_id: str, filename: str):
    """Serve an alert's snapshot so the frontend can show the evidence.

    The compose volume already keeps these files across restarts, but a volume
    is persistence, not reachability — without this route `snapshot_path` is a
    path the browser can do nothing with.

    The route deliberately mirrors `snapshot_path` exactly, because the UI
    builds its image src by joining the API base to that value. Prefer the
    `snapshot_url` field over rebuilding the path client-side: it lets this
    layout change later without breaking the frontend.
    """
    if not UPLOAD_ID.fullmatch(upload_id) or not FRAME_NAME.fullmatch(filename):
        raise HTTPException(status_code=404, detail="No such frame")

    path = UPLOAD_DIR / upload_id / "alert_frames" / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="No such frame")

    return FileResponse(path, media_type="image/jpeg")


@app.post("/videos/analyze")
async def analyze_video(
    video: UploadFile = File(...),
    threshold: float = Form(config.DEFAULT_ALERT_THRESHOLD),
):
    """Analyse an uploaded clip frame by frame.

    `threshold` gates the composite threat score from app/threat.py — how close
    a person is to a rhino, whether they are closing, and what else is in
    frame — not a bare detection confidence. Lower it to see more marginal
    activity; raise it for high-confidence alerts only.
    """
    if not video.filename:
        raise HTTPException(status_code=400, detail="Video file is required")

    if threshold < 0 or threshold > 1:
        raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1")

    original_name = Path(video.filename).name

    if not original_name.lower().endswith(".mp4"):
        raise HTTPException(
            status_code=400,
            detail="Only .mp4 videos are supported for the MVP",
        )

    upload_id = uuid4().hex
    upload_folder = UPLOAD_DIR / upload_id
    alert_frames_folder = upload_folder / "alert_frames"
    upload_folder.mkdir(parents=True, exist_ok=True)
    alert_frames_folder.mkdir(exist_ok=True)

    saved_path = upload_folder / original_name

    content = await video.read()
    saved_path.write_bytes(content)

    try:
        result = process_video(saved_path, threshold, alert_frames_folder)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    sms_result = None

    if result["alerts"]:
        # One message for the whole clip. A text per alert would mean five for a
        # single approach — same subject, seconds apart — which costs money and
        # buries the signal. Every alert is covered, so every alert is flagged.
        sms_result = send_alert_sms(video.filename, result["alerts"])
        for alert in result["alerts"]:
            alert["sms_sent"] = sms_result["sent"]

    stored_alerts = save_alerts(
        result["alerts"],
        upload_id,
        video.filename,
        str(saved_path),
    )

    return {
        "video_name": video.filename,
        "upload_id": upload_id,
        "saved_video": str(saved_path),
        "threshold": threshold,
        "processed_frames": result["processed_frames"],
        "analyzed_frames": result["analyzed_frames"],
        "frame_stride": result["frame_stride"],
        "detector": result["detector"],
        "alerts": stored_alerts,
        "sms": sms_result,
    }
