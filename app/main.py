import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_allow_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_DIR = Path("uploads")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Resolve the detector at boot so a missing or broken model shows up in the
    # logs immediately, rather than on the first upload.
    get_detector()
    yield


app = FastAPI(title="Rhino Conservation API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/alerts")
def get_alerts():
    alerts = list(reversed(read_alerts()))
    return {"count": len(alerts), "alerts": alerts}


@app.post("/videos/analyze")
async def analyze_video(video: UploadFile = File(...), threshold: float = Form(0.6)):
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
        sms_result = send_alert_sms(video.filename, result["alerts"][0])
        result["alerts"][0]["sms_sent"] = sms_result["sent"]

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
