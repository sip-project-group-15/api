from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.video_processor import process_video

app = FastAPI(title="Rhino Conservation API", version="0.1.0")
UPLOAD_DIR = Path("uploads")


@app.get("/health")
def health_check():
    return {"status": "ok"}


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

    return {
        "video_name": video.filename,
        "upload_id": upload_id,
        "saved_video": str(saved_path),
        "threshold": threshold,
        "processed_frames": result["processed_frames"],
        "alerts": result["alerts"],
    }
