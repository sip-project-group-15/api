from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

app = FastAPI(title="Rhino Conservation API", version="0.1.0")
UPLOAD_DIR = Path("uploads")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/videos/analyze")
async def analyze_video(video: UploadFile = File(...), threshold: float = Form(0.6)):
    if not video.filename:
        raise HTTPException(status_code=400, detail="Video file is required")

    if not video.filename.lower().endswith(".mp4"):
        raise HTTPException(
            status_code=400,
            detail="Only .mp4 videos are supported for the MVP",
        )

    UPLOAD_DIR.mkdir(exist_ok=True)
    saved_name = f"{uuid4().hex}_{Path(video.filename).name}"
    saved_path = UPLOAD_DIR / saved_name

    content = await video.read()
    saved_path.write_bytes(content)

    return {
        "video_name": video.filename,
        "saved_video": str(saved_path),
        "threshold": threshold,
        "processed_frames": 0,
        "alerts": [],
    }
