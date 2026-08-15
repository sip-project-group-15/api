import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ALERTS_FILE = Path("uploads") / "alerts.json"


def read_alerts(alerts_file: Path = ALERTS_FILE) -> list[dict]:
    if not alerts_file.exists():
        return []

    try:
        alerts = json.loads(alerts_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    if not isinstance(alerts, list):
        return []

    return alerts


def save_alerts(
    alerts: list[dict],
    upload_id: str,
    video_name: str,
    saved_video: str,
    alerts_file: Path = ALERTS_FILE,
) -> list[dict]:
    if not alerts:
        return []

    alerts_file.parent.mkdir(parents=True, exist_ok=True)
    existing_alerts = read_alerts(alerts_file)
    created_at = datetime.now(timezone.utc).isoformat()
    stored_alerts = []

    for alert in alerts:
        stored_alerts.append(
            {
                "id": uuid4().hex,
                "upload_id": upload_id,
                "video_name": video_name,
                "saved_video": saved_video,
                "created_at": created_at,
                **alert,
            }
        )

    updated_alerts = existing_alerts + stored_alerts
    temp_file = alerts_file.with_name(f"{alerts_file.name}.tmp")
    temp_file.write_text(json.dumps(updated_alerts, indent=2), encoding="utf-8")
    temp_file.replace(alerts_file)

    return stored_alerts
