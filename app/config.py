"""Runtime configuration, read from the environment.

Values are resolved once at import. In production the environment is populated
by docker compose at container start (see the deploy step in the CI workflow),
so there is no .env file on the server — load_dotenv is a local-dev convenience
that becomes a no-op there.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# ── Model ────────────────────────────────────────────────────────────────────
# The presence of this file is what decides real-vs-mock inference. Nothing else
# needs to change to switch modes: drop the weights in and restart.
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/best.onnx"))

# Boxes below this score are discarded before non-max suppression. Distinct from
# the per-request `threshold`, which decides whether a frame raises an alert.
MODEL_CONF_THRESHOLD = env_float("MODEL_CONF_THRESHOLD", 0.35)
MODEL_IOU_THRESHOLD = env_float("MODEL_IOU_THRESHOLD", 0.45)

# Must match the imgsz the ONNX graph was exported at, or coordinates come back
# wrong. The training notebook exports at 640.
MODEL_IMAGE_SIZE = env_int("MODEL_IMAGE_SIZE", 640)

# Fallback only. Ultralytics embeds class names in the ONNX metadata, and the
# detector prefers those — this is used when the metadata is absent.
CLASS_NAMES = ["rhino", "person", "vehicle"]

# A rhino alone is wildlife, not poaching. These are the classes that actually
# raise an alert; a rhino in the same frame escalates severity instead.
THREAT_CLASSES = {"person", "vehicle"}
ASSET_CLASSES = {"rhino"}

# ── Video ────────────────────────────────────────────────────────────────────
# Analysing all 30fps is pointless and far too slow on a shared CPU: a 5-minute
# clip would be 9,000 inferences, well past the 600s proxy timeout. Poaching
# activity does not vanish within half a second.
FRAME_SAMPLE_FPS = env_float("FRAME_SAMPLE_FPS", 2.0)
