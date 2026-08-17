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


def env_aliases(name: str, default: dict[str, str]) -> dict[str, str]:
    """Parse a "from=to,from=to" mapping, falling back to the default."""
    raw = os.getenv(name)
    if not raw:
        return dict(default)

    aliases = {}
    for pair in raw.split(","):
        source, _, target = pair.partition("=")
        if source.strip() and target.strip():
            aliases[source.strip()] = target.strip()

    return aliases or dict(default)


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
# Order is the training class order and must not be reshuffled; append only.
CLASS_NAMES = ["rhino", "person", "vehicle", "weapon"]

# A rhino alone is wildlife, not poaching. These are the classes that actually
# raise an alert; a rhino in the same frame escalates severity instead.
THREAT_CLASSES = {"person", "vehicle", "weapon"}
ASSET_CLASSES = {"rhino"}

# Model label -> project label, applied to every detection. Two jobs. It
# collapses vehicle synonyms, since a model carrying COCO-style labels emits
# car/truck/bus separately and the scorer only cares that it is a vehicle. And
# it lets stock COCO weights stand in for the trained model during a smoke
# test — export yolo26n.pt, then point an override at an animal COCO does know:
#
#   LABEL_ALIASES="car=vehicle,truck=vehicle,bus=vehicle,elephant=rhino"
#
# which exercises the whole pipeline on real footage before any training.
LABEL_ALIASES = env_aliases(
    "LABEL_ALIASES",
    {
        "car": "vehicle",
        "truck": "vehicle",
        "bus": "vehicle",
        "van": "vehicle",
        "motorcycle": "vehicle",
        "motorbike": "vehicle",
    },
)

# ── Geometry ─────────────────────────────────────────────────────────────────
# Distances are measured in rhino body-lengths, using the rhino in frame as the
# ruler, so no camera calibration is needed and one threshold works for both
# aerial and ground footage. See app/geometry.py.
RHINO_BODY_LENGTH_M = env_float("RHINO_BODY_LENGTH_M", 3.7)

# Below this the rhino box is too small to measure against: box jitter of a few
# pixels would swing the estimate wildly.
MIN_RULER_PIXELS = env_int("MIN_RULER_PIXELS", 12)

# (upper bound in METRES, score, band name), nearest first. Metres because the
# policy is stated in metres — anything within 50m of a rhino is flagged — and
# a band table in body-lengths hides whether that policy is actually met.
#
# The scores are high and closely spaced on purpose. Every band inside the
# outer bound has to clear DEFAULT_ALERT_THRESHOLD on proximity alone, even
# for a subject watched and measured as stationary. The gradation that remains
# is for triage, not for deciding whether to alert at all.
#
# Bands rather than a curve because the underlying estimate is only good to
# about ±40%: a rhino seen head-on measures short.
PROXIMITY_BANDS_M = (
    (7.0, 1.00, "critical"),   # inside charging distance
    (20.0, 0.92, "high"),
    (50.0, 0.85, "medium"),    # the outer edge of "worth waking someone for"
)

# Scored when a threat is present but no rhino is measurable. Not zero: a human
# deep inside a protected area is mildly suspicious even with nothing to
# measure against. Not high either — it is an unknown, not a sighting.
UNKNOWN_PROXIMITY = env_float("UNKNOWN_PROXIMITY", 0.35)

# ── Tracking ─────────────────────────────────────────────────────────────────
# How far an object may move between analysed frames, as a multiple of its own
# size. At ~2fps sampling a walking person moves most of a body width.
TRACK_GATE_SCALE = env_float("TRACK_GATE_SCALE", 2.5)

# Ceiling on how far the gate may widen after a track has been missing, so a
# long gap does not let a track claim any object on screen.
TRACK_GATE_MAX_STRETCH = env_float("TRACK_GATE_MAX_STRETCH", 3.0)

# Frames a track survives without a match, so one missed detection does not
# restart it and throw away the distance history.
TRACK_MAX_MISSES = env_int("TRACK_MAX_MISSES", 3)

# Positions kept per track — at 2fps this is a four-second window.
TRACK_HISTORY = env_int("TRACK_HISTORY", 8)

# ── Threat scoring ───────────────────────────────────────────────────────────
# Weights sum to 1.0. Proximity dominates because it is the measure the project
# is built around; approach is close behind because a sustained approach is far
# rarer among tourists and rangers than mere proximity is.
PROXIMITY_WEIGHT = env_float("PROXIMITY_WEIGHT", 0.45)
APPROACH_WEIGHT = env_float("APPROACH_WEIGHT", 0.30)
CONTEXT_WEIGHT = env_float("CONTEXT_WEIGHT", 0.15)
PERSISTENCE_WEIGHT = env_float("PERSISTENCE_WEIGHT", 0.10)

# Closing speed, in body-lengths/second, that maxes out the approach term.
APPROACH_SATURATION = env_float("APPROACH_SATURATION", 0.5)

# Samples needed before a closing rate is trusted at all.
MIN_APPROACH_SAMPLES = env_int("MIN_APPROACH_SAMPLES", 3)

# Consecutive threatening frames that max out the persistence term.
PERSISTENCE_SATURATION = env_int("PERSISTENCE_SATURATION", 4)

# Aggravating factors. A weapon alone maxes the context term but still cannot
# raise an alert by itself — weapon recall is expected to be poor from the air,
# so it escalates, never gates.
CONTEXT_WEIGHTS = {"weapon": 1.0, "vehicle": 0.5}
GROUP_SIZE = env_int("GROUP_SIZE", 3)
GROUP_WEIGHT = env_float("GROUP_WEIGHT", 0.35)

# (minimum score, name), highest first.
SEVERITY_BANDS = ((0.75, "critical"), (0.5, "high"), (0.25, "medium"))

# Default gate on the composite score for /videos/analyze. Lower than the old
# 0.6 because that threshold was compared against a raw detection confidence,
# and a composite score is a harder thing to max out: a person well inside
# charging distance but standing still scores under 0.5 on its own.
DEFAULT_ALERT_THRESHOLD = env_float("DEFAULT_ALERT_THRESHOLD", 0.45)

# ── SMS ─────────────────────────────────────────────────────────────────────
# One message covers a whole clip, capped at ONE GSM-7 segment.
#
# Past 160 septets an SMS becomes a concatenated message, which needs a user
# data header and septet padding to stay bit-aligned. Our provider gets that
# wrong: multi-segment messages arrive as GSM-7 mojibake — Greek letters and a
# trailing run of '@' — with no error on the sending side. Alerts that do not
# fit are counted instead; the dashboard has the detail.
#
# Raise this only after verifying a >160 septet message actually arrives intact.
SMS_MAX_SEPTETS = env_int("SMS_MAX_SEPTETS", 160)

# ── Video ────────────────────────────────────────────────────────────────────
# Analysing all 30fps is pointless and far too slow on a shared CPU: a 5-minute
# clip would be 9,000 inferences, well past the 600s proxy timeout. Poaching
# activity does not vanish within half a second.
FRAME_SAMPLE_FPS = env_float("FRAME_SAMPLE_FPS", 2.0)
