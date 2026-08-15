# Rhino Conservation API

FastAPI backend for the AI-powered aerial surveillance MVP.

## Setup

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

```powershell
python -m uvicorn app.main:app --reload
```

Open this in your browser:

```text
http://127.0.0.1:8000/health
```

You can also open the API docs:

```text
http://127.0.0.1:8000/docs
```

## How detection works

Poaching is not an object, so YOLO is never asked to detect it. The model
reports only what is in a frame — `rhino`, `person`, `vehicle`, `weapon` — and
the judgement is made above it, where it can be read and tuned without
retraining.

| module | responsibility |
| --- | --- |
| `app/detector.py` | objects and boxes, nothing else |
| `app/tracking.py` | identity across frames, so motion is measurable |
| `app/geometry.py` | how far apart things are, in rhino body-lengths |
| `app/threat.py` | the score, and the sentence explaining it |

Distances are measured in **rhino body-lengths** rather than pixels. An adult
is roughly 3.7m nose to tail and is already in the frame, so it serves as a
ruler that needs no camera calibration — which is what lets one threshold work
for both ground-level and aerial footage. Positions are taken at the bottom
edge of each box, where the subject meets the ground, so oblique ground shots
do not read "tall" as "close".

The score is a weighted sum of four components, all reported on every alert:

```
score = 0.45 * proximity     # how close, banded (~7m / ~20m / ~45m)
      + 0.30 * approach      # is that distance shrinking, and how fast
      + 0.15 * context       # weapon, vehicle, group size
      + 0.10 * persistence   # how many consecutive frames it has held
```

Approach carries nearly as much weight as proximity because a sustained,
direct approach is far rarer among tourists and rangers than mere proximity is.
Weapons only ever escalate an alert and can never raise one alone — recall on a
rifle at aerial resolution is expected to be poor, and gating on it would sink
alert recall with it.

Every weight and threshold is in `app/config.py` and overridable by environment
variable. The scoring components are pure functions over boxes, so they can be
tuned against `tests/test_threat.py` in under a second with no model, video or
GPU.

## Tests

```powershell
python -m pytest
```
