# Training and model architecture

How the poaching detector was built, what it learned, what it cannot do, and
why each decision was made. Companion to [DATASETS.md](DATASETS.md), which
covers the data sources themselves.

Written against commit `9e6cd82`.

---

## 1. The idea the whole system rests on

**YOLO cannot detect poaching.** Poaching is not an object with an appearance;
it is a *relationship between objects over time* — a person closing on a rhino.
Asking a detector to learn it directly would need a labelled poaching dataset,
which does not exist, and would produce a black box a ranger could not argue
with.

So the model detects objects only, and the judgement lives above it in ordinary
Python that can be read, tested and tuned without retraining anything:

```
video ─> detector ─> tracker ─> geometry ─> threat scorer ─> alert
         (what)      (which)    (how far)   (how bad)
```

| file | responsibility |
|---|---|
| [`app/detector.py`](../app/detector.py) | objects and boxes, nothing else |
| [`app/tracking.py`](../app/tracking.py) | identity across frames, so motion is measurable |
| [`app/geometry.py`](../app/geometry.py) | how far apart things are |
| [`app/threat.py`](../app/threat.py) | the score, and the sentence explaining it |
| [`app/video_processor.py`](../app/video_processor.py) | orchestration and frame sampling |
| [`app/config.py`](../app/config.py) | every threshold and weight, env-overridable |
| [`app/main.py`](../app/main.py) | the FastAPI surface |

Everything in `geometry.py` and `threat.py` is a pure function over plain
dicts, so thresholds can be tuned against [`tests/test_threat.py`](../tests/test_threat.py)
in under a second with no model, no video and no GPU.

---

## 2. Measuring distance without a calibrated camera

### The problem

Pixel distance is meaningless on its own. Eighty pixels is tens of metres from a
drone at altitude and one step from a ground camera. Calibrating every camera
and altitude is not realistic for this project.

### The solution: the rhino is the ruler

An adult white rhino is roughly **3.7 m** nose to tail, and it is already in the
frame. So every distance is expressed in *rhino body-lengths*:

```python
separation = foot_point_distance_in_pixels / max(rhino_box_width, rhino_box_height)
```

This is scale-invariant and needs no calibration — the same scene at two zoom
levels reads identically. That property is asserted directly in
[`tests/test_geometry.py`](../tests/test_geometry.py) (`test_separation_is_scale_invariant`).

Two details make one formula work for both aerial and ground footage:

- **Foot points, not centroids.** Positions are the bottom-centre of each box,
  which is roughly where the subject meets the ground. In an oblique ground-level
  view a centroid mostly encodes how *tall* something is, so using it would read
  "tall" as "close".
- **The ruler is the box's longest side.** From directly above, the animal's long
  axis can lie along either image axis.

### Bands, not a precise number

The estimate is good to roughly **±40%** — a head-on rhino measures short — so
the result is banded rather than quoted as metres of fact:

| body-lengths | approx. | band | score |
|---|---|---|---|
| < 2 | < 7 m | critical | 1.00 |
| 2–5 | 7–20 m | high | 0.75 |
| 5–12 | 20–45 m | medium | 0.40 |
| > 12 | — | — | 0.00 |

A rhino smaller than `MIN_RULER_PIXELS` (12 px) is refused as a ruler entirely:
at that size a few pixels of box jitter would swing the estimate by hundreds of
percent, and no reading beats a confidently wrong one.

---

## 3. Tracking, and why not IoU

Proximity is a snapshot; **approach is its derivative**, and you cannot take a
derivative without knowing that the person in this frame is the person from the
last one.

[`app/tracking.py`](../app/tracking.py) is a greedy nearest-foot-point tracker.
Matching is by **distance, not IoU**, and that is deliberate: the API analyses
about 2 frames per second, and over half a second a walking person's boxes
frequently do not overlap at all. IoU matching would silently break every track
and destroy the distance history that "approaching" depends on. This is pinned
by `test_boxes_that_no_longer_overlap_still_match`.

The match gate scales two ways:

- **By object size**, so it means the same for a person filling the frame and one
  twenty pixels tall.
- **By elapsed time**, capped at 3×. A fixed gate would be quietly calibrated to
  one value of `FRAME_SAMPLE_FPS`; halve the sample rate and every track would
  break.

Tracks survive `TRACK_MAX_MISSES` (3) frames without a match, so one missed
detection does not restart a track and throw away its history.

---

## 4. The score

Four components, a plain weighted sum, every term reported on the alert. It is
not learned, deliberately — a ranger acting on an alert needs *"a person closed
from 22 m to 6 m over eight seconds"*, not a float.

```
score = 0.45 × proximity     how close, banded
      + 0.30 × approach      is that distance shrinking, and how fast
      + 0.15 × context       weapon, vehicle, group size
      + 0.10 × persistence   how many consecutive analysed frames it has held
```

**Approach** is a least-squares slope over the pair's distance history, not a
first-minus-last difference, so one noisy box does not read as a charge. It
needs `MIN_APPROACH_SAMPLES` (3) points before it is trusted at all, and
saturates at `APPROACH_SATURATION` (0.5 body-lengths/second). Retreating and
standing still both score zero.

**Context** is additive and capped: weapon 1.0, vehicle 0.5, a group of ≥3
people 0.35. Nothing here can raise an alert alone — at 0.15 weight, a maxed
context term is well under the 0.45 threshold.

**Severity** bands the total: ≥0.75 critical, ≥0.5 high, ≥0.25 medium.

Two judgement calls worth knowing about:

- **A rhino alone is never an alert.** Wildlife presence is not poaching. It only
  ever *raises* the score of a threat that is already present.
- **No rhino in frame scores a baseline of 0.35, not zero.** A human deep inside a
  protected area is mildly suspicious even with nothing to measure against.
  Scoring it zero would make it read identically to an empty frame.

---

## 5. The model

**YOLO26n** — 2,375,616 parameters, 5.3 GFLOPs. Nano is the only sane size for
a 6-core CPU server with no GPU.

| class | id | trained? |
|---|---|---|
| rhino | 0 | yes |
| person | 1 | yes |
| vehicle | 2 | yes |
| weapon | 3 | **declared, zero instances** |

`weapon` is in the class list but has no data — see [DATASETS.md](DATASETS.md#weapons)
for why none usable exists. Keeping the slot costs a few unused parameters and
keeps ids stable for when data arrives. The scorer treats it as an escalator
that can never raise an alert alone.

### Serving ONNX, not PyTorch

The deployment box has 6 shared cores and no GPU. `onnxruntime` keeps the image
around 250 MB starting in ~1 s; shipping PyTorch would mean ~2 GB and 10–20 s.

`*.pt` is gitignored; only `models/best.onnx` is committed. See
[`models/README.md`](../models/README.md).

### The output-layout trap

YOLO26 exports **end-to-end**, with non-max suppression inside the graph. Its
output is `(1, 300, 6)` — 300 finished detections of
`[x1, y1, x2, y2, score, class]`. Older YOLOv8/v11 heads emit
`(1, 4 + classes, 8400)` of raw centre-xywh plus per-class scores, leaving NMS
to the server.

**Feeding one layout to the other's parser does not raise an error.** It silently
produces confident, fictional boxes, and every distance computed from them is
meaningless. This was a real bug in this repo, fixed in `ec5c3c9`.

[`app/detector.py`](../app/detector.py) now picks the parser by shape before any
value is read (`is_end_to_end`), and
[`training/export_onnx.py`](export_onnx.py) verifies an export using **that same
predicate** rather than a copy — a duplicated rule could agree today and diverge
after any edit, which would make the verification worse than none.

---

## 6. Building the dataset

Full source list and licences: **[DATASETS.md](DATASETS.md)**.

Three sources, merged by [`training/build_dataset.py`](build_dataset.py):

| source | supplies | viewpoint | fetch |
|---|---|---|---|
| African Wildlife | `rhino` | ground | `african-wildlife.yaml` |
| VisDrone | `person`, `vehicle` | **aerial** | `VisDrone.yaml` |
| COCO val2017 | `person`, `vehicle` | **ground** | [`fetch_coco.py`](fetch_coco.py) |

```bash
python training/fetch_coco.py --output /content/datasets/coco-ground

python training/build_dataset.py \
    --source african-wildlife.yaml:rhino=0 \
    --source VisDrone.yaml:pedestrian=1,people=1,car=2,van=2,truck=2,bus=2,motor=2 \
    --source /content/datasets/coco-ground \
    --output /content/datasets/kifaru-merged \
    --cap 1000
```

### Remapping by name, not id

Source class ids are arbitrary. `rhino` is class 2 in African Wildlife and class
0 for us. A positional remap would silently mislabel an entire dataset, so the
mapping is by name and an unrecognised name is reported rather than ignored.

### The caps, and why they exist

**Instance cap** (`--cap`). VisDrone frames carry hundreds of tiny vehicles and
pedestrians each. Merged unrestricted against a few hundred rhino boxes that is
roughly **1000:1**, and a detector trained on that stops predicting the rare
class while its headline mAP still looks respectable.

The cap applies **per source, not globally** — deliberately. `person` has to be
learned from the air *and* the ground; a global cap would let whichever source
is read first consume the entire budget and starve the other viewpoint,
recreating the exact failure of run v2 below.

**Background ratio** (default 0.15). Images with no labels teach the model what
an empty scene looks like and suppress false positives — but the original
hand-rolled merge produced **75% background images**, which mostly teaches that
predicting nothing is right.

### COCO specifics

[`fetch_coco.py`](fetch_coco.py) uses **val2017** rather than train2017: 816 MB
instead of 19 GB, still measured at **10,777 person and 2,982 vehicle instances
across 2,951 images**, and genuinely unseen — the YOLO base weights were
pretrained on train2017.

- Only leaf categories are mapped (`car`/`truck`/`bus`/`motorcycle` → `vehicle`),
  so nothing is boxed twice.
- `iscrowd` regions are **dropped**. One box drawn over many overlapping people
  would train a single enormous "person".
- Labels are written in our ids directly, so there is no second place for the
  mapping to drift.

### The balance report

`build_dataset.py` ends by printing the class balance and classifying it:

- **no threat class at all** → FATAL, "cannot raise a single alert"
- **no rhino** → FATAL, "no ruler to measure distances against"
- anything else empty → a note (expected for `weapon`)
- otherwise, the imbalance ratio, warned above 10:1

*Which* class is missing matters far more than how many. Warning identically
about a deliberately-empty `weapon` and a fatally-empty `person` trains the
reader to ignore the message that matters.

---

## 7. How Colab was used

### The one fact that explains everything

A Colab runtime is a **temporary Linux VM in a Google data centre**. The
notebook is an editing surface; the code runs there. This holds whether you use
the browser or the VS Code extension.

| path | location | survives the session? |
|---|---|---|
| `/content/datasets/...` | VM | **no** |
| `MyDrive/kifaru/runs/` | Drive | yes |
| `MyDrive/kifaru/weights/` | Drive | yes |

`/content` is wiped when the session ends (~90 min idle, 12 h hard cap, and free
tier VMs get reclaimed sooner). Checkpoints therefore write to Drive every
epoch. The **dataset deliberately does not** — it is thousands of small files,
Drive is a network mount, and training would stall on I/O reading it every
epoch. It rebuilds in a few minutes.

**Consequence:** if the VM is reclaimed, re-run the dataset cell *before*
resuming. The checkpoint survives; the data it points at does not.

### The notebook

[`training/notebooks/01_train_yolo.ipynb`](notebooks/01_train_yolo.ipynb),
18 cells:

1. **Setup** — installs pinned deps, reports whether it is on a Colab VM,
   clones the repo, prints the commit
2. **Mount Drive** — establishes `runs/` and `weights/`
3. **Config** — epochs, imgsz, `RUN_NAME`; reads `CLASSES` from `app/config.py`
4. **Build the training set** — `fetch_coco.py` then `build_dataset.py`
5. **Train** — resumable from Drive
6. **Validate** — per-class precision/recall
7. **Export** — calls `export_onnx.py`
8. **CPU timing** — raw graph throughput

The export is **not reimplemented in the notebook**. It clones the repo and
calls [`export_onnx.py`](export_onnx.py), so the opset, image size and layout
verification cannot drift from what the server does.

### Repo access

The notebook clones from GitHub, so **anything only on your laptop is invisible
to Colab.** Push before training.

The clone is anonymous first and only reaches for a token if refused — a public
repo then costs no secret-fetch timeout. If the repo is private, a fine-grained
PAT goes in a Colab secret named `GH_TOKEN`.

> **Caveat that cost us an hour:** Colab secrets are readable **only from the
> Colab web UI**. Through the VS Code extension `userdata.get()` times out with
> *"Secrets can only be fetched when running from the Colab UI."* A private repo
> therefore cannot be cloned that way at all. We made the repo public.

If the repo is already on the runtime, cell 1 detects it and skips the clone
entirely — which also means it uses your *uncommitted* edits.

### The resume trap

Cell 5 resumes automatically if `runs/<RUN_NAME>/weights/last.pt` exists. But
`model.train(resume=True)` restores the **original arguments, including the
dataset**. After changing the data you must change `RUN_NAME`, or the run
silently continues with the old configuration. Each run here got a new name:
`kifaru-v1`, `-v2`, `-v3`.

### `.pt` versus `.onnx`

**Training only ever updates `best.pt`.** It is the model — what accumulates the
training, what a resumed run continues from, and the only thing a re-export can
start from. ONNX is a one-way build artifact frozen out of it for serving; you
cannot train it, and nothing recovers a `.pt` from an `.onnx`.

So `best.pt` stays in Drive. Only `best.onnx` is downloaded, committed, and
deployed.

### Hyperparameters

```python
base_model = "yolo26n.pt"   # nano — the only sane size for the target CPU
epochs     = 30             # fine-tuning a pretrained backbone converges fast
patience   = 10
imgsz      = 640            # MUST match the ONNX export and server preprocessing
batch      = 16
seed       = 0
```

`imgsz=640` is load-bearing. It is frozen into the ONNX graph and mirrored by
`MODEL_IMAGE_SIZE`; changing one without the other returns wrong coordinates.
Do not lower it for speed — aerial targets are tiny (SAVMAP's average annotation
is **25 × 23 pixels**).

30 epochs, not 100: most of the gain lands in the first 20 or so, and a ~25
minute run means less exposure to the VM being reclaimed. Training on a T4 ran
~51 s/epoch.

**Use the GPU, not the TPU.** Ultralytics has no XLA backend, and detection's
dynamic shapes (variable box counts, NMS) would force constant recompilation.

---

## 8. What actually happened across three runs

The two failed runs are the most instructive part of this document. Both
produced *plausible-looking metrics* while being incapable of raising an alert.

### Run 1 — rhino only

Merged African Wildlife alone: **399 rhino instances, zero person, zero
vehicle**, across 1052 images of which **790 were background**.

Two failures, one subtle:

- The scorer only alerts on `person`/`vehicle`/`weapon`. With none of them
  trained, `ThreatMonitor.update()` returns `None` on every frame. **Zero
  alerts, structurally.**
- Worse, Ultralytics logged `Remapped 1/3 cls head rows from pretrained weights
  by class name` — `person` had inherited COCO's pretrained weights. Training on
  1052 images containing no people **actively suppressed** it. The run did not
  merely fail to teach the threat classes; it unlearned the one that came free.

### Run 2 — rhino (ground) + person/vehicle (aerial)

Added VisDrone. Validation looked reasonable. Measured on the real exported
weights, it was not:

| test | result |
|---|---|
| rhino, ground-level (55 val images) | 95% detected, mean confidence 0.902 |
| person/vehicle, aerial (60 VisDrone images) | 92% of images, 869 detections |
| **person/vehicle, ground-level** | **nothing — max score 0.016** |
| **end-to-end alert on aerial footage** | **0.310, under the 0.45 threshold** |

Each class had only learned its source's viewpoint, so **no single viewpoint saw
all three classes**:

- *Aerial* → threats detected, rhino missed → no ruler → proximity pinned at the
  0.35 unknown baseline. A frame with **13 people and 2 vehicles** still scored
  only 0.310.
- *Ground* → rhino detected, no threats → no alert by design.

Zero alerts in either domain. This is the mirror image of the warning already in
the notebook that ground-level rhino would not transfer to aerial — the same
principle applied to person and vehicle, in reverse.

### Run 3 — plus ground-level COCO

Adding [`fetch_coco.py`](fetch_coco.py) closed the ground-level half.

---

## 9. Results (run 3)

### Validation, 167 images / 1208 instances

| class | precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **rhino** | 0.847 | **0.918** | 0.946 | 0.838 |
| **person** | 0.666 | **0.423** | 0.485 | 0.271 |
| **vehicle** | 0.627 | **0.420** | 0.456 | 0.280 |
| weapon | — | — | — | — |
| **all** | 0.713 | 0.587 | 0.629 | 0.463 |

### Independent checks

Run with [`check_model.py`](check_model.py), which drives the *production*
detector, tracker and scorer rather than Ultralytics' own validation:

| test | v2 | v3 |
|---|---|---|
| Ground-level person/vehicle | nothing (max 0.016) | person 0.98, vehicle 0.90 |
| Rhino batch, 55 ground images | 95%, conf 0.902 | 89%, conf **0.934** |
| Aerial batch, 60 images | 92%, 869 dets, 0.716 | **98%, 969 dets, 0.861** |
| End-to-end alert | impossible | **critical, 0.850** |

### The end-to-end test

No public image shows a person near a rhino, so one was **synthesised**: a real
person crop composited into a real rhino photograph, walking from 2.6 to 0.5
body-lengths over 40 frames.

```
[ 1] rhino:0.97, person:0.71   score 0.362 (medium)   prox=0.75 appr=0.00 pers=0.25
                               person 2.6 body-lengths from a rhino (~9.7m, high)
[11] rhino:0.97, person:0.64   score 0.713 (high)     prox=0.75 appr=1.00 pers=0.75
                               2.1 body-lengths (~7.7m); closing at 0.54 body-lengths/s
[16] rhino:0.97, person:0.84   score 0.850 (critical) prox=1.00 appr=1.00 pers=1.00
                               1.8 body-lengths (~6.7m, critical); closing at 0.55/s
[31] rhino:0.97, person:0.58   score 0.850 (critical) 1.0 body-lengths (~3.7m)

would alert 5 of 8 analysed frames
```

Every layer behaved: detection → the track held across all frames → distance in
body-lengths → approach appeared exactly when three samples existed → severity
escalated medium → high → critical. The measured closing rate (0.55 bl/s)
matches the synthesised motion (0.6 bl/s).

---

## 10. Caveats

### Blocking — the model is not field-ready

- **No aerial rhino data exists.** `rhino` is trained entirely on ground-level
  photography. On top-down drone frames the silhouette, scale and background are
  all different, and it will underperform badly. **This is the single most
  important gap**, because losing the rhino loses the *ruler* — every distance
  becomes unmeasurable and every frame falls back to the 0.35 baseline. Only
  your own annotated drone footage can close it.
- **Person/vehicle recall is 0.42.** The model misses about 58% of them. For this
  application that is backwards: a missed poacher is far worse than a false alarm
  a ranger dismisses. In the end-to-end test the person was lost at the *closest*
  range, exactly where it matters most. Lowering `MODEL_CONF_THRESHOLD` from 0.35
  to ~0.25 buys recall cheaply; the scorer filters the extra noise downstream via
  persistence and proximity.
- **`weapon` is untrained** and will never fire.

### Method

- **The end-to-end test is synthetic.** It proves the pipeline reasons correctly
  on real detections; it is **not** evidence about field accuracy.
- **There is no scenario-level evaluation set.** Detector mAP is only half the
  story — the scorer needs its own test set of 20–30 clips labelled
  poaching/benign, including hard negatives (tourists, rangers, vehicles on
  roads), measured as *alert* precision and recall. The two move independently
  and only the second is what a demo shows. This does not exist yet.
- **Every weight and threshold is a reasoned guess**, not fitted to data. The
  0.45/0.30/0.15/0.10 split, the band edges, `APPROACH_SATURATION` — all are
  defensible and all are untuned.
- **COCO val2017 is used as training data.** Fine here (we do not benchmark on
  COCO, and it was unseen by the pretrained weights) but worth stating.

### Measurement assumptions

- **The 3.7 m body-length is an average**, and a head-on rhino measures short.
  Expect ±40% on any metre figure. This is why the code bands distances rather
  than quoting metres as fact.
- **Foot points assume ground contact.** A partially occluded or elevated subject
  reads as further away than it is.
- **The tracker is greedy with no re-identification.** Two people crossing paths
  can swap ids, which corrupts that pair's distance history.
- **`UNKNOWN_PROXIMITY = 0.35` is a judgement call**, not a measurement.

### Engineering

- **`process_video` blocks the event loop.** [`app/main.py`](../app/main.py)
  declares `async def analyze_video` but calls the synchronous
  [`process_video`](../app/video_processor.py) directly, so the entire server
  freezes for the duration of every upload. On a shared 6-core box this will bite
  under any concurrency. The fix is a background job returning `upload_id`
  immediately; **still outstanding.**
- **onnxruntime thread count is not pinned.** ORT will grab every core and fight
  uvicorn for them; `intra_op_num_threads` should be set explicitly.
- **No INT8 quantisation.** `quantize_dynamic` would give roughly 2–3× for a small
  mAP cost.
- **Alerts are stored in a JSON file** ([`app/alert_store.py`](../app/alert_store.py)),
  not a database.
- **The SMS message does not carry the reasoning.**
  [`app/sms_service.py`](../app/sms_service.py) sends only the probability,
  discarding the "closed from 22 m to 6 m" sentence that makes an alert
  actionable.
- **One model for both viewpoints.** `MODEL_PATH` indirection makes a
  `view=aerial|ground` switch nearly free if the mixed model proves too
  compromised.
- **Server timing is extrapolated**, not measured — `check_model.py` assumes ~2×
  the local per-frame cost. Calibrate on the real box.
- **Class balance is ~5:1** (399 rhino against ~2000 each of person and vehicle).
  Rhino is the weakest class and the one that gates everything.

---

## 11. Reproducing a run

```bash
# 1. Push. Colab clones from GitHub; local-only work is invisible to it.
git push origin main

# 2. Colab: Runtime -> Disconnect and delete runtime, then run cells top to bottom.
#    Confirm the commit hash cell 1 prints.
#    Change RUN_NAME if the dataset changed since the last run.

# 3. Stop at the balance table. FATAL means do not train.
#    Expect ~399 rhino / ~2000 person / ~2000 vehicle / 0 weapon.

# 4. Train (~25 min), validate, export.

# 5. Download best.onnx from MyDrive/kifaru/weights/ into models/best.onnx.

# 6. Verify before trusting it:
python training/check_model.py <clip.mp4> --annotate out/
# Check out/ — misplaced boxes mean the export layout is being misread,
# and no amount of scorer tuning will fix that.

# 7. Deploy.
git add models/best.onnx && git commit -m "feat: add trained weights" && git push
```

Local test suite (108 tests, no model or GPU required):

```bash
python -m pytest
```

---

## 12. Next, in priority order

1. **Aerial rhino footage.** Everything else is secondary; without it the ruler
   fails in the target domain.
2. **A scenario-level evaluation set** — 20–30 clips labelled poaching/benign
   with hard negatives. Without it, "does it work?" has no answer.
3. **Move `process_video` off the event loop.**
4. **Try `MODEL_CONF_THRESHOLD=0.25`** and measure alert precision/recall against
   (2).
5. **60 epochs** now that the data composition is right.
6. **More rhino instances** — Open Images V7 `Rhinoceros` and the Roboflow sets in
   [DATASETS.md](DATASETS.md#more-rhinos).
7. **A separate thermal/night model** from BIRDSAI. Poaching happens at night, and
   the anti-poaching literature is unanimous that thermal outperforms RGB.
