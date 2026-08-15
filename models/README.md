# models/

Weights served in production. **The presence of `best.onnx` here is the switch
between real inference and the mock** — nothing else needs changing.

```
models/
├── best.onnx           # committed, deployed — exported from a trained best.pt
└── baseline-coco.onnx  # optional, gitignored — stock COCO, for local testing
```

## What is and is not kept here

Only `best.onnx` is committed. Three things deliberately live elsewhere:

- **`yolo26n.pt`, the base model.** Ultralytics downloads it on demand from a
  version-pinned URL in about four seconds, and `ultralytics` is pinned in the
  notebook, so committing it would vendor a dependency that is already fixed.
- **`best.pt`, the trained checkpoint.** It stays in Drive. It is the model —
  the thing training actually updates, the only artifact a re-export or a
  resumed run can start from — and nothing can recover it from an `.onnx`. It
  is too large for git and useless to the server.
- **`baseline-coco.onnx`.** Gitignored and excluded from the Docker image; see
  below.

## Installing a trained model

1. Run `training/notebooks/01_train_yolo.ipynb` on Colab.
2. Download `best.onnx` from `Drive/MyDrive/kifaru/weights/`.
3. Drop it here, commit, push to `main`.

The image bakes this directory in at build time, so a deploy ships the weights.
Confirm which backend is live without uploading anything:

```bash
curl -s https://api.kifaru.site/health
# {"status":"ok","detector":"yolo-onnx","model_loaded":true,...}
```

`"detector": "mock"` means the file did not reach the image.

## Why ONNX and not .pt

The server has 6 shared cores and no GPU. `onnxruntime` keeps the image around
250MB and starts in ~1s; shipping PyTorch would mean ~2GB and 10–20s. `*.pt` is
gitignored for that reason — keep checkpoints in Drive and commit only the
export.

Class names are read from the ONNX metadata that Ultralytics writes at export,
so served labels always match the weights even if `app/config.py` drifts.

## baseline-coco.onnx

Stock COCO weights exported to ONNX. COCO has no rhino, but it detects `person`
and vehicles well, so this exercises the full pipeline — detection, tracking,
distance, scoring — on real footage before any training has happened:

```bash
python training/check_model.py clip.mp4 --model models/baseline-coco.onnx \
    --aliases "car=vehicle,truck=vehicle,bus=vehicle,elephant=rhino"
```

Aliasing an animal COCO *does* know into the rhino slot gives the geometry
layer something to measure against. See `LABEL_ALIASES` in `app/config.py`.

It is gitignored **and** excluded from the Docker image, because a model that
has never seen a rhino must not be deployable by accident.

To regenerate it you need `ultralytics`, which pulls in PyTorch and is not in
`requirements.txt` — that is the point of serving ONNX. Either install it into
a throwaway environment, or export it in Colab and download the result:

```bash
python training/export_onnx.py yolo26n.pt --destination models/baseline-coco.onnx
```

`yolo26n.pt` does not need to exist first; Ultralytics fetches it.
