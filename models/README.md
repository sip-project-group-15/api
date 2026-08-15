# models/

Weights served in production. **The presence of `best.onnx` here is the switch
between real inference and the mock** — nothing else needs changing.

```
models/
└── best.onnx    # exported by training/notebooks/01_train_yolo.ipynb
```

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
