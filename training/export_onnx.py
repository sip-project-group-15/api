"""Export a trained checkpoint to the ONNX file the API serves.

The notebook does this too, but keeping it as a script means a re-export never
requires a GPU session — useful when only the opset or image size changes.

    python training/export_onnx.py runs/kifaru-v1/weights/best.pt

Verifies the result loads under onnxruntime before overwriting models/best.onnx,
so a broken export cannot reach a deploy.
"""

import argparse
import shutil
import sys
from pathlib import Path

DEFAULT_DESTINATION = Path("models/best.onnx")


def export(checkpoint: Path, destination: Path, imgsz: int, opset: int) -> Path:
    from ultralytics import YOLO

    if not checkpoint.is_file():
        sys.exit(f"No checkpoint at {checkpoint}")

    model = YOLO(str(checkpoint))
    exported = Path(model.export(format="onnx", opset=opset, simplify=True, imgsz=imgsz))

    verify(exported, imgsz)

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported, destination)
    return destination


def verify(onnx_path: Path, imgsz: int) -> None:
    """Load through onnxruntime exactly as the server does.

    Ultralytics can emit a graph it is happy with but onnxruntime rejects; far
    better to fail here than during a deploy.
    """
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]

    dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
    output = session.run(None, {input_meta.name: dummy})[0]

    metadata = session.get_modelmeta().custom_metadata_map or {}
    print(f"  input   {input_meta.name} {input_meta.shape}")
    print(f"  output  {output.shape}")
    print(f"  classes {metadata.get('names', '(none embedded)')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="path to best.pt")
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--imgsz", type=int, default=640)
    # 12 is broadly supported by onnxruntime builds; raise only if a newer op
    # is genuinely needed.
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args()

    written = export(args.checkpoint, args.destination, args.imgsz, args.opset)
    size_mb = written.stat().st_size / 1e6
    print(f"\nWrote {written} ({size_mb:.1f} MB)")
    print("Commit it and push to main to deploy.")


if __name__ == "__main__":
    main()
