"""Export a trained checkpoint to the ONNX file the API serves.

This is the *only* place the export happens — the training notebook calls this
script rather than carrying its own copy, so the opset, image size and
verification cannot drift between the two.

    python training/export_onnx.py runs/kifaru-v1/weights/best.pt

Training updates the `.pt` checkpoint; ONNX is a one-way build artifact frozen
from it for serving. Keep `best.pt` safe — a re-export needs it, and nothing
can recover it from the `.onnx`.

Verifies the result loads under onnxruntime *and* that the server can parse its
output layout, before overwriting models/best.onnx, so a broken export cannot
reach a deploy.
"""

import argparse
import ast
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def class_count(session) -> int:
    """How many classes the graph was trained on, per its embedded metadata."""
    raw = (session.get_modelmeta().custom_metadata_map or {}).get("names")
    try:
        return len(ast.literal_eval(raw))
    except (ValueError, SyntaxError, TypeError):
        from app import config

        return len(config.CLASS_NAMES)


def verify(onnx_path: Path, imgsz: int) -> None:
    """Load through onnxruntime exactly as the server does, and parse its output.

    Two distinct failures are caught here. Ultralytics can emit a graph it is
    happy with but onnxruntime rejects — better to fail now than during a
    deploy. And the graph may use an output layout the server cannot read: YOLO
    exports come in an end-to-end form with NMS baked in, `(1, N, 6)`, and a
    classic form, `(1, 4 + classes, N)`. Handing one to the other's parser does
    not raise, it silently yields nonsense boxes, so this asserts the server
    recognises what it is about to be given.

    The check reuses app.detector's own predicate deliberately: a copy of that
    rule here could agree with the server today and disagree after any edit,
    which would make this verification worse than none at all.
    """
    import numpy as np
    import onnxruntime as ort

    from app.detector import is_end_to_end

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]

    dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
    output = session.run(None, {input_meta.name: dummy})[0]

    metadata = session.get_modelmeta().custom_metadata_map or {}
    print(f"  input   {input_meta.name} {input_meta.shape}")
    print(f"  output  {output.shape}")
    print(f"  classes {metadata.get('names', '(none embedded)')}")

    if output.ndim != 3:
        sys.exit(f"\nUnusable output {output.shape}: expected a batched 3-D tensor.")

    classes = class_count(session)
    rows = np.squeeze(output, axis=0)

    if is_end_to_end(rows, classes):
        print(f"  layout  end-to-end — {rows.shape[0]} slots of "
              "[x1, y1, x2, y2, score, class], NMS inside the graph")
    elif rows.shape[0] == 4 + classes:
        print(f"  layout  classic — {rows.shape[1]} candidate boxes of "
              f"4 coords + {classes} class scores, server runs NMS")
    else:
        sys.exit(
            f"\nUnrecognised output layout {output.shape} for {classes} classes.\n"
            f"Expected (1, N, 6) end-to-end or (1, {4 + classes}, N) classic.\n"
            "app/detector.py cannot parse this — it would produce silently wrong\n"
            "boxes rather than an error, so this export is not deployable."
        )


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
