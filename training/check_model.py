"""Run the production pipeline over one image or video and show its reasoning.

This is the fastest way to answer "does the model actually see anything, and
does the scorer agree?" — it imports the same detector, tracker and scorer the
API serves, so what it prints is what a deploy would do. Uploading through the
endpoint tells you only whether an alert fired; this tells you why.

    python training/check_model.py clip.mp4
    python training/check_model.py frame.jpg --model models/baseline-coco.onnx
    python training/check_model.py clip.mp4 --annotate out/

It also times inference, which is the number that decides whether a clip fits
inside the request timeout on the deployment box. Halve the machine, double the
per-frame figure.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def annotate(frame, detections, assessment):
    import cv2

    for detection in detections:
        x, y, w, h = detection["box"]
        threat = detection["label"] in ("person", "vehicle", "weapon")
        colour = (0, 0, 255) if threat else (0, 200, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
        cv2.putText(
            frame,
            f"{detection['label']} {detection['confidence']:.2f}",
            (x, max(12, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colour,
            1,
        )

    if assessment:
        cv2.putText(
            frame,
            f"{assessment['severity']} {assessment['score']:.2f}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    return frame


def report(index, timestamp, detections, assessment, elapsed_ms):
    labels = ", ".join(
        f"{d['label']}:{d['confidence']:.2f}" for d in detections
    ) or "nothing detected"
    print(f"\n[{index:>4}] t={timestamp:>6.2f}s  {elapsed_ms:>5.1f}ms  {labels}")

    if assessment is None:
        print("       no threat class in frame")
        return

    components = assessment["components"]
    print(
        f"       score {assessment['score']:.3f} ({assessment['severity']})  "
        f"prox={components['proximity']:.2f} appr={components['approach']:.2f} "
        f"ctx={components['context']:.2f} pers={components['persistence']:.2f}"
    )
    print(f"       {assessment['reason']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="image or video to analyse")
    parser.add_argument("--model", type=Path, help="override MODEL_PATH")
    parser.add_argument(
        "--aliases",
        help='override LABEL_ALIASES, e.g. "car=vehicle,elephant=rhino"',
    )
    parser.add_argument("--annotate", type=Path, help="write boxed frames here")
    parser.add_argument("--max-frames", type=int, default=40)
    args = parser.parse_args()

    if not args.source.is_file():
        sys.exit(f"No such file: {args.source}")

    # Set before importing config, which resolves the environment at import.
    if args.model:
        os.environ["MODEL_PATH"] = str(args.model)
    if args.aliases:
        os.environ["LABEL_ALIASES"] = args.aliases

    import cv2

    from app import config
    from app.detector import get_detector
    from app.threat import ThreatMonitor
    from app.video_processor import frame_stride

    detector = get_detector()
    print(f"model     {config.MODEL_PATH}")
    print(f"detector  {detector.name}")
    if detector.is_mock:
        print("\n!! No model loaded — these are simulated detections, not a test")
        print("!! of anything. Export weights first; see the notebook step 8.\n")
    print(f"aliases   {config.LABEL_ALIASES}")

    monitor = ThreatMonitor()
    if args.annotate:
        args.annotate.mkdir(parents=True, exist_ok=True)

    timings = []
    assessments = []

    if args.source.suffix.lower() in IMAGE_SUFFIXES:
        frames = [(1, 0.0, cv2.imread(str(args.source)))]
        if frames[0][2] is None:
            sys.exit(f"Could not decode {args.source}")
        print(f"\nSingle image {frames[0][2].shape[1]}x{frames[0][2].shape[0]}")
    else:
        capture = cv2.VideoCapture(str(args.source))
        if not capture.isOpened():
            sys.exit(f"Could not open {args.source}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 30
        stride = frame_stride(fps)
        print(f"\n{fps:.1f}fps source, analysing every {stride} frames")

        frames = []
        number = 0
        while len(frames) < args.max_frames:
            success, frame = capture.read()
            if not success:
                break
            number += 1
            if (number - 1) % stride == 0:
                frames.append((number, (number - 1) / fps, frame))
        capture.release()

    for number, timestamp, frame in frames:
        start = time.perf_counter()
        detections = detector.predict(frame, number)
        timings.append((time.perf_counter() - start) * 1000)

        assessment = monitor.update(detections, timestamp)
        assessments.append(assessment)
        report(number, timestamp, detections, assessment, timings[-1])

        if args.annotate:
            cv2.imwrite(
                str(args.annotate / f"frame_{number:06d}.jpg"),
                annotate(frame.copy(), detections, assessment),
            )

    if not timings:
        sys.exit("No frames analysed")

    threshold = config.DEFAULT_ALERT_THRESHOLD
    alerting = [a for a in assessments if a and a["score"] >= threshold]
    median = sorted(timings)[len(timings) // 2]

    print(f"\n{'─' * 64}")
    print(f"frames analysed   {len(timings)}")
    print(f"would alert       {len(alerting)} at threshold {threshold}")
    print(f"median inference  {median:.1f} ms/frame")
    # The deployment box is a shared 6-core VM with no GPU; whatever this
    # machine manages, assume roughly half of it there.
    print(f"est. on server    {median * 2:.0f} ms/frame")
    for minutes in (1, 5):
        sampled = minutes * 60 * config.FRAME_SAMPLE_FPS
        print(
            f"  {minutes}min clip @{config.FRAME_SAMPLE_FPS:g}fps = {sampled:.0f} frames"
            f" ~ {sampled * median * 2 / 1000:.0f}s"
        )

    if args.annotate:
        print(f"\nannotated frames -> {args.annotate}")


if __name__ == "__main__":
    main()
