"""Turn raw drone footage into YOLO-format training images.

Annotation is the real bottleneck in this project, and annotating 30 near
identical frames per second of footage is wasted effort. This samples frames at
a fixed rate so what you hand to Roboflow/CVAT is varied rather than redundant.

    python training/prepare_dataset.py datasets/raw --fps 1

Produces datasets/extracted/<video-stem>/frame_000123.jpg, ready to upload for
labelling. Labels come back as YOLO .txt files and go under datasets/poaching/.
"""

import argparse
from pathlib import Path

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}


def extract_frames(video_path: Path, output_dir: Path, target_fps: float) -> int:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"  !! could not open {video_path.name}")
        return 0

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30
    stride = max(1, round(source_fps / target_fps)) if target_fps > 0 else 1

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_number = 0
    written = 0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            frame_number += 1
            if (frame_number - 1) % stride:
                continue

            cv2.imwrite(str(output_dir / f"frame_{frame_number:06d}.jpg"), frame)
            written += 1
    finally:
        capture.release()

    print(f"  {video_path.name}: {written} frames (source {source_fps:.1f}fps)")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="directory of raw videos")
    parser.add_argument("--output", type=Path, default=Path("datasets/extracted"))
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="frames to keep per second (default 1)",
    )
    args = parser.parse_args()

    if not args.source.is_dir():
        raise SystemExit(f"No such directory: {args.source}")

    videos = sorted(
        path
        for path in args.source.rglob("*")
        if path.suffix.lower() in VIDEO_SUFFIXES
    )
    if not videos:
        raise SystemExit(f"No videos found under {args.source}")

    print(f"Extracting from {len(videos)} video(s) at {args.fps}fps:")
    total = sum(
        extract_frames(video, args.output / video.stem, args.fps) for video in videos
    )

    print(f"\n{total} frames -> {args.output}")
    print("Next: upload for annotation, then place YOLO labels in datasets/poaching/")
    print("Classes must match app/config.py — 0=rhino, 1=person, 2=vehicle")


if __name__ == "__main__":
    main()
