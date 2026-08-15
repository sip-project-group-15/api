"""Fetch ground-level people and vehicles from COCO, in our class space.

The first trained model detected rhinos from the ground and people from the
air, and nothing could see all three classes at once — so it never alerted.
This closes the ground-level half of that gap.

    python training/fetch_coco.py --output /content/datasets/coco-ground

Writes a YOLO directory that `build_dataset.py` consumes directly:

    --source /content/datasets/coco-ground

COCO's **val2017** split is used rather than train2017: it is 816MB instead of
19GB, and it still carries roughly eleven thousand person instances, far more
than the caps ever take. It is also genuinely unseen data — the YOLO base
weights were pretrained on train2017, so val2017 was never in that.

Labels are written in our ids directly (`app/config.py`), so no remap is
needed downstream and there is no second place for the mapping to drift.
"""

import argparse
import json
import random
import shutil
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# COCO category name -> our class name. Only leaf categories are used: COCO has
# no overlapping parent like "vehicle", so nothing is boxed twice.
COCO_TO_OURS = {
    "person": "person",
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "motorcycle": "vehicle",
}


def our_class_ids() -> dict[str, int]:
    from app import config

    by_name = {name: index for index, name in enumerate(config.CLASS_NAMES)}
    return {
        coco: by_name[ours] for coco, ours in COCO_TO_OURS.items() if ours in by_name
    }


def to_yolo(box: list[float], width: int, height: int) -> tuple[float, ...] | None:
    """COCO's absolute [x, y, w, h] to YOLO's normalised centre form.

    Returns None for a box that is degenerate or lies outside the image; a few
    such rows exist in COCO and a zero-area box is a bad training target.
    """
    x, y, w, h = box
    if w <= 0 or h <= 0 or width <= 0 or height <= 0:
        return None

    left, top = max(0.0, x), max(0.0, y)
    right, bottom = min(float(width), x + w), min(float(height), y + h)
    if right <= left or bottom <= top:
        return None

    return (
        ((left + right) / 2) / width,
        ((top + bottom) / 2) / height,
        (right - left) / width,
        (bottom - top) / height,
    )


def group_annotations(data: dict, wanted: dict[str, int]) -> dict[int, list[str]]:
    """Per image id, the YOLO label lines we keep. Images with none are omitted."""
    keep = {
        category["id"]: wanted[category["name"]]
        for category in data["categories"]
        if category["name"] in wanted
    }
    sizes = {
        image["id"]: (image["width"], image["height"]) for image in data["images"]
    }

    lines: dict[int, list[str]] = {}
    for annotation in data["annotations"]:
        class_id = keep.get(annotation["category_id"])
        if class_id is None or annotation.get("iscrowd"):
            # Crowd regions are one box over many overlapping people. Training
            # on them teaches a single enormous "person", so they are dropped.
            continue

        width, height = sizes.get(annotation["image_id"], (0, 0))
        converted = to_yolo(annotation["bbox"], width, height)
        if converted is None:
            continue

        lines.setdefault(annotation["image_id"], []).append(
            " ".join([str(class_id), *(f"{v:.6f}" for v in converted)])
        )

    return lines


def download(url: str, destination: Path) -> Path:
    if destination.is_file():
        print(f"  cached {destination.name}")
        return destination

    print(f"  downloading {url}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    urllib.request.urlretrieve(url, partial)
    # Only becomes the real filename once complete, so an interrupted run
    # cannot leave a truncated zip that looks cached.
    partial.rename(destination)
    return destination


def unzip(archive: Path, destination: Path, marker: Path) -> None:
    if marker.exists():
        print(f"  already extracted {archive.name}")
        return

    print(f"  extracting {archive.name}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(destination)


def build(args) -> None:
    work = Path(args.work)
    output = Path(args.output)
    work.mkdir(parents=True, exist_ok=True)

    print("COCO val2017")
    images_zip = download(IMAGES_URL, work / "val2017.zip")
    annotations_zip = download(ANNOTATIONS_URL, work / "annotations.zip")
    unzip(images_zip, work, work / "val2017")
    unzip(annotations_zip, work, work / "annotations")

    annotations = work / "annotations" / "instances_val2017.json"
    if not annotations.is_file():
        sys.exit(f"Missing {annotations}")

    print("  reading annotations")
    data = json.loads(annotations.read_text())
    wanted = our_class_ids()
    lines = group_annotations(data, wanted)
    names = {image["id"]: image["file_name"] for image in data["images"]}

    image_ids = sorted(lines)
    random.Random(args.seed).shuffle(image_ids)
    if args.max_images:
        image_ids = image_ids[: args.max_images]

    cut = int(len(image_ids) * (1 - args.val_fraction))
    splits = {"train": image_ids[:cut], "val": image_ids[cut:]}

    shutil.rmtree(output, ignore_errors=True)
    totals = {}

    for split, ids in splits.items():
        image_dir = output / "images" / split
        label_dir = output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        counts: Counter = Counter()
        for image_id in ids:
            source = work / "val2017" / names[image_id]
            if not source.is_file():
                continue
            shutil.copy2(source, image_dir / source.name)
            (label_dir / f"{source.stem}.txt").write_text(
                "\n".join(lines[image_id]) + "\n"
            )
            counts.update(int(line.split()[0]) for line in lines[image_id])

        totals[split] = counts
        print(f"  {split}: {len(ids)} images, {sum(counts.values())} instances")

    from app import config

    print()
    for split, counts in totals.items():
        readable = {config.CLASS_NAMES[i]: n for i, n in sorted(counts.items())}
        print(f"{split}: {readable}")
    print(f"\nwrote {output}")
    print(f"use it with:  --source {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/content/datasets/coco-ground")
    parser.add_argument(
        "--work", default="/content/datasets/coco-raw", help="download/extract scratch"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=3000,
        help="cap on images kept (default 3000; 0 for all)",
    )
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
