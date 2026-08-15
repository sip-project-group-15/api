"""Merge several labelled datasets into one balanced YOLO training set.

The training set has to contain rhinos *and* the things that threaten them, but
no single public dataset has both. This stitches them together, remapping each
source's classes onto ours and capping how many instances of each class get in.

    python training/build_dataset.py \
        --source african-wildlife.yaml:rhino=0 \
        --source VisDrone.yaml:pedestrian=1,people=1,car=2,van=2,truck=2,bus=2,motor=2 \
        --output /content/datasets/kifaru-merged

Sources are either an Ultralytics dataset name, which downloads on demand, or a
directory laid out as images/{train,val} + labels/{train,val}. Classes are
remapped **by name** rather than by id, because source ids are arbitrary and
silently wrong remaps are the easiest way to poison a dataset.

Two caps do the real work:

*Instance caps* stop a dense source from drowning a sparse one. VisDrone frames
carry hundreds of tiny cars and pedestrians each; merged unrestricted against a
few hundred rhino boxes it produces a roughly 1000:1 imbalance, and a detector
trained on that quietly stops predicting the rare class while mAP still reads
respectably.

*Background ratio* caps images with no labels at all. A few teach the model
what an empty scene looks like and suppress false positives; mostly-empty
training sets teach it that predicting nothing is usually right.

The instance cap applies **per source, not globally**, and that is deliberate.
`person` needs to be learned from the air *and* from the ground — the first
trained model saw people only from VisDrone and so could not recognise one at
ground level at all. A global cap would let whichever source is processed first
consume the entire budget and starve the viewpoint that comes after it.
"""

import argparse
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val")


def label_for(image_path: Path) -> Path:
    """The YOLO label file beside an image: .../images/x.jpg -> .../labels/x.txt."""
    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def parse_source(spec: str) -> tuple[str, dict[str, int]]:
    """Split "dataset.yaml:name=id,name=id" into its source and its remap."""
    source, _, mapping = spec.rpartition(":")
    if "=" not in mapping:
        return spec, {}

    remap = {}
    for pair in mapping.split(","):
        name, _, target = pair.partition("=")
        if name.strip() and target.strip():
            remap[name.strip()] = int(target)
    return source, remap


def resolve(source: str) -> tuple[dict[str, list[Path]], dict[int, str]]:
    """Return image paths per split, and the source's own id -> name mapping."""
    if source.endswith((".yaml", ".yml")):
        from ultralytics.data.utils import check_det_dataset

        data = check_det_dataset(source)
        names = data.get("names", {})
        images = {}
        for split in SPLITS:
            entries = data.get(split) or []
            images[split] = collect(entries if isinstance(entries, list) else [entries])
        return images, names

    root = Path(source)
    if not root.is_dir():
        sys.exit(f"No such dataset source: {source}")
    return {split: collect([root / "images" / split]) for split in SPLITS}, {}


def collect(entries: list) -> list[Path]:
    """Expand directories and .txt manifests into a flat list of image paths."""
    images = []
    for entry in entries:
        path = Path(entry)
        if path.is_dir():
            images.extend(
                p for p in sorted(path.rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES
            )
        elif path.suffix == ".txt" and path.is_file():
            images.extend(
                Path(line.strip())
                for line in path.read_text().splitlines()
                if line.strip()
            )
    return images


def remap_table(source_names: dict[int, str], remap: dict[str, int]) -> dict[int, int]:
    """Turn a name-keyed remap into the id -> id table the label rewrite needs.

    Names that the source does not have are reported rather than ignored: a
    typo here would silently drop an entire class from the training set.
    """
    if not remap:
        return {}

    by_name = {str(name).lower(): int(index) for index, name in source_names.items()}
    table = {}
    for name, target in remap.items():
        source_id = by_name.get(name.lower())
        if source_id is None:
            print(f"  !! '{name}' is not a class in this source — ignored")
            continue
        table[source_id] = target

    return table


def read_boxes(label_path: Path, table: dict[int, int]) -> list[str]:
    """Label lines rewritten into our class space; unmapped classes dropped."""
    if not label_path.is_file():
        return []

    lines = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        source_id = int(float(parts[0]))
        target = table.get(source_id, source_id if not table else None)
        if target is not None:
            lines.append(" ".join([str(target), *parts[1:]]))
    return lines


def select(
    candidates: list[tuple[Path, list[str], Counter]],
    caps: dict[int, int],
    background_ratio: float,
    seed: int,
) -> list[tuple[Path, list[str]]]:
    """Pick images until every class hits its cap, then top up with backgrounds.

    Shuffled rather than taken in order, because datasets are usually stored
    grouped by scene or video, and the first N images of one are far less
    varied than N drawn from across it.
    """
    labelled = [item for item in candidates if item[2]]
    backgrounds = [item for item in candidates if not item[2]]

    random.Random(seed).shuffle(labelled)
    random.Random(seed + 1).shuffle(backgrounds)

    totals: Counter = Counter()
    chosen = []

    for image_path, lines, counts in labelled:
        # Keep the image if it carries any class that still has room. Its other
        # classes ride along and may overshoot their cap slightly, which is
        # fine — dropping their boxes would mean training on wrong labels.
        if any(totals[class_id] < caps.get(class_id, 0) for class_id in counts):
            chosen.append((image_path, lines))
            totals.update(counts)

    allowance = int(len(chosen) * background_ratio)
    chosen.extend((path, lines) for path, lines, _ in backgrounds[:allowance])

    return chosen


def build(args) -> None:
    output = Path(args.output)
    shutil.rmtree(output, ignore_errors=True)

    caps = {0: args.cap, 1: args.cap, 2: args.cap, 3: args.cap}
    totals = {split: Counter() for split in SPLITS}

    for spec in args.source:
        source, remap = parse_source(spec)
        print(f"\n{source}")

        images, source_names = resolve(source)
        table = remap_table(source_names, remap)
        if table:
            readable = {source_names[k]: v for k, v in table.items()}
            print(f"  remap: {readable}")

        for split in SPLITS:
            if not images[split]:
                print(f"  {split}: no images")
                continue

            candidates = []
            for image_path in images[split]:
                lines = read_boxes(label_for(image_path), table)
                counts = Counter(int(line.split()[0]) for line in lines)
                candidates.append((image_path, lines, counts))

            split_cap = caps if split == "train" else {
                class_id: max(40, cap // 5) for class_id, cap in caps.items()
            }
            chosen = select(candidates, split_cap, args.background_ratio, args.seed)

            image_dir = output / "images" / split
            label_dir = output / "labels" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)

            # Sources are prefixed because two datasets very often contain a
            # 0000001.jpg, and a collision would silently overwrite.
            prefix = Path(source).stem.replace(".", "_")

            for image_path, lines in chosen:
                stem = f"{prefix}_{image_path.stem}"
                shutil.copy2(image_path, image_dir / f"{stem}{image_path.suffix}")
                (label_dir / f"{stem}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else "")
                )
                totals[split].update(int(line.split()[0]) for line in lines)

            empty = sum(1 for _, lines in chosen if not lines)
            print(f"  {split}: {len(chosen)} images ({empty} background)")

    write_yaml(output, args.names)
    report(totals, args.names)


def write_yaml(output: Path, names: dict[int, str]) -> Path:
    import yaml

    path = output / "kifaru.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "path": str(output),
                "train": "images/train",
                "val": "images/val",
                "names": names,
            },
            sort_keys=False,
        )
    )
    print(f"\nwrote {path}")
    return path


def report(totals: dict[str, Counter], names: dict[int, str]) -> None:
    """Print the class balance, and say plainly when it is bad.

    This is the number to check before starting a long run: a detector trained
    on a lopsided set ignores the rare class while its headline mAP, dominated
    by the common one, still looks fine.
    """
    print(f"\n{'─' * 52}\nclass balance")

    for split in SPLITS:
        counts = totals[split]
        total = sum(counts.values())
        print(f"\n{split}: {total} instances")
        if not total:
            continue

        for class_id, name in sorted(names.items()):
            n = counts.get(class_id, 0)
            share = 100 * n / total
            flag = "  <-- no examples" if not n else ""
            print(f"  {name:<10} {n:>7}  ({share:>5.1f}%){flag}")

    from app import config

    train = totals["train"]
    counted = {name: train.get(class_id, 0) for class_id, name in names.items()}
    empty = [name for name, n in counted.items() if not n]
    present = [n for n in counted.values() if n]

    # Which classes are missing matters far more than how many. An empty
    # `weapon` is expected and harmless; an empty `person` is fatal, and saying
    # "no examples for X" about both trains the reader to ignore the warning.
    threats = [n for name, n in counted.items() if name in config.THREAT_CLASSES and n]
    assets = [n for name, n in counted.items() if name in config.ASSET_CLASSES and n]

    print()
    if not threats:
        print("!! FATAL: no threat class has any training data.")
        print("!! The scorer only ever alerts on " + "/".join(sorted(config.THREAT_CLASSES)))
        print("!! — a rhino alone is wildlife, not poaching — so this dataset")
        print("!! trains a model that cannot raise a single alert. Add a source")
        print("!! supplying them; see training/DATASETS.md.")
        return

    if not assets:
        print("!! FATAL: no " + "/".join(sorted(config.ASSET_CLASSES)) + " training data.")
        print("!! Distances are measured in rhino body-lengths using the rhino")
        print("!! in frame as the ruler, so without it nothing is measurable and")
        print("!! every frame falls back to an unknown-proximity baseline.")
        return

    if empty:
        print(f"Note: no examples for {', '.join(empty)} — declared but not learned.")
        print("      Expected for `weapon`; see training/DATASETS.md for why.")

    ratio = max(present) / min(present)
    if ratio > 10:
        print(f"!! Imbalance among trained classes is {ratio:.0f}:1 — the rarest")
        print("!! will likely be ignored while mAP still looks fine.")
        print("!! Lower --cap, or add more of the rare class.")
    else:
        print(f"Balance looks usable ({ratio:.1f}:1 across trained classes).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help='dataset, optionally with a remap: "VisDrone.yaml:pedestrian=1,car=2"',
    )
    parser.add_argument("--output", default="/content/datasets/kifaru-merged")
    parser.add_argument(
        "--cap",
        type=int,
        default=2000,
        help="max training instances per class, PER SOURCE (default 2000)",
    )
    parser.add_argument(
        "--background-ratio",
        type=float,
        default=0.15,
        help="unlabelled images as a fraction of labelled ones (default 0.15)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from app import config

    args.names = {index: name for index, name in enumerate(config.CLASS_NAMES)}
    build(args)


if __name__ == "__main__":
    main()
