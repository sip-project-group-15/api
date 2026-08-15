"""Tests for the dataset merger.

Balancing is the part worth testing: an unbalanced merge does not fail, it
trains a model that ignores the rare class while reporting a healthy-looking
mAP, which costs a GPU hour before anyone notices.
"""

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

import build_dataset


def candidate(name, class_ids):
    """One (image, label lines, counts) triple as select() expects."""
    lines = [f"{class_id} 0.5 0.5 0.1 0.1" for class_id in class_ids]
    return Path(f"{name}.jpg"), lines, Counter(class_ids)


# ── Label plumbing ───────────────────────────────────────────────────────────


def test_label_path_mirrors_the_image_path():
    assert build_dataset.label_for(
        Path("/data/VisDrone/images/train/0001.jpg")
    ) == Path("/data/VisDrone/labels/train/0001.txt")


def test_label_path_swaps_only_the_last_images_component():
    """A source rooted at a directory literally called "images" must survive."""
    assert build_dataset.label_for(
        Path("/images/set/images/val/a.png")
    ) == Path("/images/set/labels/val/a.txt")


def test_source_spec_splits_into_dataset_and_remap():
    source, remap = build_dataset.parse_source("VisDrone.yaml:pedestrian=1,car=2")

    assert source == "VisDrone.yaml"
    assert remap == {"pedestrian": 1, "car": 2}


def test_source_spec_without_a_remap():
    assert build_dataset.parse_source("/content/custom") == ("/content/custom", {})


def test_remap_is_by_name_not_position():
    """Source ids are arbitrary; a positional remap would silently mislabel."""
    names = {0: "buffalo", 1: "elephant", 2: "rhino", 3: "zebra"}

    table = build_dataset.remap_table(names, {"rhino": 0})

    assert table == {2: 0}


def test_remap_reports_a_name_the_source_lacks(capsys):
    """A typo here would drop a whole class without any error."""
    table = build_dataset.remap_table({0: "car"}, {"rhinoceros": 0})

    assert table == {}
    assert "not a class in this source" in capsys.readouterr().out


def test_unmapped_classes_are_dropped(tmp_path):
    label = tmp_path / "a.txt"
    label.write_text("2 0.5 0.5 0.1 0.1\n0 0.1 0.1 0.2 0.2\n")

    lines = build_dataset.read_boxes(label, {2: 0})

    assert lines == ["0 0.5 0.5 0.1 0.1"]


def test_identity_remap_keeps_every_class(tmp_path):
    """Custom data is already in our id space, so nothing should be dropped."""
    label = tmp_path / "a.txt"
    label.write_text("0 0.5 0.5 0.1 0.1\n2 0.1 0.1 0.2 0.2\n")

    assert len(build_dataset.read_boxes(label, {})) == 2


def test_a_missing_label_file_is_a_background(tmp_path):
    assert build_dataset.read_boxes(tmp_path / "nothing.txt", {}) == []


# ── Balancing ────────────────────────────────────────────────────────────────


def test_a_dense_source_cannot_swamp_a_sparse_one():
    """The reason this script exists.

    VisDrone frames carry hundreds of vehicles each. Merged unrestricted
    against a few hundred rhinos it yields a ~1000:1 imbalance and the detector
    stops predicting rhino entirely.
    """
    crowded = [candidate(f"drone{i}", [2] * 200) for i in range(100)]

    chosen = build_dataset.select(crowded, {2: 1000}, 0.0, seed=0)

    kept = sum(len(lines) for _, lines in chosen)
    assert kept < 2000  # not the 20,000 available
    assert len(chosen) < 20


def test_selection_stops_once_every_cap_is_met():
    plenty = [candidate(f"x{i}", [0]) for i in range(500)]

    chosen = build_dataset.select(plenty, {0: 50}, 0.0, seed=0)

    assert len(chosen) == 50


def test_an_image_is_kept_for_a_class_that_still_has_room():
    """Mixed images must not be rejected because one of their classes is full."""
    items = [candidate("a", [0] * 100), candidate("b", [0, 1])]

    chosen = build_dataset.select(items, {0: 10, 1: 10}, 0.0, seed=0)

    assert len(chosen) == 2


def test_backgrounds_are_capped_not_excluded():
    """A few teach what an empty scene is; a majority teach predicting nothing."""
    items = [candidate(f"obj{i}", [0]) for i in range(10)]
    items += [candidate(f"bg{i}", []) for i in range(500)]

    chosen = build_dataset.select(items, {0: 100}, 0.2, seed=0)

    empty = sum(1 for _, lines in chosen if not lines)
    assert empty == 2
    assert len(chosen) == 12


def test_no_backgrounds_when_the_ratio_is_zero():
    items = [candidate("obj", [0]), candidate("bg", [])]

    chosen = build_dataset.select(items, {0: 10}, 0.0, seed=0)

    assert len(chosen) == 1


def test_selection_is_deterministic():
    """A run must be reproducible, or a metric change cannot be attributed."""
    items = [candidate(f"x{i}", [0]) for i in range(200)]

    first = build_dataset.select(items, {0: 20}, 0.0, seed=7)
    second = build_dataset.select(items, {0: 20}, 0.0, seed=7)

    assert [p for p, _ in first] == [p for p, _ in second]


def test_selection_samples_across_the_source_not_just_the_head():
    """Datasets are stored grouped by scene, so the first N are near-duplicates."""
    items = [candidate(f"x{i}", [0]) for i in range(200)]

    chosen = build_dataset.select(items, {0: 20}, 0.0, seed=0)
    picked = [int(path.stem[1:]) for path, _ in chosen]

    assert max(picked) > 100


# ── Reporting ────────────────────────────────────────────────────────────────

NAMES = {0: "rhino", 1: "person", 2: "vehicle", 3: "weapon"}


def test_no_threat_class_is_fatal(capsys):
    """The exact failure of the first training run: rhino-only data."""
    totals = {"train": Counter({0: 399}), "val": Counter({0: 85})}

    build_dataset.report(totals, NAMES)
    out = capsys.readouterr().out

    assert "FATAL" in out
    assert "cannot raise a single alert" in out


def test_no_rhino_is_fatal(capsys):
    """Without the asset there is no ruler, so no distance is measurable."""
    totals = {"train": Counter({1: 500, 2: 500}), "val": Counter()}

    build_dataset.report(totals, NAMES)
    out = capsys.readouterr().out

    assert "FATAL" in out
    assert "ruler" in out


def test_an_empty_weapon_class_is_a_note_not_a_warning(capsys):
    """Weapon has no usable public data; alarming about it every run would
    teach the reader to ignore the genuinely fatal messages above."""
    totals = {"train": Counter({0: 399, 1: 1500, 2: 1500}), "val": Counter()}

    build_dataset.report(totals, NAMES)
    out = capsys.readouterr().out

    assert "FATAL" not in out
    assert "no examples for weapon" in out
    assert "Balance looks usable" in out


def test_a_lopsided_split_is_called_out(capsys):
    totals = {"train": Counter({0: 100, 1: 5000, 2: 5000}), "val": Counter()}

    build_dataset.report(totals, NAMES)

    assert "Imbalance among trained classes is 50:1" in capsys.readouterr().out


def test_a_usable_split_is_reported_as_such(capsys):
    totals = {"train": Counter({0: 1000, 1: 1800, 2: 1500}), "val": Counter()}

    build_dataset.report(totals, NAMES)

    assert "Balance looks usable" in capsys.readouterr().out


# ── End to end ───────────────────────────────────────────────────────────────


def make_source(root: Path, split: str, images: dict[str, list[int]]) -> None:
    """Write a miniature YOLO dataset: {name: [class ids]}."""
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for name, class_ids in images.items():
        (image_dir / f"{name}.jpg").write_bytes(b"\xff\xd8\xff")  # jpeg magic
        (label_dir / f"{name}.txt").write_text(
            "".join(f"{c} 0.5 0.5 0.2 0.2\n" for c in class_ids)
        )


def test_two_sources_merge_into_one_balanced_dataset(tmp_path, capsys):
    wildlife = tmp_path / "wildlife"
    drone = tmp_path / "drone"
    make_source(wildlife, "train", {f"w{i}": [2] for i in range(40)})
    make_source(wildlife, "val", {f"wv{i}": [2] for i in range(10)})
    make_source(drone, "train", {f"d{i}": [0, 3] * 50 for i in range(40)})
    make_source(drone, "val", {f"dv{i}": [0, 3] for i in range(10)})

    args = pytest.importorskip("argparse").Namespace(
        source=[f"{wildlife}:rhino=0", f"{drone}:pedestrian=1,car=2"],
        output=str(tmp_path / "merged"),
        cap=100,
        background_ratio=0.0,
        seed=0,
        names=NAMES,
    )
    # Directory sources carry no names, so remaps cannot resolve and ids pass
    # through unchanged — which is exactly how custom data is meant to work.
    build_dataset.build(args)

    merged = tmp_path / "merged"
    assert (merged / "kifaru.yaml").is_file()

    train_labels = list((merged / "labels" / "train").glob("*.txt"))
    assert train_labels

    counts = Counter()
    for label in train_labels:
        counts.update(int(line.split()[0]) for line in label.read_text().splitlines())

    # The dense source is capped rather than allowed to dominate.
    assert counts[0] <= 100 * 3
    assert "class balance" in capsys.readouterr().out


def test_filenames_are_prefixed_so_sources_cannot_collide(tmp_path):
    """Two datasets both containing 0000001.jpg must not overwrite each other."""
    first = tmp_path / "alpha"
    second = tmp_path / "beta"
    make_source(first, "train", {"0000001": [0]})
    make_source(first, "val", {"0000002": [0]})
    make_source(second, "train", {"0000001": [1]})
    make_source(second, "val", {"0000003": [1]})

    import argparse

    build_dataset.build(
        argparse.Namespace(
            source=[str(first), str(second)],
            output=str(tmp_path / "merged"),
            cap=100,
            background_ratio=0.0,
            seed=0,
            names=NAMES,
        )
    )

    names = {p.name for p in (tmp_path / "merged" / "images" / "train").glob("*.jpg")}
    assert names == {"alpha_0000001.jpg", "beta_0000001.jpg"}
