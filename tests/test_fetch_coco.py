"""Tests for the COCO ground-level fetcher.

The conversion is the risky part: COCO stores absolute corner-and-size boxes,
YOLO wants normalised centre form, and getting that wrong produces labels that
are silently plausible — training completes, metrics look odd, and nothing
points at the cause.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

import fetch_coco


def coco(images, annotations, categories=None):
    return {
        "images": images,
        "annotations": annotations,
        "categories": categories
        or [
            {"id": 1, "name": "person"},
            {"id": 3, "name": "car"},
            {"id": 8, "name": "truck"},
            {"id": 88, "name": "teddy bear"},
        ],
    }


# ── Box conversion ───────────────────────────────────────────────────────────


def test_box_converts_to_normalised_centre_form():
    # A 100x50 box at (200, 100) in a 1000x500 image.
    assert fetch_coco.to_yolo([200, 100, 100, 50], 1000, 500) == pytest.approx(
        (0.25, 0.25, 0.1, 0.1)
    )


def test_a_full_frame_box_is_centred_and_full_size():
    assert fetch_coco.to_yolo([0, 0, 640, 480], 640, 480) == pytest.approx(
        (0.5, 0.5, 1.0, 1.0)
    )


def test_a_box_overflowing_the_frame_is_clipped():
    """COCO contains boxes that extend past the image edge.

    Clipping must move the centre, not just trim the size: this box spans
    50..150 on both axes, so unclipped it would report a centre at the very
    corner of the image with full width, which is a target that does not
    describe anything real.
    """
    cx, cy, w, h = fetch_coco.to_yolo([50, 50, 100, 100], 100, 100)

    assert (cx, cy) == pytest.approx((0.75, 0.75))
    assert (w, h) == pytest.approx((0.5, 0.5))


def test_clipping_leaves_a_fully_enclosed_box_untouched():
    assert fetch_coco.to_yolo([25, 25, 50, 50], 100, 100) == pytest.approx(
        (0.5, 0.5, 0.5, 0.5)
    )


def test_a_zero_area_box_is_rejected():
    """A degenerate box is not a training target, it is noise."""
    assert fetch_coco.to_yolo([10, 10, 0, 50], 100, 100) is None
    assert fetch_coco.to_yolo([10, 10, 50, 0], 100, 100) is None


def test_a_box_entirely_outside_the_frame_is_rejected():
    assert fetch_coco.to_yolo([200, 200, 50, 50], 100, 100) is None


def test_a_missing_image_size_is_rejected():
    """Unknown dimensions would divide by zero rather than fail loudly."""
    assert fetch_coco.to_yolo([10, 10, 50, 50], 0, 0) is None


# ── Grouping and class mapping ───────────────────────────────────────────────

WANTED = {"person": 1, "car": 2, "truck": 2}


def test_classes_are_collapsed_onto_ours():
    """car and truck are distinct in COCO but both are `vehicle` to us."""
    data = coco(
        [{"id": 7, "width": 100, "height": 100, "file_name": "a.jpg"}],
        [
            {"image_id": 7, "category_id": 1, "bbox": [10, 10, 20, 20]},
            {"image_id": 7, "category_id": 3, "bbox": [40, 40, 20, 20]},
            {"image_id": 7, "category_id": 8, "bbox": [70, 70, 20, 20]},
        ],
    )

    ids = [int(line.split()[0]) for line in fetch_coco.group_annotations(data, WANTED)[7]]

    assert sorted(ids) == [1, 2, 2]


def test_unwanted_categories_are_dropped():
    data = coco(
        [{"id": 1, "width": 100, "height": 100, "file_name": "a.jpg"}],
        [{"image_id": 1, "category_id": 88, "bbox": [10, 10, 20, 20]}],
    )

    assert fetch_coco.group_annotations(data, WANTED) == {}


def test_crowd_regions_are_dropped():
    """One box drawn over many overlapping people would train a giant `person`."""
    data = coco(
        [{"id": 1, "width": 100, "height": 100, "file_name": "a.jpg"}],
        [
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 100, 100], "iscrowd": 1},
            {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "iscrowd": 0},
        ],
    )

    assert len(fetch_coco.group_annotations(data, WANTED)[1]) == 1


def test_images_with_nothing_we_want_are_omitted():
    """They would become background images, and backgrounds are capped later."""
    data = coco(
        [
            {"id": 1, "width": 100, "height": 100, "file_name": "a.jpg"},
            {"id": 2, "width": 100, "height": 100, "file_name": "b.jpg"},
        ],
        [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]}],
    )

    assert set(fetch_coco.group_annotations(data, WANTED)) == {1}


def test_label_lines_are_valid_yolo():
    data = coco(
        [{"id": 1, "width": 200, "height": 100, "file_name": "a.jpg"}],
        [{"image_id": 1, "category_id": 1, "bbox": [50, 25, 100, 50]}],
    )

    line = fetch_coco.group_annotations(data, WANTED)[1][0]
    class_id, *coords = line.split()

    assert class_id == "1"
    assert len(coords) == 4
    assert all(0.0 <= float(c) <= 1.0 for c in coords)
    assert [float(c) for c in coords] == pytest.approx([0.5, 0.5, 0.5, 0.5])


def test_mapping_targets_exist_in_our_class_list():
    """A typo here would silently drop a whole class from the fetch."""
    from app import config

    assert set(fetch_coco.COCO_TO_OURS.values()) <= set(config.CLASS_NAMES)


def test_our_class_ids_match_config_positions():
    from app import config

    ids = fetch_coco.our_class_ids()

    assert ids["person"] == config.CLASS_NAMES.index("person")
    assert ids["car"] == config.CLASS_NAMES.index("vehicle")
    assert ids["truck"] == ids["bus"] == ids["car"]


def test_every_mapped_class_is_a_class_the_scorer_acts_on():
    """Fetching a class the scorer ignores would be wasted training capacity."""
    from app import config

    targets = set(fetch_coco.COCO_TO_OURS.values())

    assert targets <= config.THREAT_CLASSES | config.ASSET_CLASSES
