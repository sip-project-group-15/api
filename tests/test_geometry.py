import pytest

from app import config, geometry


def box(x, y, w, h):
    return [x, y, w, h]


def test_foot_point_is_the_bottom_centre():
    """Positions must be ground contact, not centroid, or oblique views break."""
    assert geometry.foot_point(box(100, 50, 40, 80)) == (120, 130)


def test_body_length_uses_the_longest_side():
    """From above the animal's long axis can lie along either image axis."""
    assert geometry.body_length_pixels(box(0, 0, 30, 90)) == 90
    assert geometry.body_length_pixels(box(0, 0, 90, 30)) == 90


def test_separation_is_measured_in_body_lengths():
    rhino = box(0, 0, 100, 40)  # ruler = 100px, foot point (50, 40)
    person = box(240, 20, 20, 20)  # foot point (250, 40)

    assert geometry.separation(person, rhino) == 2.0


def test_separation_is_scale_invariant():
    """The whole point: the same scene at two zoom levels must read the same.

    A ground camera and a drone produce wildly different pixel distances for
    identical real-world geometry, and no calibration is available to reconcile
    them — using the rhino as the ruler is what makes one threshold work for
    both.
    """
    close_up = geometry.separation(box(240, 20, 20, 20), box(0, 0, 100, 40))
    far_away = geometry.separation(box(120, 10, 10, 10), box(0, 0, 50, 20))

    assert close_up == far_away


def test_separation_refuses_an_unreliable_ruler():
    """A rhino a few pixels across cannot measure anything trustworthy."""
    tiny_rhino = box(0, 0, config.MIN_RULER_PIXELS - 1, 4)

    assert geometry.separation(box(50, 50, 10, 10), tiny_rhino) is None


def test_nearest_picks_the_closest_rhino():
    person = box(0, 0, 10, 10)
    far = box(500, 500, 100, 100)
    near = box(40, 40, 100, 100)

    index, gap = geometry.nearest(person, [far, near])

    assert index == 1
    assert gap == pytest.approx(geometry.separation(person, near))
    assert gap < geometry.separation(person, far)


def test_nearest_is_none_when_nothing_is_measurable():
    unmeasurable = box(0, 0, 2, 2)

    assert geometry.nearest(box(50, 50, 10, 10), [unmeasurable]) is None
    assert geometry.nearest(box(50, 50, 10, 10), []) is None


def test_proximity_bands_decrease_with_distance():
    critical, _ = geometry.proximity(1.0)
    high, _ = geometry.proximity(3.0)
    medium, _ = geometry.proximity(8.0)
    far, band = geometry.proximity(50.0)

    assert critical > high > medium > far == 0.0
    assert band is None


def test_proximity_names_the_band():
    assert geometry.proximity(1.0)[1] == "critical"
    assert geometry.proximity(3.0)[1] == "high"
    assert geometry.proximity(8.0)[1] == "medium"


def test_unmeasurable_proximity_is_a_baseline_not_zero():
    """No rhino in frame is an unknown, not an all-clear.

    Scoring it zero would mean a person deep inside a reserve reads exactly the
    same as an empty frame.
    """
    score, band = geometry.proximity(None)

    assert score == config.UNKNOWN_PROXIMITY
    assert 0 < score < 1
    assert band is None


def test_estimated_metres_converts_for_humans():
    assert geometry.estimated_metres(2.0) == round(2 * config.RHINO_BODY_LENGTH_M, 1)
