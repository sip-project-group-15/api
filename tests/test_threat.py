"""Scenario tests for the scoring layer.

These need no model, no video and no GPU — the components are pure functions
over boxes, which is the point of keeping the judgement out of the detector.
Thresholds can be tuned against these in seconds.
"""

from collections import deque

import pytest

from app import config, threat

# Boxes are kept to life-size proportions against the 100px rhino ruler (3.7m),
# so a person is ~48px tall rather than a toy square — box size sets how far
# the tracker will let something move between frames, and an undersized person
# would make these scenarios pass or fail for the wrong reason.
RHINO = {"label": "rhino", "confidence": 0.95, "box": [0, 20, 100, 40]}
PERSON_W, PERSON_H = 14, 48  # ~0.5m by ~1.8m at this scale

# Both foot points sit on y=60, so separation is the horizontal gap alone and
# person_at(n) is exactly n body-lengths from the rhino.
GROUND_Y = 60


def person_at(body_lengths, confidence=0.9):
    x = int(50 + body_lengths * 100) - PERSON_W // 2
    return {
        "label": "person",
        "confidence": confidence,
        "box": [x, GROUND_Y - PERSON_H, PERSON_W, PERSON_H],
    }


# Half a metre per analysed frame at 2fps is a brisk walk; a full body-length
# would be 7.4m in half a second, which is a vehicle, not a person.
APPROACHING = (5.0, 4.5, 4.0, 3.5, 3.0)
RETREATING = tuple(reversed(APPROACHING))


def run(frames, interval=0.5):
    """Feed a monitor a sequence of frames; return the last assessment."""
    monitor = threat.ThreatMonitor()
    assessment = None
    for index, detections in enumerate(frames):
        assessment = monitor.update(detections, index * interval)
    return assessment


# ── The core rule ────────────────────────────────────────────────────────────


def test_a_rhino_alone_is_never_an_alert():
    """Wildlife presence is not poaching. This is the rule the project rests on."""
    assert run([[RHINO]] * 5) is None


def test_an_empty_frame_is_never_an_alert():
    assert run([[], [], []]) is None


def test_a_person_beside_a_rhino_outscores_one_far_away():
    near = run([[RHINO, person_at(1.0)]] * 4)
    far = run([[RHINO, person_at(9.0)]] * 4)

    assert near["score"] > far["score"]
    assert near["proximity_band"] == "critical"


def test_distance_is_reported_in_both_units():
    assessment = run([[RHINO, person_at(2.0)]])

    assert assessment["separation_body_lengths"] == pytest.approx(2.0)
    assert assessment["separation_metres_estimate"] == pytest.approx(
        2.0 * config.RHINO_BODY_LENGTH_M, abs=0.1
    )


# ── Approach ─────────────────────────────────────────────────────────────────


def test_approaching_outscores_standing_still_at_the_same_distance():
    """The measure the project is really after: closing, not just being near."""
    approaching = run([[RHINO, person_at(gap)] for gap in APPROACHING])
    static = run([[RHINO, person_at(APPROACHING[-1])]] * len(APPROACHING))

    assert approaching["closing_rate"] > 0
    assert approaching["score"] > static["score"]
    assert static["components"]["approach"] == 0.0


def test_retreating_scores_no_approach():
    retreating = run([[RHINO, person_at(gap)] for gap in RETREATING])

    assert retreating["closing_rate"] < 0
    assert retreating["components"]["approach"] == 0.0


def test_closing_rate_needs_enough_history():
    """Two points are a line through noise, not a trend."""
    early = run([[RHINO, person_at(5.0)], [RHINO, person_at(4.5)]])

    assert early["closing_rate"] is None


def test_closing_rate_is_in_body_lengths_per_second():
    # Two body-lengths per half-second step = 4.0/s.
    history = deque([(0.0, 8.0), (0.5, 6.0), (1.0, 4.0)])

    assert threat.closing_rate(history) == pytest.approx(4.0)


def test_closing_rate_ignores_a_single_noisy_box():
    """Least-squares over the window, not first-minus-last."""
    steady = deque([(0.0, 8.0), (0.5, 6.0), (1.0, 4.0), (1.5, 2.0)])
    with_blip = deque([(0.0, 8.0), (0.5, 6.0), (1.0, 4.0), (1.5, 2.5)])

    assert threat.closing_rate(with_blip) == pytest.approx(
        threat.closing_rate(steady), rel=0.2
    )


def test_approach_component_saturates():
    assert threat.approach(config.APPROACH_SATURATION * 10) == 1.0
    assert threat.approach(None) == 0.0
    assert threat.approach(-1.0) == 0.0


# ── Context ──────────────────────────────────────────────────────────────────


def test_a_weapon_escalates_but_cannot_alert_alone():
    """Weapon recall will be poor from the air, so it must never be a gate."""
    unarmed = run([[RHINO, person_at(4.0)]] * 3)
    armed = run(
        [[RHINO, person_at(4.0), {"label": "weapon", "confidence": 0.5, "box": [0, 0, 5, 5]}]] * 3
    )

    assert armed["score"] > unarmed["score"]
    # Context alone maxes out at its weight, well under any sensible threshold.
    assert config.CONTEXT_WEIGHT < config.DEFAULT_ALERT_THRESHOLD


def test_a_vehicle_escalates():
    vehicle = {"label": "vehicle", "confidence": 0.8, "box": [900, 0, 80, 40]}

    with_vehicle = run([[RHINO, person_at(4.0), vehicle]] * 3)
    without = run([[RHINO, person_at(4.0)]] * 3)

    assert with_vehicle["components"]["context"] > without["components"]["context"]


def test_a_group_escalates():
    crowd = [RHINO] + [person_at(4.0 + index) for index in range(config.GROUP_SIZE)]

    assessment = run([crowd] * 3)

    assert any("group of" in reason for reason in assessment["context_factors"])


# ── Persistence ──────────────────────────────────────────────────────────────


def test_persistence_grows_with_sustained_presence():
    brief = run([[RHINO, person_at(3.0)]])
    sustained = run([[RHINO, person_at(3.0)]] * config.PERSISTENCE_SATURATION)

    assert sustained["components"]["persistence"] > brief["components"]["persistence"]


def test_the_streak_resets_when_the_threat_leaves():
    monitor = threat.ThreatMonitor()
    for index in range(4):
        monitor.update([RHINO, person_at(3.0)], index * 0.5)
    for index in range(4, 8):
        monitor.update([RHINO], index * 0.5)

    resumed = monitor.update([RHINO, person_at(3.0)], 4.0)

    assert resumed["frame_streak"] == 1


# ── No rhino in frame ────────────────────────────────────────────────────────


def test_a_person_with_no_rhino_visible_scores_low_but_not_zero():
    assessment = run([[person_at(0)]] * 4)

    assert assessment is not None
    assert assessment["separation_body_lengths"] is None
    assert 0 < assessment["score"] < config.DEFAULT_ALERT_THRESHOLD


# ── Reporting ────────────────────────────────────────────────────────────────


def test_the_worst_threat_in_frame_is_the_one_reported():
    assessment = run([[RHINO, person_at(9.0), person_at(1.0)]] * 3)

    assert assessment["separation_body_lengths"] == pytest.approx(1.0)


def test_components_sum_to_the_score():
    """The score must stay a readable weighted sum, not a black box."""
    assessment = run([[RHINO, person_at(gap)] for gap in APPROACHING])
    components = assessment["components"]

    expected = (
        config.PROXIMITY_WEIGHT * components["proximity"]
        + config.APPROACH_WEIGHT * components["approach"]
        + config.CONTEXT_WEIGHT * components["context"]
        + config.PERSISTENCE_WEIGHT * components["persistence"]
    )

    assert assessment["score"] == pytest.approx(expected, abs=0.01)


def test_the_reason_is_human_readable():
    """A ranger acts on the sentence, not the float."""
    assessment = run([[RHINO, person_at(gap)] for gap in APPROACHING])

    assert "person" in assessment["reason"]
    assert "body-lengths from a rhino" in assessment["reason"]
    assert "closing at" in assessment["reason"]


def test_severity_rises_with_the_score():
    assert threat.severity(0.9) == "critical"
    assert threat.severity(0.6) == "high"
    assert threat.severity(0.3) == "medium"
    assert threat.severity(0.1) == "low"


def test_a_clear_poaching_scenario_clears_the_default_threshold():
    """End to end: an armed pair approaching a rhino by vehicle must alert."""
    vehicle = {"label": "vehicle", "confidence": 0.8, "box": [1200, 0, 80, 40]}
    weapon = {"label": "weapon", "confidence": 0.4, "box": [0, 0, 8, 8]}

    assessment = run(
        [[RHINO, person_at(gap), weapon, vehicle] for gap in APPROACHING]
    )

    assert assessment["score"] >= config.DEFAULT_ALERT_THRESHOLD
    assert assessment["severity"] in {"high", "critical"}


def test_a_distant_bystander_does_not_alert():
    """The false positive that matters: tourists and rangers exist."""
    assessment = run([[RHINO, person_at(15.0)]] * 6)

    assert assessment["score"] < config.DEFAULT_ALERT_THRESHOLD
