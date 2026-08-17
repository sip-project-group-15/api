"""Turns tracked detections into an explainable threat score.

Poaching is not an object, so it is not something YOLO can be asked to detect.
It is a relationship between objects over time — a person closing on a rhino —
and that judgement belongs here, above the detector, where it can be read,
argued with and tuned without retraining anything.

The score is a plain weighted sum of four components rather than a learned
function, for the same reason: a ranger acting on an alert needs "a person
closed from 22m to 6m over eight seconds, vehicle present", not a float. Every
component is reported alongside the total so an alert always carries its
reasoning.

    proximity    how close the nearest threat is, in rhino body-lengths
    approach     whether that distance is shrinking, and how fast
    context      weapons, vehicles, group size — aggravating factors
    persistence  how many consecutive analysed frames this has held for

Weights live in config.PROXIMITY_WEIGHT and friends, and sum to 1.0.
"""

from collections import deque

from app import config, geometry
from app.tracking import CentroidTracker, Track


def closing_rate(history: deque[tuple[float, float]]) -> float | None:
    """Rate at which a pair's separation is shrinking, in body-lengths/second.

    Least-squares slope over the whole window rather than a first-to-last
    difference, so one noisy box does not read as a charge. Positive means
    closing; None means not enough history to say yet.
    """
    if len(history) < config.MIN_APPROACH_SAMPLES:
        return None

    times = [t for t, _ in history]
    gaps = [gap for _, gap in history]
    mean_time = sum(times) / len(times)
    variance = sum((t - mean_time) ** 2 for t in times)

    if variance <= 0:
        return None

    mean_gap = sum(gaps) / len(gaps)
    slope = sum((t - mean_time) * (gap - mean_gap) for t, gap in history) / variance

    # Slope is change in separation; closing is the negative of it.
    return -slope


def approach(rate: float | None) -> float:
    """Score a closing rate 0-1. Retreating and standing still both score zero."""
    if rate is None or rate <= 0:
        return 0.0

    return min(1.0, rate / config.APPROACH_SATURATION)


def context(detections: list[dict]) -> tuple[float, list[str]]:
    """Aggravating factors present in the frame, and their labels.

    Deliberately additive and capped rather than multiplicative: a weapon alone
    should be able to max this component out, but nothing here can raise an
    alert by itself, because weapon recall in particular is expected to be poor
    at aerial resolution.
    """
    labels = [detection["label"] for detection in detections]
    score = 0.0
    reasons = []

    for label, weight in config.CONTEXT_WEIGHTS.items():
        if label in labels:
            score += weight
            reasons.append(f"{label} present")

    people = labels.count("person")
    if people >= config.GROUP_SIZE:
        score += config.GROUP_WEIGHT
        reasons.append(f"group of {people}")

    return min(1.0, score), reasons


def persistence(streak: int) -> float:
    """Score how long the situation has held, to damp single-frame flukes."""
    return min(1.0, streak / config.PERSISTENCE_SATURATION)


def combine(
    proximity_score: float,
    approach_score: float,
    context_score: float,
    persistence_score: float,
    approach_measured: bool,
    distance_known: bool = True,
) -> float:
    """Weighted sum over the terms that could actually be measured.

    A term that is zero because it *was measured* as zero is evidence: we
    watched, and nobody moved closer. A term that is zero because it could not
    be measured is not evidence of anything, and letting it drag the total down
    scores uncertainty as innocence.

    That distinction matters most for a still image, where approach is
    structurally unmeasurable. Counting its 0.30 weight as a zero caps any
    single frame at 0.625 — a photo could never be `critical` however damning
    it was — and meant a still only cleared the threshold with a threat inside
    two body-lengths, leaving everything from 7m to 45m unreachable.

    So an unmeasurable term is dropped and its weight redistributed over the
    rest. A person 13m from a rhino scores 0.52 from one frame instead of 0.36,
    while a person filmed for eight frames and measured as stationary still
    scores the lower number — which is the right way round.

    Redistribution requires a real distance to redistribute *onto*. With no
    rhino in frame, approach can never be measured however long the clip runs,
    and proximity is already only a baseline guess; shifting approach's weight
    onto that guess compounds one assumption with another and inflates every
    frame of, say, tourists beside a vehicle. So when the distance is unknown
    the full weighting stands and approach simply contributes nothing.
    """
    terms = [
        (config.PROXIMITY_WEIGHT, proximity_score),
        (config.CONTEXT_WEIGHT, context_score),
        (config.PERSISTENCE_WEIGHT, persistence_score),
    ]
    if approach_measured or not distance_known:
        terms.append((config.APPROACH_WEIGHT, approach_score))

    total = sum(weight for weight, _ in terms)

    return sum(weight * value for weight, value in terms) / total


def severity(score: float) -> str:
    for limit, name in config.SEVERITY_BANDS:
        if score >= limit:
            return name

    return "low"


def describe(
    label: str, gap: float | None, band: str | None, rate: float | None, reasons: list[str]
) -> str:
    """One human-readable sentence, carried on the alert and into the SMS."""
    if gap is None:
        parts = [f"{label} detected, no rhino visible to measure against"]
    else:
        parts = [
            f"{label} {gap:.1f} rhino body-lengths from a rhino "
            f"(~{geometry.estimated_metres(gap)}m, {band})"
        ]

    if rate is not None and rate > 0:
        parts.append(f"closing at {rate:.2f} body-lengths/s")

    parts.extend(reasons)

    return "; ".join(parts)


class ThreatMonitor:
    """Stateful per-video assessor. One instance per video, fed frame by frame.

    Owns the tracker and the per-pair distance histories that `approach`
    needs. Everything it depends on above is a pure function, so thresholds can
    be tuned against synthetic boxes without a model, a video or a GPU.
    """

    def __init__(self) -> None:
        self._tracker = CentroidTracker()
        self._pairs: dict[tuple[int, int], deque[tuple[float, float]]] = {}
        self._streak = 0

    def _history(self, threat: Track, rhino: Track) -> deque[tuple[float, float]]:
        key = (threat.track_id, rhino.track_id)
        if key not in self._pairs:
            self._pairs[key] = deque(maxlen=config.TRACK_HISTORY)
        return self._pairs[key]

    def update(self, detections: list[dict], timestamp: float) -> dict | None:
        """Assess one analysed frame. Returns None when nothing threatening is present.

        A rhino on its own is wildlife, not poaching, so it never produces a
        result — it only ever raises the score of a threat that is already
        there.
        """
        tracks = self._tracker.update(detections, timestamp)
        threats = [track for track in tracks if track.label in config.THREAT_CLASSES]
        rhinos = [track for track in tracks if track.label in config.ASSET_CLASSES]

        if not threats:
            self._streak = 0
            return None

        self._streak += 1
        context_score, reasons = context(detections)
        persistence_score = persistence(self._streak)

        worst = None

        for threat in threats:
            gap, rate = None, None
            nearest = geometry.nearest(threat.box, [rhino.box for rhino in rhinos])

            if nearest is not None:
                index, gap = nearest
                history = self._history(threat, rhinos[index])
                history.append((timestamp, gap))
                rate = closing_rate(history)

            proximity_score, band = geometry.proximity(gap)
            approach_score = approach(rate)
            score = combine(
                proximity_score,
                approach_score,
                context_score,
                persistence_score,
                approach_measured=rate is not None,
                distance_known=gap is not None,
            )

            candidate = {
                "score": round(score, 4),
                "severity": severity(score),
                "label": threat.label,
                "track_id": threat.track_id,
                "detection_confidence": threat.confidence,
                "separation_body_lengths": None if gap is None else round(gap, 2),
                "separation_metres_estimate": (
                    None if gap is None else geometry.estimated_metres(gap)
                ),
                "proximity_band": band,
                "closing_rate": None if rate is None else round(rate, 3),
                # False means approach could not be measured yet and its weight
                # was redistributed — not that the subject was standing still.
                # Anything displaying the components has to tell those apart.
                "approach_measured": rate is not None,
                "distance_known": gap is not None,
                "components": {
                    "proximity": round(proximity_score, 3),
                    "approach": round(approach_score, 3),
                    "context": round(context_score, 3),
                    "persistence": round(persistence_score, 3),
                },
                "context_factors": reasons,
                "frame_streak": self._streak,
                "reason": describe(threat.label, gap, band, rate, reasons),
            }

            if worst is None or candidate["score"] > worst["score"]:
                worst = candidate

        return worst
