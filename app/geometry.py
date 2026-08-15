"""Scale-invariant frame geometry.

Pixel distance on its own is meaningless: 80px is tens of metres from a drone
at altitude and a step or two from a ground camera. Calibrating every camera is
not realistic, so every distance here is expressed in *rhino body-lengths*
instead — the rhino is already in the frame, an adult is roughly 3.7m nose to
tail, so it is a ruler that travels with the footage and needs no calibration.

Two choices make one formula work for both aerial and ground-level views:

* Positions are the *foot point* (bottom-centre of the box), not the centroid.
  The bottom edge is roughly where the subject meets the ground, which is a far
  better proxy for ground position in an oblique view, where a centroid mostly
  encodes how tall something is.
* The ruler is the rhino box's longest side, so it holds up whichever way the
  animal is facing in a top-down shot.

The estimate is coarse — a head-on rhino measures short, so expect ±40% — which
is why callers band the result rather than quoting metres as fact.
"""

from app import config


def foot_point(box: list[int]) -> tuple[float, float]:
    """Bottom-centre of an ``[x, y, w, h]`` box: its contact point with the ground."""
    x, y, w, h = box
    return x + w / 2, y + h


def body_length_pixels(rhino_box: list[int]) -> float:
    """One rhino body-length, in pixels.

    The longest side rather than the width: from directly above, the animal's
    long axis can lie along either image axis.
    """
    return float(max(rhino_box[2], rhino_box[3]))


def separation(threat_box: list[int], rhino_box: list[int]) -> float | None:
    """Foot-point distance between two boxes, in rhino body-lengths.

    Returns None when the rhino is too small on screen to be a trustworthy
    ruler — at that size a few pixels of box jitter would swing the estimate by
    hundreds of percent, and no reading beats a confidently wrong one.
    """
    ruler = body_length_pixels(rhino_box)
    if ruler < config.MIN_RULER_PIXELS:
        return None

    threat_x, threat_y = foot_point(threat_box)
    rhino_x, rhino_y = foot_point(rhino_box)

    return ((threat_x - rhino_x) ** 2 + (threat_y - rhino_y) ** 2) ** 0.5 / ruler


def nearest(threat_box: list[int], rhino_boxes: list[list[int]]) -> tuple[int, float] | None:
    """Index of the closest rhino and its separation, or None if none is measurable."""
    measured = [
        (index, gap)
        for index, box in enumerate(rhino_boxes)
        if (gap := separation(threat_box, box)) is not None
    ]
    if not measured:
        return None

    return min(measured, key=lambda pair: pair[1])


def proximity(gap: float | None) -> tuple[float, str | None]:
    """Score a separation into a 0-1 component and a named band.

    Bands rather than a smooth curve, because the underlying estimate is only
    good to roughly ±40% — a continuous score would imply a precision the
    measurement does not have, and bands are what a ranger can be told.

    A gap of None means no rhino was measurable in frame. That is not the same
    as "far away": a human deep inside a protected area is mildly suspicious on
    its own, so it scores a neutral baseline rather than zero.
    """
    if gap is None:
        return config.UNKNOWN_PROXIMITY, None

    for limit, score, band in config.PROXIMITY_BANDS:
        if gap < limit:
            return score, band

    return 0.0, None


def estimated_metres(gap: float) -> float:
    """Body-lengths back to metres, for human-readable alert text only."""
    return round(gap * config.RHINO_BODY_LENGTH_M, 1)
