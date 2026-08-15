"""Minimal multi-object tracking, so detections gain identity across frames.

Proximity is a snapshot; *approach* is its derivative, and you cannot take a
derivative without knowing that the person in this frame is the same person as
in the last one. This is the smallest thing that provides that.

Matching is by foot-point distance rather than IoU, which is deliberate: the
API analyses roughly 2 frames per second (see config.FRAME_SAMPLE_FPS), and
over half a second a walking person's boxes frequently do not overlap at all.
IoU would silently break every track; a distance gate scaled to the box size
survives it, and stays scale-invariant between aerial and ground footage.

Greedy nearest-first assignment is used instead of the Hungarian algorithm.
With a handful of objects per frame the two rarely disagree, and greedy costs
no dependency and no explaining.
"""

from collections import deque
from itertools import count

from app import config
from app.geometry import foot_point


class Track:
    """One object followed across frames, with a short position history."""

    __slots__ = (
        "track_id",
        "label",
        "detection",
        "box",
        "foot",
        "history",
        "misses",
        "last_seen_at",
    )

    def __init__(self, track_id: int, detection: dict, timestamp: float):
        self.track_id = track_id
        self.label = detection["label"]
        self.history: deque[tuple[float, list[int]]] = deque(
            maxlen=config.TRACK_HISTORY
        )
        self.misses = 0
        self.observe(detection, timestamp)

    def observe(self, detection: dict, timestamp: float) -> None:
        self.detection = detection
        self.box = detection["box"]
        self.foot = foot_point(self.box)
        self.last_seen_at = timestamp
        self.history.append((timestamp, self.box))
        self.misses = 0

    @property
    def confidence(self) -> float:
        return self.detection["confidence"]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Track {self.track_id} {self.label} misses={self.misses}>"


def gate(box: list[int], elapsed: float) -> float:
    """How far an object may plausibly move since a track was last seen.

    Two things scale it. Its own size, so the gate means the same for a person
    filling the frame and one twenty pixels tall — a running adult covers
    roughly its own height in half a second at any zoom level. And the time
    actually elapsed, because a fixed gate would be quietly calibrated to one
    value of FRAME_SAMPLE_FPS: halve the sample rate and every track would
    break, taking the distance histories with it.

    The time stretch is capped, or a track that vanished for several seconds
    would come back with a gate wide enough to claim anything on screen.
    """
    expected = 1.0 / config.FRAME_SAMPLE_FPS if config.FRAME_SAMPLE_FPS > 0 else 0.0
    stretch = 1.0
    if expected > 0 and elapsed > expected:
        stretch = min(config.TRACK_GATE_MAX_STRETCH, elapsed / expected)

    return config.TRACK_GATE_SCALE * max(box[2], box[3]) * stretch


class CentroidTracker:
    """Assigns stable ids to detections. One instance per video."""

    def __init__(self) -> None:
        self._tracks: dict[int, Track] = {}
        self._ids = count(1)

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks.values())

    def update(self, detections: list[dict], timestamp: float) -> list[Track]:
        """Fold one frame's detections in; return the tracks seen in this frame.

        Tracks that went unmatched are kept for a few frames before being
        dropped, so a single missed detection does not restart a track and
        discard the distance history that "approaching" depends on.
        """
        candidates = []
        for track_id, track in self._tracks.items():
            for index, detection in enumerate(detections):
                if detection["label"] != track.label:
                    continue
                foot = foot_point(detection["box"])
                distance = (
                    (foot[0] - track.foot[0]) ** 2 + (foot[1] - track.foot[1]) ** 2
                ) ** 0.5
                if distance <= gate(detection["box"], timestamp - track.last_seen_at):
                    candidates.append((distance, track_id, index))

        candidates.sort()
        claimed_tracks: set[int] = set()
        claimed_detections: set[int] = set()

        for _, track_id, index in candidates:
            if track_id in claimed_tracks or index in claimed_detections:
                continue
            claimed_tracks.add(track_id)
            claimed_detections.add(index)
            self._tracks[track_id].observe(detections[index], timestamp)

        seen = [self._tracks[track_id] for track_id in claimed_tracks]

        for index, detection in enumerate(detections):
            if index in claimed_detections:
                continue
            track = Track(next(self._ids), detection, timestamp)
            self._tracks[track.track_id] = track
            seen.append(track)

        for track_id, track in list(self._tracks.items()):
            if track_id in claimed_tracks or track in seen:
                continue
            track.misses += 1
            if track.misses > config.TRACK_MAX_MISSES:
                del self._tracks[track_id]

        return seen
