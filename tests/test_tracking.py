from app import config
from app.tracking import CentroidTracker


def detection(label, x, y, w=20, h=40, confidence=0.9):
    return {"label": label, "confidence": confidence, "box": [x, y, w, h]}


def test_a_moving_object_keeps_its_id():
    tracker = CentroidTracker()

    first = tracker.update([detection("person", 100, 100)], 0.0)
    second = tracker.update([detection("person", 110, 100)], 0.5)

    assert first[0].track_id == second[0].track_id


def test_boxes_that_no_longer_overlap_still_match():
    """The reason matching is by distance and not IoU.

    At ~2fps sampling a walking person's boxes routinely fail to overlap
    between analysed frames. IoU matching would break the track every time and
    silently destroy the history that 'approaching' is derived from.
    """
    tracker = CentroidTracker()
    width = 20

    first = tracker.update([detection("person", 100, 100, w=width)], 0.0)
    # Moved further than its own width — zero overlap with the previous box.
    second = tracker.update([detection("person", 100 + width * 2, 100, w=width)], 0.5)

    assert first[0].track_id == second[0].track_id


def test_an_implausible_jump_starts_a_new_track():
    tracker = CentroidTracker()

    first = tracker.update([detection("person", 100, 100)], 0.0)
    second = tracker.update([detection("person", 5000, 3000)], 0.5)

    assert first[0].track_id != second[0].track_id


def test_classes_never_match_each_other():
    """A rhino must not inherit the id of a person standing where it now is."""
    tracker = CentroidTracker()

    person = tracker.update([detection("person", 100, 100)], 0.0)
    rhino = tracker.update([detection("rhino", 100, 100)], 0.5)

    assert person[0].track_id != rhino[0].track_id


def test_a_missed_frame_does_not_restart_the_track():
    """One dropped detection must not cost the accumulated distance history."""
    tracker = CentroidTracker()

    first = tracker.update([detection("person", 100, 100)], 0.0)
    tracker.update([], 0.5)
    resumed = tracker.update([detection("person", 105, 100)], 1.0)

    assert first[0].track_id == resumed[0].track_id


def test_a_long_absence_drops_the_track():
    tracker = CentroidTracker()

    first = tracker.update([detection("person", 100, 100)], 0.0)
    for index in range(config.TRACK_MAX_MISSES + 1):
        tracker.update([], 0.5 * (index + 1))

    returned = tracker.update([detection("person", 100, 100)], 10.0)

    assert first[0].track_id != returned[0].track_id
    assert len(tracker.tracks) == 1


def test_two_people_keep_separate_ids():
    tracker = CentroidTracker()

    first = tracker.update([detection("person", 100, 100), detection("person", 400, 100)], 0.0)
    second = tracker.update([detection("person", 110, 100), detection("person", 390, 100)], 0.5)

    assert {track.track_id for track in first} == {track.track_id for track in second}
    assert len({track.track_id for track in second}) == 2


def test_history_is_bounded():
    tracker = CentroidTracker()

    for index in range(config.TRACK_HISTORY * 3):
        tracker.update([detection("person", 100 + index, 100)], 0.5 * index)

    assert len(tracker.tracks[0].history) == config.TRACK_HISTORY


def test_update_returns_only_tracks_seen_this_frame():
    tracker = CentroidTracker()

    tracker.update([detection("person", 100, 100)], 0.0)
    seen = tracker.update([detection("rhino", 400, 400)], 0.5)

    assert [track.label for track in seen] == ["rhino"]
    # The person is retained internally, still inside its miss allowance.
    assert len(tracker.tracks) == 2
