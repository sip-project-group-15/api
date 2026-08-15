from pathlib import Path

import numpy as np
import pytest

from app import config, video_processor
from tests.test_threat import RHINO, person_at

FRAME_W, FRAME_H = 800, 200
SOURCE_FPS = 10.0


@pytest.mark.parametrize(
    ("fps", "sample_fps", "expected"),
    [
        (30.0, 2.0, 15),
        (60.0, 2.0, 30),
        (30.0, 30.0, 1),
        (10.0, 30.0, 1),  # never upsample past the source
    ],
)
def test_frame_stride(fps, sample_fps, expected, monkeypatch):
    monkeypatch.setattr(config, "FRAME_SAMPLE_FPS", sample_fps)

    assert video_processor.frame_stride(fps) == expected


def test_frame_stride_handles_disabled_sampling(monkeypatch):
    monkeypatch.setattr(config, "FRAME_SAMPLE_FPS", 0)

    assert video_processor.frame_stride(30.0) == 1


# Scoring moved out of this module into app/threat.py, where it is assessed
# across frames rather than one at a time. See tests/test_threat.py.


class ApproachingPersonDetector:
    """Stub detector staging a person walking towards a rhino.

    Substituted for the model so the pipeline can be exercised end to end
    without weights: what is under test here is the wiring — sampling, the
    per-video monitor, the alert payload — not detection quality.
    """

    name = "stub"
    is_mock = False

    def predict(self, frame, frame_number):
        # Analysed frames are 1, 1+stride, 1+2*stride...; step the person in
        # towards the rhino on each one.
        step = (frame_number - 1) // video_processor.frame_stride(SOURCE_FPS)
        return [RHINO, person_at(5.0 - 0.5 * step)]


@pytest.fixture
def clip(tmp_path):
    """A real, decodable mp4 — the pipeline opens it with OpenCV for real."""
    import cv2

    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), SOURCE_FPS, (FRAME_W, FRAME_H)
    )
    for _ in range(30):
        writer.write(np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8))
    writer.release()

    return path


def test_process_video_samples_rather_than_analysing_every_frame(clip, tmp_path, monkeypatch):
    """Sampling is what keeps a long clip inside the proxy timeout."""
    monkeypatch.setattr(video_processor, "get_detector", ApproachingPersonDetector)

    result = video_processor.process_video(clip, 1.1, tmp_path)

    assert result["frame_stride"] == 5
    assert result["analyzed_frames"] < result["processed_frames"]
    assert result["analyzed_frames"] == pytest.approx(result["processed_frames"] / 5, abs=1)


def test_an_approach_produces_an_alert_with_its_reasoning(clip, tmp_path, monkeypatch):
    monkeypatch.setattr(video_processor, "get_detector", ApproachingPersonDetector)

    result = video_processor.process_video(clip, config.DEFAULT_ALERT_THRESHOLD, tmp_path)

    assert result["alerts"]
    alert = result["alerts"][-1]

    # The alert must carry why it fired, not just that it did.
    assert alert["components"]["approach"] > 0
    assert alert["closing_rate"] > 0
    assert alert["separation_body_lengths"] is not None
    assert "closing at" in alert["message"]
    assert alert["severity"] in {"medium", "high", "critical"}

    # Keys the existing API response, SMS template and alert store rely on.
    assert alert["probability"] >= config.DEFAULT_ALERT_THRESHOLD
    assert alert["label"] == "possible_poaching"
    assert alert["sms_sent"] is False
    assert (tmp_path / f"frame_{alert['frame_number']:06d}.jpg").is_file()


def test_a_high_threshold_suppresses_the_alert(clip, tmp_path, monkeypatch):
    monkeypatch.setattr(video_processor, "get_detector", ApproachingPersonDetector)

    result = video_processor.process_video(clip, 0.99, tmp_path)

    assert result["alerts"] == []


def test_each_video_is_assessed_independently(clip, tmp_path, monkeypatch):
    """Track ids and distance history from one clip must not leak into the next."""
    monkeypatch.setattr(video_processor, "get_detector", ApproachingPersonDetector)

    first = video_processor.process_video(clip, config.DEFAULT_ALERT_THRESHOLD, tmp_path)
    second = video_processor.process_video(clip, config.DEFAULT_ALERT_THRESHOLD, tmp_path)

    assert [alert["frame_number"] for alert in first["alerts"]] == [
        alert["frame_number"] for alert in second["alerts"]
    ]
    assert [alert["probability"] for alert in first["alerts"]] == [
        alert["probability"] for alert in second["alerts"]
    ]


# ── Frame rate robustness ────────────────────────────────────────────────────


def test_a_sane_frame_rate_is_used_as_is():
    assert video_processor.source_fps(29.97) == pytest.approx(29.97)
    assert video_processor.source_fps(10.0) == 10.0


def test_a_nan_frame_rate_does_not_reach_round():
    """OpenCV returns NaN for some malformed containers, and NaN is truthy —
    an `or 30` fallback lets it through to round(), which raises and turns a
    bad upload into a 500."""
    assert video_processor.source_fps(float("nan")) == 30.0
    video_processor.frame_stride(video_processor.source_fps(float("nan")))


def test_a_missing_frame_rate_falls_back():
    assert video_processor.source_fps(0.0) == 30.0
    assert video_processor.source_fps(-5.0) == 30.0
    assert video_processor.source_fps(float("inf")) == 30.0


def test_an_absurd_frame_rate_falls_back():
    """A huge rate yields a huge stride and only the first frame is analysed;
    a rate near zero yields stride 1 and analyses all 9,000 frames of a clip."""
    assert video_processor.source_fps(1e9) == 30.0


def test_sampling_hits_the_configured_rate(monkeypatch):
    """The property the stride exists to deliver, across common frame rates."""
    monkeypatch.setattr(config, "FRAME_SAMPLE_FPS", 2.0)

    for fps in (10.0, 15.0, 24.0, 30.0, 60.0):
        stride = video_processor.frame_stride(fps)
        effective = fps / stride
        assert 1.5 <= effective <= 3.0, f"{fps}fps -> {effective}fps sampled"


def test_the_first_frame_is_always_analysed_at_time_zero(clip, tmp_path, monkeypatch):
    monkeypatch.setattr(video_processor, "get_detector", ApproachingPersonDetector)

    result = video_processor.process_video(clip, config.DEFAULT_ALERT_THRESHOLD, tmp_path)

    assert result["processed_frames"] == 30
    assert result["analyzed_frames"] == 6      # frames 1, 6, 11, 16, 21, 26
    assert result["alerts"][0]["timestamp_seconds"] >= 0.0


def test_timestamps_advance_by_the_sampling_interval(clip, tmp_path, monkeypatch):
    monkeypatch.setattr(video_processor, "get_detector", ApproachingPersonDetector)

    result = video_processor.process_video(clip, 0.0, tmp_path)
    stamps = [a["timestamp_seconds"] for a in result["alerts"]]
    gaps = [round(b - a, 3) for a, b in zip(stamps, stamps[1:])]

    # stride 5 at 10fps = one analysed frame every 0.5s
    assert all(g == pytest.approx(0.5) for g in gaps), gaps


def test_frame_number_matches_the_snapshot_written(clip, tmp_path, monkeypatch):
    monkeypatch.setattr(video_processor, "get_detector", ApproachingPersonDetector)

    result = video_processor.process_video(clip, 0.0, tmp_path)

    for alert in result["alerts"]:
        assert Path(alert["snapshot_path"]).name == f"frame_{alert['frame_number']:06d}.jpg"
        assert Path(alert["snapshot_path"]).is_file()
