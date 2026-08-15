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
