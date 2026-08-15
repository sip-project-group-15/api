import pytest

from app import config, video_processor


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


def test_rhino_alone_is_not_an_alert():
    """The core rule: wildlife presence is not poaching."""
    probability, severity = video_processor.assess(
        [{"label": "rhino", "confidence": 0.99, "box": [0, 0, 10, 10]}]
    )

    assert probability == 0.0
    assert severity is None


def test_person_alone_is_a_medium_alert():
    probability, severity = video_processor.assess(
        [{"label": "person", "confidence": 0.8, "box": [0, 0, 10, 10]}]
    )

    assert probability == 0.8
    assert severity == "medium"


def test_person_with_rhino_escalates():
    probability, severity = video_processor.assess(
        [
            {"label": "person", "confidence": 0.7, "box": [0, 0, 10, 10]},
            {"label": "rhino", "confidence": 0.9, "box": [0, 0, 10, 10]},
        ]
    )

    # Probability tracks the threat, not the rhino, so the score stays a
    # measure of how sure we are that a person is present.
    assert probability == 0.7
    assert severity == "high"


def test_empty_frame_is_not_an_alert():
    assert video_processor.assess([]) == (0.0, None)
