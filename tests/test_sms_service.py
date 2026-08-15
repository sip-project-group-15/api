"""Tests for the alert SMS.

One clip produces one message. A text per alert would mean five for a single
approach — the same subject, seconds apart — so the interesting behaviour is
how several alerts collapse into one readable body that still fits in a
sensible number of segments.
"""

import pytest

from app import config, sms_service


def alert(seconds=1.5, severity="critical", separation=1.8, metres=6.7,
          closing=0.55, factors=None):
    return {
        "timestamp_seconds": seconds,
        "severity": severity,
        "separation_body_lengths": separation,
        "separation_metres_estimate": metres,
        "closing_rate": closing,
        "context_factors": factors or [],
    }


# ── Formatting ───────────────────────────────────────────────────────────────


def test_seconds_become_a_clock_a_ranger_can_scrub_to():
    assert sms_service.clock(0) == "0:00"
    assert sms_service.clock(9.4) == "0:09"
    assert sms_service.clock(75) == "1:15"
    assert sms_service.clock(3661) == "61:01"


def test_a_line_carries_time_severity_distance_and_speed():
    line = sms_service.summarise(alert())

    assert "1:30".replace("1:30", "0:01") in line  # 1.5s -> 0:01
    assert "CRITICAL" in line
    assert "1.8 body-lengths" in line
    assert "~7m" in line
    assert "closing 0.55/s" in line


def test_a_stationary_threat_reports_no_closing_speed():
    assert "closing" not in sms_service.summarise(alert(closing=0.0))
    assert "closing" not in sms_service.summarise(alert(closing=None))


def test_a_retreating_threat_reports_no_closing_speed():
    """Negative is moving away; reporting it as a speed would read as a charge."""
    assert "closing" not in sms_service.summarise(alert(closing=-0.4))


def test_an_unmeasurable_distance_says_so_rather_than_inventing_one():
    line = sms_service.summarise(alert(separation=None, metres=None))

    assert "no rhino in view" in line
    assert "body-lengths" not in line


def test_context_factors_are_included():
    line = sms_service.summarise(alert(factors=["vehicle present", "group of 4"]))

    assert "vehicle present" in line
    assert "group of 4" in line


# ── One message per clip ─────────────────────────────────────────────────────


def test_every_alert_is_numbered_in_one_message():
    message = sms_service.build_alert_message("clip.mp4", [alert(), alert(), alert()])

    assert message.count("\n") == 3          # header + three lines
    assert "1. " in message and "2. " in message and "3. " in message
    assert "3 alerts" in message
    assert "clip.mp4" in message


def test_a_single_alert_is_not_pluralised():
    message = sms_service.build_alert_message("clip.mp4", [alert()])

    assert "1 alert in" in message
    assert "1 alerts" not in message


def test_no_alerts_still_produces_a_sane_message():
    assert "no poaching alerts" in sms_service.build_alert_message("clip.mp4", [])


def test_alerts_keep_their_order():
    """Chronological, so the message reads as the event it describes."""
    message = sms_service.build_alert_message(
        "clip.mp4", [alert(seconds=1), alert(seconds=30), alert(seconds=90)]
    )
    body = message.splitlines()

    assert body[1].startswith("1. 0:01")
    assert body[2].startswith("2. 0:30")
    assert body[3].startswith("3. 1:30")


def test_a_long_clip_is_truncated_rather_than_sent_as_ten_segments(monkeypatch):
    """A truncated message that arrives beats a huge one that does not."""
    monkeypatch.setattr(config, "SMS_MAX_CHARS", 300)

    message = sms_service.build_alert_message("clip.mp4", [alert()] * 40)

    assert len(message) <= 300
    assert "more" in message.splitlines()[-1]


def test_truncation_counts_every_alert_it_left_out(monkeypatch):
    monkeypatch.setattr(config, "SMS_MAX_CHARS", 300)

    message = sms_service.build_alert_message("clip.mp4", [alert()] * 40)
    listed = sum(1 for line in message.splitlines() if line[:1].isdigit())
    dropped = int(message.splitlines()[-1].lstrip("+").split()[0])

    assert listed + dropped == 40


def test_a_realistic_clip_fits_in_two_segments():
    """The whole point of the compact line format."""
    alerts = [
        alert(seconds=1.0, severity="high", separation=2.09, metres=7.7, closing=0.542),
        alert(seconds=1.5, separation=1.81, metres=6.7, closing=0.546),
        alert(seconds=2.0, separation=1.54, metres=5.7, closing=0.547),
        alert(seconds=2.5, separation=1.27, metres=4.7, closing=0.546),
        alert(seconds=3.0, separation=1.00, metres=3.7, closing=0.545),
    ]

    message = sms_service.build_alert_message("approach.mp4", alerts)

    assert len(message) <= 320
    assert "+" not in message  # nothing dropped


def test_an_overlong_filename_cannot_crowd_out_the_alerts():
    message = sms_service.build_alert_message("a" * 200 + ".mp4", [alert()])

    assert len(message) <= config.SMS_MAX_CHARS
    assert "1. " in message


# ── Sending ──────────────────────────────────────────────────────────────────


def test_disabled_sms_reports_cleanly_without_sending(monkeypatch):
    monkeypatch.delenv("SMS_ENABLED", raising=False)

    result = sms_service.send_alert_sms("clip.mp4", [alert()])

    assert result == {"enabled": False, "sent": False, "detail": "SMS disabled"}


def test_missing_credentials_are_reported_not_raised(monkeypatch):
    monkeypatch.setenv("SMS_ENABLED", "true")
    monkeypatch.setenv("SMS_API_KEY", "")
    monkeypatch.setenv("SMS_RECIPIENTS", "")

    result = sms_service.send_alert_sms("clip.mp4", [alert()])

    assert result["enabled"] is True
    assert result["sent"] is False
    assert "missing" in result["detail"]


def test_the_provider_receives_one_message_for_all_alerts(monkeypatch):
    """The behaviour change: not one request per alert."""
    monkeypatch.setenv("SMS_ENABLED", "true")
    monkeypatch.setenv("SMS_API_KEY", "k")
    monkeypatch.setenv("SMS_RECIPIENTS", "250780000000")
    sent = []

    class Response:
        status = 200
        def read(self): return b'{"message":"ok"}'
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=None):
        sent.append(req)
        return Response()

    monkeypatch.setattr(sms_service.request, "urlopen", fake_urlopen)

    result = sms_service.send_alert_sms("clip.mp4", [alert(), alert(), alert()])

    assert result["sent"] is True
    assert len(sent) == 1
    import json
    body = json.loads(sent[0].data)
    assert "3 alerts" in body["message"]
    assert body["recipients"] == ["250780000000"]
