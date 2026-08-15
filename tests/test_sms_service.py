"""Tests for the alert SMS.

One clip produces one message. A text per alert would mean five for a single
approach — the same subject, seconds apart — so the interesting behaviour is
how several alerts collapse into one readable body that still fits in a
sensible number of segments.
"""

import pytest

from app import config, sms_service


def alert(seconds=1.5, severity="critical", metres=6.7, closing=0.55,
          track=1, label="person"):
    return {
        "timestamp_seconds": seconds,
        "severity": severity,
        "separation_body_lengths": None if metres is None else metres / 3.7,
        "separation_metres_estimate": metres,
        "closing_rate": closing,
        "context_factors": [],
        "track_id": track,
        "threat_label": label,
    }


# ── Wording ─────────────────────────────────────────────────────────────────
# The message goes to someone deciding whether to get up, not to someone tuning
# thresholds. Scores, body-lengths and severity codes belong on the dashboard.


def test_seconds_become_a_clock_a_ranger_can_scrub_to():
    assert sms_service.clock(0) == "0:00"
    assert sms_service.clock(9.4) == "0:09"
    assert sms_service.clock(75) == "1:15"


def test_an_approach_is_described_as_one():
    line = sms_service.describe_event([alert()])

    assert line == "Someone approaching a rhino, about 7m away"


def test_a_stationary_subject_is_not_described_as_approaching():
    line = sms_service.describe_event([alert(closing=0.0)])

    assert "approaching" not in line
    assert "close to a rhino" in line


def test_a_retreating_subject_is_not_described_as_approaching():
    assert "approaching" not in sms_service.describe_event([alert(closing=-0.4)])


def test_the_subject_is_named():
    assert sms_service.describe_event([alert(label="vehicle")]).startswith("A vehicle")
    assert sms_service.describe_event([alert(label="weapon")]).startswith("An armed person")


def test_an_unmeasurable_distance_says_so_rather_than_inventing_one():
    line = sms_service.describe_event([alert(metres=None)])

    assert "no rhino was in view" in line
    assert "m away" not in line


def test_the_message_carries_no_internal_units_or_scores():
    """Body-lengths are how the scorer measures, not something to act on."""
    message = sms_service.build_alert_message("clip.mp4", [alert(), alert(metres=3.0)])

    for jargon in ("body-length", "lengths", "septet", "proximity", "persistence",
                   "0.4", "score", "CRIT", "threshold"):
        assert jargon not in message, jargon


def test_the_message_says_what_to_do_next():
    assert "dashboard" in sms_service.build_alert_message("clip.mp4", [alert()])


# ── One message per clip ─────────────────────────────────────────────────────


def test_one_subject_across_many_frames_is_one_line():
    """Five alerts from one approach are one event, not five."""
    walk = [alert(seconds=t, metres=m, track=1)
            for t, m in ((1.0, 8.7), (1.5, 7.6), (2.0, 5.4), (2.5, 4.3), (3.0, 3.1))]

    message = sms_service.build_alert_message("clip.mp4", walk)

    assert len([l for l in message.splitlines() if "rhino" in l]) == 1
    assert "1." not in message           # nothing to number when there is one event


def test_the_nearest_distance_of_the_event_is_the_one_reported():
    walk = [alert(seconds=1.0, metres=20.0), alert(seconds=2.0, metres=4.0)]

    assert "about 4m away" in sms_service.build_alert_message("clip.mp4", walk)


def test_separate_subjects_are_numbered():
    message = sms_service.build_alert_message(
        "clip.mp4", [alert(track=1), alert(track=2, label="vehicle")]
    )

    assert "1. " in message and "2. " in message


def test_an_event_spanning_time_reports_a_range():
    walk = [alert(seconds=1.0, track=1), alert(seconds=65.0, track=1)]

    assert "(0:01-1:05)" in sms_service.build_alert_message("clip.mp4", walk)


def test_no_alerts_still_produces_a_sane_message():
    assert "nothing suspicious" in sms_service.build_alert_message("clip.mp4", [])


def test_many_subjects_are_truncated_with_a_count():
    message = sms_service.build_alert_message(
        "clip.mp4", [alert(seconds=float(i), track=i) for i in range(1, 12)]
    )
    listed = sum(1 for line in message.splitlines() if line[:1].isdigit())
    dropped = int([l for l in message.splitlines() if l.startswith("+")][0].lstrip("+").split()[0])

    assert listed + dropped == 11
    assert sms_service.septets(message) <= 160


def test_an_overlong_filename_cannot_crowd_out_the_alerts():
    message = sms_service.build_alert_message("a" * 200 + ".mp4", [alert()])

    assert sms_service.septets(message) <= config.SMS_MAX_SEPTETS
    assert "rhino" in message


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
    assert "rhino" in body["message"]
    assert body["recipients"] == ["250780000000"]


# ── GSM-7 encoding ───────────────────────────────────────────────────────────
# Past 160 septets an SMS is split into a concatenated message, which the
# provider mis-assembles into GSM-7 mojibake with no error on the sending side.
# These are the tests that would have caught that before it reached a handset.


def test_every_message_fits_one_segment():
    """A concatenated SMS is what produced the mojibake."""
    for count in (1, 2, 5, 20, 200):
        message = sms_service.build_alert_message("approach.mp4", [alert()] * count)
        assert sms_service.septets(message) <= 160, f"{count} alerts -> {septets(message)}"


def test_every_message_is_pure_basic_gsm7():
    """An escaped character costs two septets and is mangled by this provider."""
    for count in (0, 1, 5, 40):
        message = sms_service.build_alert_message("clip.mp4", [alert()] * count)
        outside = sorted(set(message) - sms_service.GSM7_BASIC)
        assert not outside, f"{count} alerts -> {outside}"


def test_the_tilde_that_caused_it_is_gone():
    """"(~7m)" needed a GSM-7 escape; it is now written as "about 7m"."""
    assert "~" not in sms_service.describe_event([alert()])
    assert "about" in sms_service.describe_event([alert()])


def test_septets_counts_escapes_as_two():
    assert sms_service.septets("abc") == 3
    assert sms_service.septets("a~c") == 4
    assert sms_service.septets("[]") == 4


def test_non_gsm7_characters_are_replaced_not_passed_through():
    """A single one of these flips the whole message to UCS-2, halving the limit."""
    assert sms_service.to_gsm7("2.4 lengths (~9m)") == "2.4 lengths (9m)"
    assert sms_service.to_gsm7("closing \u2014 fast") == "closing - fast"
    assert sms_service.to_gsm7("caf\u00e9 \u4f60\u597d") == "caf\u00e9 "


def test_an_unexpected_filename_cannot_break_the_encoding():
    """Uploads are user-named, so the filename reaches the SMS unfiltered."""
    message = sms_service.build_alert_message("\u5075\u5bdf\u673a-\ud83e\udd8f.mp4", [alert()])

    assert not set(message) - sms_service.GSM7_BASIC
    assert sms_service.septets(message) <= 160
