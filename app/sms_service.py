import json
import os
from urllib import error, request

from dotenv import load_dotenv

from app import config

load_dotenv()

SMS_API_URL = os.getenv("SMS_API_URL", "https://itecsms.rw/api/sendsms")


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_recipients() -> list[str]:
    raw_recipients = os.getenv("SMS_RECIPIENTS", "")
    return [recipient.strip() for recipient in raw_recipients.split(",") if recipient.strip()]


# GSM 03.38, the 7-bit alphabet an SMS is packed into. Anything outside this
# set costs extra or breaks: the characters below are one septet each, the
# extension set is two septets behind an ESC byte, and anything else forces the
# whole message to UCS-2, which drops the limit from 160 characters to 70.
GSM7_BASIC = frozenset(
    "@£$¥èéùìòÇ\nØø\rÅåΔ_ΦΓΛΩΠΨΣΘΞÆæßÉ !\"#¤%&'()*+,-./0123456789:;<=>?"
    "¡ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÑÜ§¿abcdefghijklmnopqrstuvwxyzäöñüà"
)
GSM7_EXTENDED = frozenset("^{}\\[~]|€")

# Characters that read naturally but are not in the basic set.
SUBSTITUTIONS = {
    "~": "",       # was used for "approximately"; needs an ESC byte
    "—": "-", "–": "-", "·": "-", "…": "...",
    "’": "'", "‘": "'", "“": '"', "”": '"',
}


def septets(text: str) -> int:
    """Length in septets, which is what the 160 limit actually counts."""
    return sum(2 if character in GSM7_EXTENDED else 1 for character in text)


def to_gsm7(text: str) -> str:
    """Reduce text to the basic alphabet, dropping anything that survives.

    Escaped and non-GSM characters are the difference between a message that
    arrives and one that arrives as mojibake, and neither failure is visible
    from this side — the provider accepts the request either way.
    """
    out = []
    for character in text:
        replacement = SUBSTITUTIONS.get(character, character)
        out.extend(c for c in replacement if c in GSM7_BASIC)

    return "".join(out)


def clock(seconds: float) -> str:
    """Seconds into m:ss, so a ranger can scrub straight to the moment."""
    total = int(seconds)
    return f"{total // 60}:{total % 60:02d}"


def group_events(alerts: list[dict]) -> list[list[dict]]:
    """Collapse consecutive alerts about the same subject into one event.

    A person walking towards a rhino raises an alert on every analysed frame.
    Those are one thing happening, not five, and a ranger who gets five lines
    describing the same approach learns nothing the first line did not say.
    """
    events: list[list[dict]] = []
    for alert in alerts:
        track = alert.get("track_id")
        if events and track is not None and events[-1][-1].get("track_id") == track:
            events[-1].append(alert)
        else:
            events.append([alert])

    return events


SUBJECTS = {"person": "Someone", "vehicle": "A vehicle", "weapon": "An armed person"}


def describe_event(event: list[dict]) -> str:
    """One event in the words a ranger would use.

    Distance in metres, not body-lengths: the body-length figure is how the
    scorer measures, not something anyone reading a text at night can act on.
    """
    subject = SUBJECTS.get(str(event[-1].get("threat_label", "")).lower(), "Someone")

    metres = [a["separation_metres_estimate"] for a in event
              if a.get("separation_metres_estimate") is not None]
    closing = any((a.get("closing_rate") or 0) > 0 for a in event)

    if not metres:
        return f"{subject} was seen, but no rhino was in view"

    nearest = min(metres)
    verb = "approaching" if closing else "close to"

    return f"{subject} {verb} a rhino, about {nearest:.0f}m away"


def when(event: list[dict]) -> str:
    """The moment in the clip, so the frame can be found."""
    start = clock(event[0].get("timestamp_seconds", 0))
    end = clock(event[-1].get("timestamp_seconds", 0))

    return start if start == end else f"{start}-{end}"


def build_alert_message(video_name: str, alerts: list[dict]) -> str:
    """One plain-language SMS for the whole clip.

    Written for someone deciding whether to get up, not for someone tuning
    thresholds. No scores, no body-lengths, no severity codes — those are all
    on the dashboard, and the message says to go there.

    Capped at one GSM-7 segment. Past 160 septets an SMS is split into a
    concatenated message needing a header and septet padding to stay aligned;
    this provider gets that wrong and the whole thing arrives as mojibake, with
    no error on the sending side.
    """
    if not alerts:
        return to_gsm7(f"Kifaru: nothing suspicious in {video_name[:30]}")

    events = group_events(alerts)
    header = to_gsm7(f"KIFARU ALERT - {video_name[:24]}")
    footer = "Check the dashboard."

    lines = []
    for index, event in enumerate(events, start=1):
        prefix = f"{index}. " if len(events) > 1 else ""
        line = to_gsm7(f"{prefix}{describe_event(event)} ({when(event)})")
        remaining = len(events) - index
        tail = f"\n+{remaining} more" if remaining else ""
        candidate = "\n".join([header, *lines, line, footer]) + tail

        if septets(candidate) > config.SMS_MAX_SEPTETS:
            dropped = len(events) - index + 1
            return "\n".join([header, *lines, f"+{dropped} more", footer])

        lines.append(line)

    return "\n".join([header, *lines, footer])


def send_alert_sms(video_name: str, alerts: list[dict]) -> dict:
    """Send exactly one message covering every alert in the clip."""
    if not env_bool("SMS_ENABLED"):
        return {"enabled": False, "sent": False, "detail": "SMS disabled"}

    api_key = os.getenv("SMS_API_KEY", "").strip()
    recipients = get_recipients()

    if not api_key or not recipients:
        return {
            "enabled": True,
            "sent": False,
            "detail": "SMS API key or recipients missing",
        }

    payload = {
        "key": api_key,
        "message": build_alert_message(video_name, alerts),
        "recipients": recipients,
    }
    body = json.dumps(payload).encode("utf-8")
    sms_request = request.Request(
        SMS_API_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(sms_request, timeout=10) as response:
            response_body = response.read().decode("utf-8")
            provider_response = json.loads(response_body) if response_body else {}
            sent = response.status == 200
            return {
                "enabled": True,
                "sent": sent,
                "status_code": response.status,
                "detail": provider_response.get("message", "SMS request completed"),
            }
    except error.HTTPError as exception:
        response_body = exception.read().decode("utf-8")
        try:
            provider_response = json.loads(response_body) if response_body else {}
        except json.JSONDecodeError:
            provider_response = {}

        return {
            "enabled": True,
            "sent": False,
            "status_code": exception.code,
            "detail": provider_response.get("message", "SMS request failed"),
        }
    except (error.URLError, TimeoutError) as exception:
        reason = getattr(exception, "reason", str(exception))
        return {
            "enabled": True,
            "sent": False,
            "detail": f"SMS request failed: {reason}",
        }
