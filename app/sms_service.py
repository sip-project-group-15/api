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


def summarise(alert: dict) -> str:
    """One alert as a compact line.

    Built from the structured fields rather than reusing `message`, which is
    written for a screen and is far too long once several are stacked into a
    160-character segment.
    """
    severity = {"critical": "CRIT", "high": "HIGH", "medium": "MED"}.get(
        str(alert.get("severity", "")).lower(), "ALERT"
    )
    parts = [clock(alert.get("timestamp_seconds", 0)), severity]

    separation = alert.get("separation_body_lengths")
    if separation is None:
        parts.append("no rhino in view")
    else:
        metres = alert.get("separation_metres_estimate")
        parts.append(f"{separation:.1f} lengths {metres:.0f}m")

    rate = alert.get("closing_rate")
    if rate and rate > 0:
        parts.append("closing")

    for factor in alert.get("context_factors") or []:
        parts.append(factor)

    return " ".join(parts)


def build_alert_message(video_name: str, alerts: list[dict]) -> str:
    """One SMS for the whole clip, with every alert numbered.

    A message per alert would mean five texts for a single approach — the same
    subject, seconds apart — which costs money and buries the signal. One
    message, ordered by time, reads as the event it actually is.

    The body is capped at SMS_MAX_SEPTETS and the remainder counted rather than
    sent. That default is one segment, deliberately: a message longer than 160
    septets is split into a concatenated SMS, which needs a header and a bit of
    septet padding to stay aligned. Get that wrong and the whole thing arrives
    as GSM-7 mojibake — Greek letters and a run of '@' — which is exactly what
    this provider does with multi-segment messages. One segment always
    survives; the dashboard has the rest.
    """
    if not alerts:
        return to_gsm7(f"Kifaru: no poaching alerts in {video_name}")

    plural = "s" if len(alerts) > 1 else ""
    header = to_gsm7(f"Kifaru: {len(alerts)} alert{plural} in {video_name[:24]}")

    lines = []
    for index, alert in enumerate(alerts, start=1):
        line = to_gsm7(f"{index}. {summarise(alert)}")
        remaining = len(alerts) - index

        # Leave room for the "+N more" tail before committing to this line.
        tail = f"\n+{remaining} more" if remaining else ""
        candidate = "\n".join([header, *lines, line]) + tail
        if septets(candidate) > config.SMS_MAX_SEPTETS:
            return "\n".join([header, *lines, f"+{len(alerts) - index + 1} more"])

        lines.append(line)

    return "\n".join([header, *lines])


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
