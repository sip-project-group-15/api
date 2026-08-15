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
    parts = [clock(alert.get("timestamp_seconds", 0)), str(alert.get("severity", "")).upper()]

    separation = alert.get("separation_body_lengths")
    if separation is None:
        parts.append(f"{alert.get('label_seen', 'threat')}, no rhino in view")
    else:
        metres = alert.get("separation_metres_estimate")
        parts.append(f"{separation:.1f} body-lengths (~{metres:.0f}m)")

    rate = alert.get("closing_rate")
    if rate and rate > 0:
        parts.append(f"closing {rate:.2f}/s")

    for factor in alert.get("context_factors") or []:
        parts.append(factor)

    return " ".join(parts[:2]) + " " + ", ".join(parts[2:])


def build_alert_message(video_name: str, alerts: list[dict]) -> str:
    """One SMS for the whole clip, with every alert numbered.

    A message per alert would mean five texts for a single approach — the same
    subject, seconds apart — which costs money and buries the signal. One
    message, ordered by time, reads as the event it actually is.

    Long clips can raise dozens of alerts, so the body is capped at
    SMS_MAX_CHARS and the remainder is counted rather than sent. A truncated
    message that arrives beats a ten-segment one that does not.
    """
    if not alerts:
        return f"Kifaru: no poaching alerts in {video_name}."

    plural = "s" if len(alerts) > 1 else ""
    header = f"Kifaru: {len(alerts)} alert{plural} in {video_name[:40]}"

    lines = []
    for index, alert in enumerate(alerts, start=1):
        line = f"{index}. {summarise(alert)}"
        candidate = "\n".join([header, *lines, line])
        remaining = len(alerts) - index

        # Leave room for the "+N more" tail before committing to this line.
        tail = f"\n+{remaining} more" if remaining else ""
        if len(candidate) + len(tail) > config.SMS_MAX_CHARS:
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
