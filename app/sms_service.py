import json
import os
from urllib import error, request

from dotenv import load_dotenv

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


def build_alert_message(video_name: str, alert: dict) -> str:
    probability = alert["probability"]
    return f"Poaching alert detected with probability {probability:.2f}."


def send_alert_sms(video_name: str, alert: dict) -> dict:
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
        "message": build_alert_message(video_name, alert),
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
