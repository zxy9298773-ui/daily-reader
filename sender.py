"""
Send the HTML newsletter via Resend HTTP API (not SMTP).
Falls back to printing the email to the console on failure.
"""
import logging
import os
import requests

import config

logger = logging.getLogger(__name__)

# 从环境变量读取 Resend API Key（在 Railway 里设置）
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
# 免费版只能用这个发件人，除非你在 Resend 验证了自定义域名
FROM_EMAIL = "外刊推送 <onboarding@resend.dev>"


def send_email(subject: str, html_content: str) -> bool:
    """Send *html_content* to the configured recipient via Resend API.

    Returns ``True`` on success, ``False`` on any failure (the fallback
    printer is always called on failure).
    """
    # ── 检查必要配置 ──────────────────────────────────────────────
    if not RESEND_API_KEY:
        logger.warning("RESEND_API_KEY not set – printing to console")
        _print_to_console(subject, html_content)
        return False

    if not config.TO_EMAIL:
        logger.warning("TO_EMAIL not set – printing to console")
        _print_to_console(subject, html_content)
        return False

    # ── 调用 Resend HTTP API ──────────────────────────────────────
    payload = {
        "from": FROM_EMAIL,
        "to": [config.TO_EMAIL],
        "subject": subject,
        "html": html_content,
    }

    headers = {
        "Authorization": f"Bearer {RESEND_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(
            "https://api.resend.com/emails",
            json=payload,
            headers=headers,
            timeout=30,  # 超时 30 秒，不会一直卡着
        )

        if resp.ok:
            logger.info("Email sent successfully via Resend API (id=%s)", resp.json().get("id"))
            return True
        else:
            logger.error(
                "Resend API error: %s %s – %s",
                resp.status_code,
                resp.reason,
                resp.text,
            )
    except requests.exceptions.Timeout:
        logger.error("Resend API timed out after 30s")
    except requests.exceptions.ConnectionError as e:
        logger.error("Resend API connection error: %s", e)
    except Exception:
        logger.exception("Resend API send failed")

    # ── fallback ───────────────────────────────────────────────────
    _print_to_console(subject, html_content)
    return False


def _print_to_console(subject: str, html_content: str):
    """Print a summary of the email to stdout (debug/dev fallback)."""
    line = "═" * 60
    print(f"\n{line}")
    print(f"  SUBJECT: {subject}")
    print(line)
    print("  HTML (first 2500 chars):")
    print(f"{line[2:]}")
    print(html_content[:2500])
    print(f"\n{line}\n")
