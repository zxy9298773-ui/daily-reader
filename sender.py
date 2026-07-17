"""
Send the HTML newsletter via QQ Mail SMTP.
If SMTP_PASSWORD is missing or the call fails, falls back to printing
the email to the console (useful during development).
"""
import logging
import smtplib
from email.mime.text import MIMEText

import config

logger = logging.getLogger(__name__)


def send_email(subject: str, html_content: str) -> bool:
    """Send *html_content* to the configured recipient via QQ Mail SMTP.

    Returns ``True`` on success, ``False`` on any failure (the fallback
    printer is always called on failure).
    """
    if not config.SMTP_PASSWORD:
        logger.warning("SMTP_PASSWORD not set – printing to console")
        _print_to_console(subject, html_content)
        return False

    if not config.TO_EMAIL or not config.FROM_EMAIL:
        logger.warning("TO_EMAIL or FROM_EMAIL not set – printing to console")
        _print_to_console(subject, html_content)
        return False

    msg = MIMEText(html_content, "html", "utf-8")
    msg["Subject"] = subject
    msg["From"] = config.FROM_EMAIL
    msg["To"] = config.TO_EMAIL

    try:
        with smtplib.SMTP_SSL(config.SMTP_HOST, config.SMTP_PORT, timeout=30) as smtp:
            smtp.login(config.FROM_EMAIL, config.SMTP_PASSWORD)
            smtp.send_message(msg)

        logger.info("Email sent successfully to %s", config.TO_EMAIL)
        return True
    except Exception:
        logger.exception("SMTP send failed")

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
