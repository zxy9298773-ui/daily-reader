"""
Cleanup — permanently delete old Daily Reader emails from the server
via IMAP, so they do not accumulate in the mailbox.

Called automatically after sending a new email (see ``main.py``).
"""
import imaplib
import logging
from datetime import datetime, timedelta

import config

logger = logging.getLogger(__name__)

# For security, only delete emails whose subject contains this string
_SUBJECT_FILTER = config.CLEANUP_SEARCH_SUBJECT.encode("utf-8")


def cleanup_old_emails(days: int | None = None) -> int:
    """Permanently delete sent Daily-Reader emails older than *days*.

    Args:
        days: Age threshold.  Defaults to ``config.CLEANUP_AFTER_DAYS``.

    Returns:
        Number of emails deleted (0 if none / error).
    """
    days = days or config.CLEANUP_AFTER_DAYS

    if not config.SMTP_PASSWORD:
        logger.warning("SMTP_PASSWORD not set — skipping IMAP cleanup")
        return 0

    before = (datetime.now() - timedelta(days=days)).strftime("%d-%b-%Y")

    try:
        mail = imaplib.IMAP4_SSL(config.IMAP_HOST, config.IMAP_PORT)
        mail.login(config.FROM_EMAIL, config.SMTP_PASSWORD)
    except Exception:
        logger.exception("IMAP login failed — cleanup skipped")
        return 0

    deleted_count = 0

    # ── Delete matching emails from INBOX ──────────────────────────
    try:
        mail.select("INBOX")
        # Only target emails we sent, with matching subject, older than N days
        status, data = mail.search(
            None,
            f'(FROM "{config.FROM_EMAIL}" SENTBEFORE {before})',
        )

        if status == "OK" and data and data[0]:
            raw_ids: list[bytes] = data[0].split()
            # Further filter by subject containing "Daily Reader"
            matched_ids = []
            for rid in raw_ids:
                _typ, msg_data = mail.fetch(rid, "(BODY.PEEK[HEADER.FIELDS (Subject)])")
                for part in msg_data:
                    if isinstance(part, tuple) and _SUBJECT_FILTER in part[1]:
                        matched_ids.append(rid)
                        break

            if matched_ids:
                for uid in matched_ids:
                    mail.store(uid, "+FLAGS", "\\Deleted")
                mail.expunge()
                deleted_count = len(matched_ids)
                logger.info(
                    "Deleted %d old email(s) from INBOX (≥%d days)",
                    deleted_count,
                    days,
                )
    except Exception:
        logger.exception("Failed to clean INBOX")

    # ── Also purge Trash to prevent recovery ───────────────────────
    trash_folders = ["Trash", "[Gmail]/Trash", "垃圾箱"]
    for folder in trash_folders:
        try:
            status, _ = mail.select(folder)
            if status == "OK":
                mail.store("1:*", "+FLAGS", "\\Deleted")
                mail.expunge()
                logger.info("Purged folder: %s", folder)
                break
        except Exception:
            continue

    try:
        mail.logout()
    except Exception:
        pass

    return deleted_count
