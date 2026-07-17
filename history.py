"""
History — SQLite-backed tracking of already-sent article URLs.

Ensures the same article is never pushed twice across days.
"""
import sqlite3
import logging
import os
from datetime import datetime
from typing import Set

logger = logging.getLogger(__name__)

_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "daily_reader.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS sent_articles ("
        "  url TEXT PRIMARY KEY,"
        "  sent_at TEXT NOT NULL"
        ")"
    )
    conn.commit()
    return conn


def get_sent_urls() -> Set[str]:
    """Return the set of all previously pushed article URLs."""
    try:
        conn = _connect()
        rows = conn.execute("SELECT url FROM sent_articles").fetchall()
        conn.close()
        return {row[0] for row in rows}
    except Exception:
        logger.exception("Failed to read sent URLs from SQLite")
        return set()


def mark_sent(url: str) -> None:
    """Record *url* as successfully pushed (today)."""
    try:
        conn = _connect()
        conn.execute(
            "INSERT OR IGNORE INTO sent_articles (url, sent_at) VALUES (?, ?)",
            (url, datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()
    except Exception:
        logger.exception("Failed to mark URL as sent: %s", url)


def mark_all_sent(urls: list[str]) -> None:
    """Batch-record multiple URLs as sent."""
    try:
        conn = _connect()
        now = datetime.now().isoformat()
        conn.executemany(
            "INSERT OR IGNORE INTO sent_articles (url, sent_at) VALUES (?, ?)",
            [(u, now) for u in urls],
        )
        conn.commit()
        conn.close()
        logger.info("Recorded %d article(s) as sent", len(urls))
    except Exception:
        logger.exception("Failed to batch-mark URLs as sent")
