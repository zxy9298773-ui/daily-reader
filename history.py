"""
History — JSON-backed tracking of already-sent article URLs.

Ensures the same article is never pushed twice across days.
The JSON file is committed to the repo so it persists across
GitHub Actions runs.
"""
import json
import logging
import os
from typing import List, Set

logger = logging.getLogger(__name__)

_JSON_PATH = os.path.join(os.path.dirname(__file__), "..", "sent_articles.json")


def _load() -> List[str]:
    """Load the list of sent URLs from the JSON file."""
    if not os.path.exists(_JSON_PATH):
        logger.info("sent_articles.json not found, initializing empty list")
        return []
    try:
        with open(_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            logger.warning("sent_articles.json is not a list, resetting")
            return []
    except (json.JSONDecodeError, OSError):
        logger.exception("Failed to read sent_articles.json, starting fresh")
        return []


def _save(urls: List[str]) -> None:
    """Write the list of sent URLs to the JSON file."""
    try:
        with open(_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(urls, f, ensure_ascii=False, indent=2)
        logger.debug("Saved %d sent URL(s) to sent_articles.json", len(urls))
    except OSError:
        logger.exception("Failed to write sent_articles.json")


def get_sent_urls() -> Set[str]:
    """Return the set of all previously pushed article URLs."""
    return set(_load())


def mark_sent(url: str) -> None:
    """Record *url* as successfully pushed."""
    urls = _load()
    if url not in urls:
        urls.append(url)
        _save(urls)


def mark_all_sent(urls: List[str]) -> None:
    """Batch-record multiple URLs as sent (deduplicated)."""
    existing = _load()
    existing_set = set(existing)
    new_urls = [u for u in urls if u not in existing_set]
    if new_urls:
        existing.extend(new_urls)
        _save(existing)
        logger.info("Recorded %d new article(s) as sent", len(new_urls))
    else:
        logger.info("All %d article(s) already recorded", len(urls))
