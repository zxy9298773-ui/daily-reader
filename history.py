"""
History — JSON-backed tracking of already-sent article URLs and source
rotation fairness.

Ensures the same article is never pushed twice across days, and that
every source gets pushed at least once per week.
The JSON files are committed to the repo so they persist across
GitHub Actions runs.
"""
import json
import logging
import os
from typing import List, Set, Dict

logger = logging.getLogger(__name__)

_JSON_PATH = os.path.join(os.path.dirname(__file__), "sent_articles.json")


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


# ═══════════════════════════════════════════════════════════════════
#  Source rotation tracking  (ensures every source is served ≥1×/week)
# ═══════════════════════════════════════════════════════════════════

_ROTATION_PATH = os.path.join(os.path.dirname(__file__), "source_rotation.json")


def _load_rotation() -> Dict[str, str]:
    """Load the {source_name: last_push_date} mapping."""
    if not os.path.exists(_ROTATION_PATH):
        return {}
    try:
        with open(_ROTATION_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
            return {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_rotation(data: Dict[str, str]) -> None:
    """Write the rotation mapping to disk."""
    try:
        with open(_ROTATION_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.debug("Saved rotation data (%d source(s))", len(data))
    except OSError:
        logger.exception("Failed to write source_rotation.json")


def get_source_last_seen() -> Dict[str, str]:
    """Return {source_name: last_push_date_string}.

    Sources that have never been pushed are not present in the dict,
    meaning they are immediately eligible for rotation priority.
    """
    return _load_rotation()


def mark_sources_pushed(sources: List[str], date_str: str) -> None:
    """Record that *sources* were pushed on *date_str* (expects YYYY-MM-DD).

    Only non-empty source names are recorded (link-list fallback and
    similar edge cases are ignored).
    """
    if not sources:
        return
    data = _load_rotation()
    changed = False
    for s in sources:
        if not s:
            continue
        if data.get(s) != date_str:
            data[s] = date_str
            changed = True
    if changed:
        _save_rotation(data)
        logger.info("Updated rotation for %d source(s): %s", len(sources), date_str)
