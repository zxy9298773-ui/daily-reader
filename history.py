"""
History — JSON-backed tracking of already-sent article URLs and source
rotation fairness.

Ensures the same article is never pushed twice across days, and that
every source gets pushed at least once per week.
The JSON file is committed to the repo so it persists across
GitHub Actions runs.

Dedup strategy (dual key):
  - Normalized URL (strip query, fragment, trailing slash, www., scheme)
  - Title hash (lowercase, stripped, first 100 chars)
  Either match → article is considered "already sent".

Data format in sent_articles.json:
  List[{"url": str, "title": str}]  (migrates old List[str] on first load)
"""
import hashlib
import json
import logging
import os
import shutil
import tempfile
from typing import Dict, List, Set, Optional
from urllib.parse import urlparse, urlunparse

logger = logging.getLogger(__name__)

_JSON_PATH = os.path.join(os.path.dirname(__file__), "sent_articles.json")
_ROTATION_PATH = os.path.join(os.path.dirname(__file__), "source_rotation.json")

# ═══════════════════════════════════════════════════════════════════
#  URL / title normalization helpers
# ═══════════════════════════════════════════════════════════════════


def normalize_url(url: str) -> str:
    """Strip query params, fragment, trailing slash, www, and scheme.

    Dedup comparison operates on the normalized form so that
    https://site.com/a/?utm_source=rss#section  and
    http://www.site.com/a  resolve to the same key.
    """
    parsed = urlparse(url)
    netloc = parsed.netloc
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parsed.path.rstrip("/")
    return urlunparse(("", netloc, path, "", "", "")).lower()


def _title_key(title: str) -> str:
    """Deterministic key derived from the article title."""
    return title.strip().lower()[:100] if title else ""


def _dedup_key(url: str, title: str) -> str:
    """Combined dedup key: normalized_url | title_key."""
    return f"{normalize_url(url)}|{_title_key(title)}"


# ═══════════════════════════════════════════════════════════════════
#  sent_articles.json — load / save
# ═══════════════════════════════════════════════════════════════════


def _load_entries() -> List[Dict]:
    """Load stored entries from JSON, migrating from old List[str] format."""
    if not os.path.exists(_JSON_PATH):
        logger.info("sent_articles.json not found, initializing empty list")
        return []

    try:
        with open(_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Migration: old format List[str] → new format List[Dict]
        if isinstance(data, list) and data and isinstance(data[0], str):
            logger.info(
                "Migrating sent_articles.json from old List[str] format "
                "(%d entries) to new List[Dict] format",
                len(data),
            )
            entries = [{"url": u, "title": ""} for u in data]
            # Rewrite in new format immediately
            _save_entries(entries)
            return entries

        if isinstance(data, list):
            return data

        logger.warning("sent_articles.json is not a list, resetting")
        return []

    except (json.JSONDecodeError, OSError) as exc:
        # ── Backup corrupted file before resetting ────────────────
        backup_path = _JSON_PATH + ".bak"
        try:
            shutil.copy2(_JSON_PATH, backup_path)
            logger.error(
                "Corrupted sent_articles.json backed up to %s: %s",
                backup_path,
                exc,
            )
        except OSError:
            logger.exception("Failed to backup corrupted sent_articles.json")

        return []


def _save_entries(entries: List[Dict]) -> None:
    """Atomic write: write to .tmp, then rename.

    Prevents partial/corrupted writes if the process crashes mid-write.
    """
    tmp_path = _JSON_PATH + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, _JSON_PATH)  # atomic on POSIX; near-atomic on Windows
        logger.debug("Saved %d entry(ies) to sent_articles.json", len(entries))
    except OSError:
        logger.exception("Failed to write sent_articles.json")
        # Clean up temp file on failure
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


# ═══════════════════════════════════════════════════════════════════
#  Public API — dedup checks
# ═══════════════════════════════════════════════════════════════════


def get_sent_normalized_urls() -> Set[str]:
    """Return set of normalized URLs that have been sent."""
    return {normalize_url(e["url"]) for e in _load_entries() if e.get("url")}


def get_sent_title_keys() -> Set[str]:
    """Return set of title keys that have been sent."""
    return {_title_key(e.get("title", "")) for e in _load_entries()}


def is_duplicate(url: str, title: str = "") -> bool:
    """Check whether *url* or *title* matches any previously sent article.

    Returns True if EITHER the normalized URL or the title key matches
    an existing entry.  This catches URL changes while the title stays
    the same.
    """
    entries = _load_entries()
    nu = normalize_url(url)
    tk = _title_key(title)
    for e in entries:
        if nu and e.get("url") and normalize_url(e["url"]) == nu:
            return True
        if tk and e.get("title") and _title_key(e["title"]) == tk:
            return True
    return False


def mark_all_sent(articles: List[Dict]) -> None:
    """Record article URLs + titles as sent.

    *articles* — list of dicts, each must contain at least ``"url"``
    and optionally ``"title"``.

    Dedup key is the combination of normalized URL + title key.
    An entry is skipped only if an identical key already exists.
    """
    entries = _load_entries()

    existing_keys: Set[str] = set()
    for e in entries:
        key = _dedup_key(e.get("url", ""), e.get("title", ""))
        existing_keys.add(key)

    new_count = 0
    for art in articles:
        url = art.get("url", "")
        if not url:
            continue
        title = art.get("title", "")
        key = _dedup_key(url, title)
        if key in existing_keys:
            continue
        entries.append({"url": url, "title": title})
        existing_keys.add(key)
        new_count += 1

    if new_count:
        _save_entries(entries)
        logger.info("Recorded %d new article(s) as sent", new_count)
    else:
        logger.info("All %d article(s) already recorded", len(articles))


# ═══════════════════════════════════════════════════════════════════
#  Backward-compat shim  (used by older callers that only check URL)
# ═══════════════════════════════════════════════════════════════════

def get_sent_urls() -> Set[str]:
    """DEPRECATED — use get_sent_normalized_urls() or is_duplicate().

    Returns raw (unnormalized) URLs for backward compatibility.
    """
    return {e["url"] for e in _load_entries() if e.get("url")}


def mark_sent(url: str) -> None:
    """DEPRECATED — single-URL record kept for compatibility."""
    mark_all_sent([{"url": url, "title": ""}])


# ═══════════════════════════════════════════════════════════════════
#  Source rotation tracking  (ensures every source is served ≥1×/week)
# ═══════════════════════════════════════════════════════════════════

def _load_rotation() -> Dict[str, str]:
    """Load the {source_name: last_push_date} mapping."""
    if not os.path.exists(_ROTATION_PATH):
        return {}
    try:
        with open(_ROTATION_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_rotation(data: Dict[str, str]) -> None:
    """Atomic write of rotation data."""
    tmp_path = _ROTATION_PATH + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, _ROTATION_PATH)
        logger.debug("Saved rotation data (%d source(s))", len(data))
    except OSError:
        logger.exception("Failed to write source_rotation.json")
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


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
