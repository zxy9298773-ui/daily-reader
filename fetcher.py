"""
Fetch articles from configured RSS feeds using feedparser + multi-strategy extraction.
Skips articles that are too short (likely paywalled or broken).
"""
import re
import feedparser
from typing import List, Dict, Optional
import logging

import config
from history import get_sent_urls

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Text cleaning — remove junk before paragraph splitting
# ---------------------------------------------------------------------------

# Patterns that indicate junk lines (case-insensitive)
_JUNK_PATTERNS = [
    r"support\s+(us|our|the\s+.*work)",
    r"donate|donation",
    r"subscribe|newsletter|\bsign\s*[ui]p?\b",
    r"follow\s+us",
    r"advertisement|sponsored?",
    r"sponsor\s+(message|content|story)",
    r"read\s+more|related\s+(stories|articles|content|topics|reading|links|coverage)",
    r"©|copyright|all\s+rights\s+reserved",
    r"terms\s+of\s+service|privacy\s+policy",
    r"this\s+article\s+was\s+originally\s+published",
    r"you\s+may\s+also\s+like",
    r"what\s+to\s+read\s+next",
    r"your\s+(support|donation).*make\s+a\s+difference",
    r"become\s+a\s+(member|subscriber|supporter)",
    r"already\s+(a\s+)?(member|subscriber)",
    r"click\s+here",
    r"editor'?s?\s*(note|pick)",
    r"share\s+this|share\s+on",
    r"more\s+on\s+",
    r"related\s+(topics|reading|links|coverage)",
    r"photograph(y|ed|er)?\s+by|image\s+(credit|by|via)",
    r"you\s+might\s+also|in\s+this\s+article",
    r"top\s+stories|must\s+read|trending",
    r"most\s+(popular|read|viewed|shared)",
    r"comments?\s+(are\s+)?(closed|disabled)",
    r"external\s+(link|site|links)",
    r"load\s+more|show\s+more",
    r"the\s+latest\s+(news|updates|stories|headlines)",
    r"updates?\s+and\s+(analysis|coverage)",
]


def _is_junk_line(line: str) -> bool:
    """Return True if *line* looks like junk (sponsor, nav, copyright, etc.)."""
    stripped = line.strip()
    if not stripped:
        return True
    # Very short lines (≤ 15 chars) without sentence-ending punctuation
    if len(stripped) <= 15 and not stripped.rstrip().endswith((".", "!", "?", ":", "”", '"', "。")):
        return True
    # Match junk patterns
    for pat in _JUNK_PATTERNS:
        if re.search(pat, stripped, re.IGNORECASE):
            return True
    return False


def _clean_text(raw: str) -> str:
    """Remove junk lines from extracted article text.

    Strategies:
      1. Strip each line; discard empty / junk lines.
      2. Group consecutive non-junk lines into paragraphs (double newline).
      3. Within a paragraph, rejoin mid-wrapped lines into a single line.
    """
    # Split the raw text by double-newlines to identify paragraph boundaries
    paragraphs_raw = re.split(r"\n\s*\n", raw)
    clean_paragraphs = []

    for block in paragraphs_raw:
        block_lines = [l.strip() for l in block.split("\n") if l.strip()]
        # Filter out junk lines within the block
        good_lines = [l for l in block_lines if not _is_junk_line(l)]
        if not good_lines:
            continue
        # Rejoin into a single paragraph
        paragraph = " ".join(good_lines)
        # Remove extra whitespace
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        # No length filter — any paragraph with substantive content is kept.
        # The _is_truncated() check at a higher level catches entire
        # articles that are too short / junk-heavy.
        clean_paragraphs.append(paragraph)

    return "\n\n".join(clean_paragraphs)


# ---------------------------------------------------------------------------
#  Truncation detection — reject preview / paywalled articles
# ---------------------------------------------------------------------------

def _is_truncated(cleaned: str) -> bool:
    """Return True if the article appears to be a preview (cut off mid-way)."""
    paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]

    # A real article should have at least 3 substantial paragraphs
    long_paragraphs = [p for p in paragraphs if len(p) >= 100]
    if len(long_paragraphs) < 3:
        logger.debug("Article truncated: only %d substantial paragraphs", len(long_paragraphs))
        return True

    # Total cleaned text should be reasonably long
    if len(cleaned) < 800:
        logger.debug("Article truncated: only %d chars after cleaning", len(cleaned))
        return True

    # At least 40 % of paragraphs should be substantial (≥100 chars)
    # (lowered from 50% because _clean_text no longer filters short paragraphs)
    if paragraphs:
        substantial_ratio = len(long_paragraphs) / len(paragraphs)
        if substantial_ratio < 0.4:
            logger.debug(
                "Article truncated: only %.0f%% of paragraphs are substantial (%d/%d)",
                substantial_ratio * 100,
                len(long_paragraphs),
                len(paragraphs),
            )
            return True

    # Average paragraph length check — reject topic-listing pages
    # A real article has paragraphs averaging ≥120 chars; a listing
    # of topic headlines / snippets averages well below that.
    avg_para_len = sum(len(p) for p in paragraphs) / len(paragraphs)
    if avg_para_len < 120:
        logger.debug(
            "Article rejected: avg paragraph length only %.0f chars (likely a listing page)",
            avg_para_len,
        )
        return True

    # ── mid-article truncation signal cross-check ──────────────────
    # Only check the CLEANED text.  If _clean_text already removed the
    # "subscribe" / "continue reading" junk lines, the signal won't be
    # present — and the article is fine.  A signal that *survives*
    # cleaning is genuinely in the article body.
    mid_article_signals = [
        r"to\s+continue\s+reading",
        r"become\s+a\s+(member|subscriber|supporter)",
        r"read\s+without\s+(limits|paywall)",
        r"unlock\s+(this\s+)?article",
    ]
    for pat in mid_article_signals:
        if re.search(pat, cleaned, re.IGNORECASE):
            logger.debug(
                "Article truncated: cleaned text contains '%s'",
                pat,
            )
            return True

    # The last paragraph should end with sentence-ending punctuation
    last_para = paragraphs[-1].strip()
    if not last_para:
        return True
    if not last_para[-1] in (".", "!", "?", '"', "。", "！", "？", "”"):
        # Allow closing parenthesis after punctuation, e.g. "... (Nature)."
        if not re.search(r'[.!?"。！？」][)）]?\s*$', last_para):
            logger.debug("Article truncated: last paragraph doesn't end with punctuation: %s", last_para[-30:])
            return True

    # Reject if last paragraph contains truncation signals
    for pat in mid_article_signals:
        if re.search(pat, last_para, re.IGNORECASE):
            logger.debug("Article truncated: last paragraph contains '%s'", pat)
            return True

    return False


# ---------------------------------------------------------------------------
#  Roundup / newsletter detection — reject multi-topic articles
# ---------------------------------------------------------------------------

def _is_roundup_article(title: str, cleaned_text: str) -> bool:
    """Return True if the article is a multi-topic roundup / newsletter.

    Roundups piece together several unrelated stories and should be
    skipped — they violate the "one article, one topic" rule.
    """
    title_lower = title.lower()

    # ── Title signals ──────────────────────────────────────────────
    roundup_title_keywords = [
        "the download", "daily briefing", "daily roundup", "the newsletter",
        "daily digest", "this week in", "weekly wrap", "morning briefing",
        "today's top", "your daily", "in case you missed",
        "5 things", "5 stories", "top stories", "editors' picks",
        "highlights from", "what to watch", "best of",
    ]
    for kw in roundup_title_keywords:
        if kw in title_lower:
            logger.debug("Roundup detected by title keyword '%s': %s", kw, title[:60])
            return True

    # Colon + "and" pattern: "The Download: X and Y" signals 2+ topics
    if re.search(r":\s*\w+\s+and\s+\w+", title):
        logger.debug("Roundup suspected by 'and' pattern in title: %s", title[:60])
        return True

    # ── Text signals ───────────────────────────────────────────────
    # Multiple "+ " list markers (e.g. "+ Soccer... + A cosmic... + Sir David...")
    plus_markers = re.findall(r"(?:^|\s)\+ [A-Z]", cleaned_text)
    if len(plus_markers) >= 3:
        logger.debug(
            "Roundup detected: %d '+ ' list markers in text", len(plus_markers)
        )
        return True

    # Multiple "•" bullet markers
    bullet_count = cleaned_text.count("•")
    if bullet_count >= 3:
        logger.debug(
            "Roundup detected: %d bullet markers in text", bullet_count
        )
        return True

    return False


# ---------------------------------------------------------------------------
#  Multi-strategy text extraction
# ---------------------------------------------------------------------------

def extract_text(url: str) -> Optional[str]:
    """Try multiple strategies to extract article text from *url*."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/120.0.0.0 Safari/537.36',
    }

    best_raw: str | None = None

    # ── Strategy 1: newspaper3k ────────────────────────────────────
    try:
        from newspaper import Article
        article = Article(url)
        article.download()
        article.parse()
        if len(article.text) > 200:
            cleaned = _clean_text(article.text)
            if cleaned and len(cleaned) >= 800 and not _is_truncated(cleaned):
                return cleaned
            if not best_raw or len(article.text) > len(best_raw):
                best_raw = article.text
    except Exception:
        pass

    # ── Strategy 2: readability-lxml ───────────────────────────────
    try:
        from readability import Document
        import requests
        resp = requests.get(url, headers=headers, timeout=10)
        doc = Document(resp.text)
        text = doc.summary()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text(separator='\n')
        if len(text) > 200:
            cleaned = _clean_text(text)
            if cleaned and len(cleaned) >= 800 and not _is_truncated(cleaned):
                return cleaned
            if not best_raw or len(text) > len(best_raw):
                best_raw = text
    except Exception:
        pass

    # ── Strategy 3: direct requests + BeautifulSoup ────────────────
    try:
        import requests
        resp = requests.get(url, headers=headers, timeout=10)
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, 'html.parser')
        for tag in ['article', 'main', '.post-content', '.article-body', '.story-body']:
            el = soup.select_one(tag)
            if el:
                text = el.get_text(separator='\n')
                if len(text) > 200:
                    cleaned = _clean_text(text)
                    if cleaned and len(cleaned) >= 800 and not _is_truncated(cleaned):
                        return cleaned
                    if not best_raw or len(text) > len(best_raw):
                        best_raw = text
    except Exception:
        pass

    # Last resort: try the longest raw text through cleaning once more
    if best_raw:
        cleaned = _clean_text(best_raw)
        if cleaned and len(cleaned) >= 800 and not _is_truncated(cleaned):
            return cleaned

    return None


# ---------------------------------------------------------------------------
#  Article extraction wrapper
# ---------------------------------------------------------------------------

def _strip_html(text: str) -> str:
    """Remove HTML tags from *text* and collapse whitespace."""
    import html as html_mod
    text = re.sub(r"<[^>]+>", "", text)
    text = html_mod.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_article(entry, source_name: str) -> Optional[Dict]:
    """Try to extract text + metadata from a single RSS entry.

    Priority:
      1. Full article text via ``extract_text()`` (newspaper3k / …)
      2. RSS summary / description as fallback (marked ``is_summary``)

    Returns ``None`` only when both strategies fail.
    """
    url = entry.get("link", "")
    if not url:
        return None

    text = extract_text(url)
    is_summary = False

    if not text:
        # Fallback: use RSS summary
        summary = entry.get("summary", "") or entry.get("description", "")
        if not summary:
            logger.debug("No summary available for: %s", url)
            return None
        summary = _strip_html(summary)
        if len(summary) < 50:
            logger.debug("RSS summary too short (%d chars): %s", len(summary), url)
            return None
        text = summary
        is_summary = True
        logger.info("  Using RSS summary (%d chars) as fallback for: %s", len(summary), url[:60])

    # Roundup / newsletter check — reject multi-topic articles
    title = entry.get("title", "")
    if _is_roundup_article(title, text):
        logger.debug("Rejected roundup article: %s", title[:60])
        return None

    return {
        "title": entry.get("title", ""),
        "url": url,
        "author": [],
        "published": entry.get("published", ""),
        "text": text,
        "source": source_name,
        "summary": entry.get("summary", ""),
        "top_image": "",
        "is_summary": is_summary,
    }


# ---------------------------------------------------------------------------
#  Top-level fetch
# ---------------------------------------------------------------------------

def fetch_articles(skip_urls: set[str] | None = None) -> List[Dict]:
    """Return up to ``MAX_ARTICLES_TOTAL`` parsed-article dicts.

    Two-pass strategy:
      1. **Diversity pass** — take at most **one** article from each
         feed so the newsletter contains sources from different outlets.
      2. **Fallback pass** — if not enough articles after pass 1, go
         back to feeds that already gave us an article and take more.

    Respects the ``skip_urls`` set (previously pushed articles are
    skipped entirely without attempting extraction).
    """
    skip_urls = skip_urls or set()
    articles: List[Dict] = []
    used_urls: set[str] = set()   # tracks URLs taken in pass 1 / skip
    failed_urls: set[str] = set()  # tracks URLs that failed extraction

    # Parse all feeds up front so we only fetch each RSS URL once
    feed_buckets: list[dict] = []  # {"name": str, "url": str, "entries": [...]}
    for feed in config.RSS_FEEDS:
        logger.info("Parsing feed: %s …", feed["name"])
        try:
            parsed = feedparser.parse(feed["url"])
            if not parsed.entries:
                logger.debug("  No entries in %s", feed["name"])
                continue
            feed_buckets.append({
                "name": feed["name"],
                "url": feed["url"],
                "entries": list(parsed.entries),
            })
        except Exception:
            logger.exception("Failed to parse RSS feed: %s", feed["name"])

    if not feed_buckets:
        logger.warning("Could not parse any RSS feeds")
        return []

    # ── Pass 1: diversity (≤1 per feed) ────────────────────────────
    for bucket in feed_buckets:
        if len(articles) >= config.MAX_ARTICLES_TOTAL:
            break

        taken_one = False
        for entry in bucket["entries"]:
            if len(articles) >= config.MAX_ARTICLES_TOTAL:
                break

            url = entry.get("link", "")
            if not url or url in skip_urls or url in used_urls:
                continue

            article = _extract_article(entry, bucket["name"])
            if article:
                articles.append(article)
                used_urls.add(url)
                taken_one = True
                logger.info(
                    "  [diversity] article %d/%d: %s",
                    len(articles), config.MAX_ARTICLES_TOTAL,
                    article["title"],
                )
                break  # max 1 per feed in pass 1
            else:
                failed_urls.add(url)  # don't retry in pass 2

        if not taken_one:
            logger.debug("  No valid article from %s (pass 1)", bucket["name"])

    # ── Pass 2: fallback (fill remaining slots from any feed) ─────
    if len(articles) < config.MAX_ARTICLES_TOTAL:
        needed = config.MAX_ARTICLES_TOTAL - len(articles)
        logger.info(
            "Only %d article(s) after diversity pass, "
            "fallback: need %d more …",
            len(articles), needed,
        )

        for bucket in feed_buckets:
            if len(articles) >= config.MAX_ARTICLES_TOTAL:
                break
            for entry in bucket["entries"]:
                if len(articles) >= config.MAX_ARTICLES_TOTAL:
                    break

                url = entry.get("link", "")
                if not url or url in skip_urls or url in used_urls or url in failed_urls:
                    continue

                article = _extract_article(entry, bucket["name"])
                if article:
                    articles.append(article)
                    used_urls.add(url)
                    logger.info(
                        "  [fallback] article %d/%d: %s",
                        len(articles), config.MAX_ARTICLES_TOTAL,
                        article["title"],
                    )

    # ── Pass 3: 源端过滤已发送文章 ────────────────────────────────
    sent_urls = get_sent_urls()
    if sent_urls:
        before = len(articles)
        articles = [a for a in articles if a["url"] not in sent_urls]
        filtered = before - len(articles)
        if filtered:
            logger.info("Source filter removed %d already-sent article(s)", filtered)

    if not articles:
        logger.warning("No articles could be extracted from any feed")

        # ── Link-list fallback ────────────────────────────────────
        links: list[dict] = []
        for bucket in feed_buckets:
            for entry in bucket["entries"]:
                url = entry.get("link", "")
                title = entry.get("title", "")
                if url and title and url not in skip_urls:
                    links.append({"title": title, "url": url})

        if links:
            logger.info(
                "Fallback: building link list with %d entries",
                len(links),
            )
            return [{
                "title": f"{len(links)} Stories from {len(feed_buckets)} Sources",
                "url": "",
                "author": [],
                "published": "",
                "text": "",
                "source": "",
                "summary": "",
                "top_image": "",
                "is_link_list": True,
                "link_entries": links[:15],
                "paragraphs": [],
                "vocabulary": [],
            }]

    return articles
