#!/usr/bin/env python3
"""
Daily Reader – fetch articles, translate, extract vocabulary, and
optionally send a beautiful HTML newsletter via email.

Usage
-----
    python main.py               # dry-run: print to console
    python main.py --send        # actually send email
"""
import argparse
import logging
import sys
from datetime import date

import config
from fetcher import fetch_articles
from ai_processor import process_article
from email_builder import build_email, build_empty_email
from sender import send_email, _print_to_console
from cleanup import cleanup_old_emails
from history import (
    get_sent_normalized_urls,
    mark_all_sent,
    mark_sources_pushed,
    normalize_url,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("daily-reader")


def main():
    parser = argparse.ArgumentParser(
        description="Daily Reader – translate articles & learn vocabulary"
    )
    parser.add_argument(
        "--send",
        action="store_true",
        help="Really send the email (default: dry-run to console)",
    )
    args = parser.parse_args()

    # ── 1. Fetch ───────────────────────────────────────────────────
    logger.info("=" * 48)
    logger.info("  Step 1/4 — Fetching articles from RSS feeds")
    logger.info("=" * 48)

    # Load previously sent URLs so we don't repeat articles
    sent_urls = get_sent_normalized_urls()
    if sent_urls:
        logger.info("Found %d previously sent article(s) in history", len(sent_urls))

    articles = fetch_articles(skip_urls=sent_urls)
    logger.info("Fetched %d new article(s)", len(articles))

    if not articles:
        logger.warning("No articles fetched – sending placeholder email.")
        today_str = date.today().strftime("%Y-%m-%d")
        subject = f"Daily Reader — {today_str} — 暂无新文章"
        html = build_empty_email(today_str)
        if args.send:
            send_email(subject, html)
        else:
            _print_to_console(subject, html)
        sys.exit(0)

    # ── 2. Process with DeepSeek ──────────────────────────────────
    logger.info("=" * 48)
    logger.info("  Step 2/4 — Processing articles with DeepSeek AI")
    logger.info("=" * 48)

    processed = []
    for i, art in enumerate(articles, 1):
        title = art.get("title", "")[:60]
        logger.info("  [%d/%d] %s …", i, len(articles), title)
        try:
            result = process_article(art)
            processed.append(result)
        except Exception:
            logger.exception("Failed to process: %s", title)

    if not processed:
        logger.warning("No articles processed – sending placeholder email.")
        today_str = date.today().strftime("%Y-%m-%d")
        subject = f"Daily Reader — {today_str} — 暂无新文章"
        html = build_empty_email(today_str)
        if args.send:
            send_email(subject, html)
        else:
            _print_to_console(subject, html)
        sys.exit(0)

    # ── 2.5 二次过滤：发送前再检查一遍已发送记录 ────────────────
    sent_urls = get_sent_normalized_urls()
    if sent_urls:
        before = len(processed)
        processed = [a for a in processed if normalize_url(a.get("url", "")) not in sent_urls]
        skipped = before - len(processed)
        if skipped:
            logger.info("Secondary filter removed %d already-sent article(s)", skipped)

    if not processed:
        logger.warning("All articles already sent before – sending placeholder email.")
        today_str = date.today().strftime("%Y-%m-%d")
        subject = f"Daily Reader — {today_str} — 暂无新文章"
        html = build_empty_email(today_str)
        if args.send:
            send_email(subject, html)
            cleanup_old_emails()
        else:
            _print_to_console(subject, html)
        sys.exit(0)

    # ── 3. Build email ────────────────────────────────────────────
    logger.info("=" * 48)
    logger.info("  Step 3/4 — Building HTML email")
    logger.info("=" * 48)

    today_str = date.today().strftime("%Y-%m-%d")
    first = processed[0]
    if first.get("is_link_list"):
        subject = f"Daily Reader — {today_str} — Link List"
    else:
        subject = f"Daily Reader — {today_str} — {first['source']} & more"
    html = build_email(processed, date_str=today_str)

    # ── 4. Send (or print) ────────────────────────────────────────
    if args.send:
        logger.info("=" * 48)
        logger.info("  Step 4/4 — Sending email")
        logger.info("=" * 48)
        if send_email(subject, html):
            logger.info("=" * 48)
            logger.info("  Cleanup — deleting emails ≥%d days old", config.CLEANUP_AFTER_DAYS)
            logger.info("=" * 48)
            cleanup_old_emails()
            # Record sent articles so they won't repeat tomorrow
            # (dedup uses both URL and title; title hash catches URL changes)
            mark_all_sent(processed)
            # Record source rotation so every source gets a turn
            mark_sources_pushed([a["source"] for a in processed], today_str)
    else:
        logger.info("=" * 48)
        logger.info("  Step 4/4 — Dry-run (use --send to actually mail)")
        logger.info("=" * 48)
        _print_to_console(subject, html)

    logger.info("Done.")


if __name__ == "__main__":
    main()
