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
from email_builder import build_email
from sender import send_email, _print_to_console
from cleanup import cleanup_old_emails
from history import get_sent_urls, mark_all_sent

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
    sent_urls = get_sent_urls()
    if sent_urls:
        logger.info("Found %d previously sent article(s) in history", len(sent_urls))

    articles = fetch_articles(skip_urls=sent_urls)
    logger.info("Fetched %d new article(s)", len(articles))

    if not articles:
        logger.warning("No articles fetched – exiting.")
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
        logger.warning("No articles processed – exiting.")
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
            # Record sent URLs so they won't repeat tomorrow
            mark_all_sent([a["url"] for a in articles])
    else:
        logger.info("=" * 48)
        logger.info("  Step 4/4 — Dry-run (use --send to actually mail)")
        logger.info("=" * 48)
        _print_to_console(subject, html)

    logger.info("Done.")


if __name__ == "__main__":
    main()
