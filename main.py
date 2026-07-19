#!/usr/bin/env python3
"""
Daily Reader – fetch articles, translate, extract vocabulary, and
optionally send a beautiful HTML newsletter via email.

Usage
-----
python main.py               # dry-run: print to console
python main.py --send        # actually send email
python main.py --server      # run as HTTP server (for Railway Cron)

Railway Cron 设置:
  Base URL:  https://你的项目.up.railway.app/cron/trigger
  Cron Timer: 0 1 * * *      (UTC 1:00 = 北京 9:00)
"""

import argparse
import logging
import sys
import os
from datetime import date

import requests
from bs4 import BeautifulSoup

from flask import Flask

import config
from fetcher import fetch_articles
from ai_processor import process_article
from email_builder import build_email, build_empty_email
from sender import send_email, _print_to_console
from cleanup import cleanup_old_emails
from history import get_sent_urls, mark_all_sent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("daily-reader")

app = Flask(__name__)


# ⭐ 从原文URL提取所有<p>段落，保留段落结构
def fetch_all_paragraphs(url: str) -> str:
    """从原文URL提取所有<p>段落，双换行分隔，确保段落一个不落"""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "aside",
                         "noscript", "header", "form", "button", "iframe",
                         "svg", "img", "figure", "figcaption"]):
            tag.decompose()
        paragraphs = soup.find_all("p")
        return "\n\n".join(
            p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
        )
    except Exception:
        return ""


# ── 核心逻辑 ───────────────────────────────────────────────────
def run_pipeline(send_mode: bool = False):
    """Run the full fetch → process → build → send pipeline."""

    # ── 1. Fetch ───────────────────────────────────────────────
    logger.info("=" * 48)
    logger.info("  Step 1/4 — Fetching articles from RSS feeds")
    logger.info("=" * 48)

    sent_urls = get_sent_urls()
    if sent_urls:
        logger.info("Found %d previously sent article(s) in history", len(sent_urls))

    articles = fetch_articles(skip_urls=sent_urls)
    logger.info("Fetched %d new article(s)", len(articles))

    if not articles:
        logger.warning("No articles fetched – sending placeholder email.")
        today_str = date.today().strftime("%Y-%m-%d")
        subject = f"Daily Reader — {today_str} — 暂无新文章"
        html = build_empty_email(today_str)
        if send_mode:
            send_email(subject, html)
        else:
            _print_to_console(subject, html)
        return

    # ── 2. Process with DeepSeek ──────────────────────────────
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
        if send_mode:
            send_email(subject, html)
        else:
            _print_to_console(subject, html)
        return

    # ── 2.5 二次过滤 ─────────────────────────────────────────
    sent_urls = get_sent_urls()
    if sent_urls:
        before = len(processed)
        processed = [a for a in processed if a.get("url") not in sent_urls]
        skipped = before - len(processed)
        if skipped:
            logger.info("Secondary filter removed %d already-sent article(s)", skipped)

    if not processed:
        logger.warning("All articles already sent before – sending placeholder email.")
        today_str = date.today().strftime("%Y-%m-%d")
        subject = f"Daily Reader — {today_str} — 暂无新文章"
        html = build_empty_email(today_str)
        if send_mode:
            send_email(subject, html)
            cleanup_old_emails()
        else:
            _print_to_console(subject, html)
        return

    # ── 3. Build email ────────────────────────────────────────
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

    # ── 4. Send ────────────────────────────────────────────────
    if send_mode:
        logger.info("=" * 48)
        logger.info("  Step 4/4 — Sending email")
        logger.info("=" * 48)

        if send_email(subject, html):
            logger.info("=" * 48)
            logger.info("  Cleanup — deleting emails ≥%d days old", config.CLEANUP_AFTER_DAYS)
            logger.info("=" * 48)
            cleanup_old_emails()
            mark_all_sent([a["url"] for a in articles])
    else:
        logger.info("=" * 48)
        logger.info("  Step 4/4 — Dry-run (use --send to actually mail)")
        logger.info("=" * 48)
        _print_to_console(subject, html)

    logger.info("Done.")


# ── HTTP 接口（给 Railway Cron 用）─────────────────────────────
import threading

@app.route("/cron/trigger")
def cron_trigger():
    """Railway Cron 定时访问这个地址，触发发邮件"""
    logger.info("🔥【定时任务触发】开始执行完整流程...")
    thread = threading.Thread(target=run_pipeline, kwargs={"send_mode": True})
    thread.daemon = True
    thread.start()
    return "OK - 任务已启动", 200


@app.route("/health")
def health():
    """健康检查"""
    return "OK", 200


# ── 入口 ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Daily Reader – fetch articles, translate & learn vocabulary"
    )
    parser.add_argument(
        "--send",
        action="store_true",
        help="Really send the email (default: dry-run to console)",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Run as HTTP server for Railway",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", 8080)),
        help="Port for HTTP server (default: $PORT or 8080)",
    )
    args = parser.parse_args()

    if args.server:
        logger.info("🚀 启动 HTTP 服务，端口 %d ...", args.port)
        app.run(host="0.0.0.0", port=args.port)
    else:
        run_pipeline(send_mode=args.send)


if __name__ == "__main__":
    main()
