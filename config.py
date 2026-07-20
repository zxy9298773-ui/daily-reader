"""
Configuration management - reads from environment variables.
"""
import os
from typing import List, Dict
from dotenv import load_dotenv

# Load .env file in the same directory (if it exists)
load_dotenv()

# ---------------------------------------------------------------------------
# DeepSeek API (OpenAI-compatible)
# ---------------------------------------------------------------------------
DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# ---------------------------------------------------------------------------
# SMTP (QQ Mail)
# ---------------------------------------------------------------------------
SMTP_HOST: str = "smtp.qq.com"
SMTP_PORT: int = 465
SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
TO_EMAIL: str = os.getenv("TO_EMAIL", "")
FROM_EMAIL: str = os.getenv("FROM_EMAIL", "")

# ---------------------------------------------------------------------------
# RSS feed sources
# ---------------------------------------------------------------------------
RSS_FEEDS: List[Dict[str, str]] = [
    {"name": "MIT Technology Review",  "url": "https://www.technologyreview.com/feed/"},
    {"name": "ScienceDaily",           "url": "https://www.sciencedaily.com/rss/all.xml"},
    {"name": "The Conversation",       "url": "https://theconversation.com/us/feed"},
    {"name": "The Guardian",          "url": "https://www.theguardian.com/us-news/rss"},
    {"name": "BBC News",              "url": "https://feeds.bbci.co.uk/news/rss.xml"},
    {"name": "New Scientist",         "url": "https://www.newscientist.com/feed/home"},
    {"name": "Phys.org",              "url": "https://phys.org/rss-feed/"},
    {"name": "The Marginalian",      "url": "https://www.themarginalian.org/feed/"},
    {"name": "Aeon Magazine",        "url": "https://aeon.co/feed.rss"},
    {"name": "Literary Hub",         "url": "https://lithub.com/feed/"},
    {"name": "Electric Literature",  "url": "https://electricliterature.com/feed/"},
    {"name": "Space.com",            "url": "https://www.space.com/feeds/all"},
]

# ---------------------------------------------------------------------------
# IMAP (for deleting old emails from the server)
# ---------------------------------------------------------------------------
IMAP_HOST: str = "imap.qq.com"
IMAP_PORT: int = 993
# ── cleanup ─────────────────────────────────────────────────────────
CLEANUP_AFTER_DAYS: int = 3        # auto-delete emails older than this
CLEANUP_SEARCH_SUBJECT: str = "Daily Reader"   # only delete matching emails

# ---------------------------------------------------------------------------
# Behaviour tweaks
# ---------------------------------------------------------------------------
MAX_ARTICLES_PER_FEED: int = 1   # articles per source (iterates until one succeeds)
MAX_ARTICLES_TOTAL: int = 2      # cap on total articles per run
