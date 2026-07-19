"""
Configuration management - reads from environment variables.
"""
import os
import random
from datetime import date
from typing import List, Dict
from dotenv import load_dotenv

# Load .env file in the same directory (if it exists)
load_dotenv()

# ---------------------------------------------------------------------------
# DeepSeek API (OpenAI-compatible)
# ---------------------------------------------------------------------------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# ---------------------------------------------------------------------------
# SMTP (QQ Mail)
# ---------------------------------------------------------------------------
SMTP_HOST: str = os.getenv("SMTP_HOST", "smtp.resend.com")
SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
TO_EMAIL: str = os.getenv("TO_EMAIL", "")
FROM_EMAIL: str = os.getenv("FROM_EMAIL", "")

# ---------------------------------------------------------------------------
# RSS feed sources — 用回你原来的官方 RSS（稳定可靠）
# ---------------------------------------------------------------------------
ALL_FEEDS: List[Dict[str, str]] = [
    {"name": "MIT Technology Review",  "url": "https://www.technologyreview.com/feed/"},
    {"name": "ScienceDaily",           "url": "https://www.sciencedaily.com/rss/all.xml"},
    {"name": "The Conversation",       "url": "https://theconversation.com/us/feed"},
    {"name": "The Guardian",           "url": "https://www.theguardian.com/us-news/rss"},
    {"name": "BBC News",               "url": "https://feeds.bbci.co.uk/news/rss.xml"},
    {"name": "New Scientist",          "url": "https://www.newscientist.com/feed/home"},
    {"name": "Phys.org",               "url": "https://phys.org/rss-feed/"},
]

# 每天从7个中随机选2个（用日期做种子，当天固定，不重复）
random.seed(str(date.today()))
RSS_FEEDS: List[Dict[str, str]] = random.sample(ALL_FEEDS, 2)

print(f"📰 今日推送源：{[f['name'] for f in RSS_FEEDS]}")

# ---------------------------------------------------------------------------
# IMAP (for deleting old emails from the server)
# ---------------------------------------------------------------------------
IMAP_HOST: str = "imap.qq.com"
IMAP_PORT: int = 993
CLEANUP_AFTER_DAYS: int = 3
CLEANUP_SEARCH_SUBJECT: str = "Daily Reader"

# ---------------------------------------------------------------------------
# Behaviour tweaks
# ---------------------------------------------------------------------------
MAX_ARTICLES_PER_FEED: int = 1
MAX_ARTICLES_TOTAL: int = 2
