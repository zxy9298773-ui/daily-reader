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
# RSS feed sources — 使用 RSSHub 增强版，输出完整段落
# ---------------------------------------------------------------------------
ALL_FEEDS: List[Dict[str, str]] = [
    {"name": "MIT Technology Review",  "url": "https://rsshub.app/mit/technologyreview"},
    {"name": "ScienceDaily",           "url": "https://rsshub.app/sciencedaily"},
    {"name": "The Conversation",       "url": "https://rsshub.app/theconversation/us"},
    {"name": "The Guardian",           "url": "https://rsshub.app/guardian/uk"},
    {"name": "BBC News",               "url": "https://rsshub.app/bbc"},
    {"name": "New Scientist",          "url": "https://rsshub.app/newscientist"},
    {"name": "Phys.org",               "url": "https://rsshub.app/phys"},
]

# ⭐ 每天从7个源中随机选2个（用日期做种子，当天结果固定）
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
# Behaviour tweaks — 保持你原来的数量设置
# ---------------------------------------------------------------------------
MAX_ARTICLES_PER_FEED: int = 1   # 每个源取最新1篇
MAX_ARTICLES_TOTAL: int = 2      # 每天总共2篇（2个源 × 1篇）
