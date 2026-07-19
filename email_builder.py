"""
Build a beautiful, email-client-compatible HTML newsletter.

Layout
------
+------------------------------------------------------------------+
|  Daily Reader                         3 articles · 12 words      |
+------------------------------------------------------------------+
|                                                                  |
|  +---------------------------+--------------------------------+  |
|  |  Left column (70%)       |  Right column (30%)            |  |
|  |  Original paragraph      |  Vocabulary cards              |  |
|  |  _Translation (gray it.) |  ┌──────────────────────┐      |  |
|  |                           |  │ word  /ˈfɒn.ɛt.ɪk/  │      |  |
|  |  Original paragraph      |  │ noun                 │      |  |
|  |  _Translation (gray it.) |  │ 中文释义             │      |  |
|  |                           |  │ "example sentence"   │      |  |
|  |  …                       |  └──────────────────────┘      |  |
|  +---------------------------+--------------------------------+  |
+------------------------------------------------------------------+

Highlights vocabulary words in the original text with a pale yellow
background.  Uses *only* table-based layout for Gmail / QQ / Outlook
compatibility.
"""
import re
import html as html_mod
from typing import List, Dict

import logging

logger = logging.getLogger(__name__)

# ── colour palette (Medium inspired) ──────────────────────────────
BG_PAGE = "#f4f4f4"
BG_CARD = "#ffffff"
TEXT_PRIMARY = "#1a1a1a"
TEXT_BODY = "#292929"
TEXT_MUTED = "#888888"
TEXT_META = "#aaaaaa"
ACCENT_GREEN = "#1a8917"
HIGHLIGHT = "#fff9db"  # pale yellow
BORDER_LIGHT = "#f0f0f0"
CARD_BG = "#fafafa"
CARD_BORDER = "#eeeeee"
BTN_GREEN = "#1a8917"  # "read full article" button

# ── font stacks ───────────────────────────────────────────────────
FONT_EN = "Georgia,'Times New Roman',Times,serif"
FONT_CN = "'PingFang SC','Microsoft YaHei','Hiragino Sans GB','Noto Sans SC',sans-serif"
FONT_BODY = f"-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,{FONT_CN}"
# ───────────────────────────────────────────────────────────────────


def build_email(articles: List[Dict], date_str: str = "") -> str:
    """Assemble the full HTML newsletter from *articles*."""
    articles_html = "\n".join(
        _article_section(a) for a in articles
    )

    total_vocab = sum(len(a.get("vocabulary", [])) for a in articles)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Daily Reader — {date_str}</title>
</head>
<body style="margin:0;padding:0;background-color:{BG_PAGE};font-family:{FONT_BODY};">
<table width="100%" cellpadding="0" cellspacing="0" style="background-color:{BG_PAGE};">
<tr><td align="center" style="padding:30px 15px;">

  <!-- ─────────── outer container ─────────── -->
  <!-- 🔧 修改 4：去掉 overflow:hidden，避免邮件客户端裁剪链接点击区域 -->
  <table width="640" cellpadding="0" cellspacing="0" style="max-width:640px;background-color:{BG_CARD};border-radius:12px;box-shadow:0 1px 4px rgba(0,0,0,.04);">

    <!-- header -->
    <tr><td style="padding:36px 40px 20px;border-bottom:1px solid {BORDER_LIGHT};">
      <table width="100%" cellpadding="0" cellspacing="0"><tr>
        <td>
          <h1 style="margin:0;font-size:24px;font-weight:700;color:{TEXT_PRIMARY};letter-spacing:-.5px;font-family:Georgia,'Times New Roman',Times,serif;">Daily Reader</h1>
          <p style="margin:6px 0 0;font-size:13px;color:{TEXT_MUTED};">{date_str} · Your daily dose of reading &amp; learning</p>
        </td>
        <td align="right" valign="top" style="font-size:13px;color:{TEXT_META};white-space:nowrap;">
          {len(articles)} article{"s" if len(articles) != 1 else ""} · {total_vocab} word{(total_vocab != 1) * "s"}
        </td>
      </tr></table>
    </td></tr>

    <!-- articles -->
    {articles_html}

    <!-- footer -->
    <tr><td style="padding:24px 40px;border-top:1px solid {BORDER_LIGHT};">
      <p style="margin:0;font-size:12px;color:#bbb;text-align:center;">
        Daily Reader · Powered by DeepSeek &nbsp;·&nbsp;
        <a href="#" style="color:{ACCENT_GREEN};text-decoration:none;">Unsubscribe</a>
      </p>
    </td></tr>

  </table>
  <!-- ─────────── end container ─────────── -->

</td></tr></table>
</body>
</html>"""


def build_empty_email(date_str: str = "") -> str:
    """Build a short "no new articles today" placeholder HTML."""
    if not date_str:
        from datetime import date
        date_str = date.today().strftime("%Y-%m-%d")
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Daily Reader — {date_str} — 暂无新文章</title>
</head>
<body style="margin:0;padding:0;background-color:{BG_PAGE};font-family:{FONT_BODY};">
<table width="100%" cellpadding="0" cellspacing="0" style="background-color:{BG_PAGE};">
<tr><td align="center" style="padding:30px 15px;">

  <table width="640" cellpadding="0" cellspacing="0" style="max-width:640px;background-color:{BG_CARD};border-radius:12px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.04);">

    <!-- header -->
    <tr><td style="padding:36px 40px 20px;border-bottom:1px solid {BORDER_LIGHT};text-align:center;">
      <h1 style="margin:0;font-size:24px;font-weight:700;color:{TEXT_PRIMARY};letter-spacing:-.5px;font-family:Georgia,'Times New Roman',Times,serif;">Daily Reader</h1>
      <p style="margin:6px 0 0;font-size:13px;color:{TEXT_MUTED};">{date_str} · Your daily dose of reading &amp; learning</p>
    </td></tr>

    <!-- 暂无新文章提示 -->
    <tr><td style="padding:60px 40px;text-align:center;">
      <p style="margin:0 0 12px;font-size:15px;color:{TEXT_MUTED};line-height:1.6;">
        Today&rsquo;s RSS feeds haven&rsquo;t published anything new yet.
      </p>
      <p style="margin:0;font-size:14px;color:{TEXT_META};line-height:1.6;">
        今日 RSS 源暂无新内容。<br>
        Check back tomorrow for fresh articles.
      </p>
    </td></tr>

    <!-- footer -->
    <tr><td style="padding:24px 40px;border-top:1px solid {BORDER_LIGHT};">
      <p style="margin:0;font-size:12px;color:#bbb;text-align:center;">
        Daily Reader · Powered by DeepSeek
      </p>
    </td></tr>

  </table>

</td></tr></table>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════
#  per-article section
# ═══════════════════════════════════════════════════════════════════

def _article_section(article: Dict) -> str:
    title = _esc(article.get("title", ""))
    # 🔧 修改 2：用 _url_esc 替代 _esc，保留 & 不变
    raw_url = article.get("url") or ""
    url = _url_esc(raw_url)
    source = _esc(article.get("source", ""))
    author = ", ".join(article.get("author", [])) if article.get("author") else ""

    paragraphs = article.get("paragraphs", [])
    vocabulary = article.get("vocabulary", [])
    is_summary = article.get("is_summary", False)
    is_link_list = article.get("is_link_list", False)

    if is_link_list:
        # ── Link-list: show clickable titles from all feeds ──
        links = article.get("link_entries", [])
        links_html = "".join(
            # 🔧 修改 1+3：用 _url_esc + target="_blank"
            f'<tr><td style="padding:6px 0;font-size:14px;line-height:1.5;">'
            f'<a href="{_url_esc(link["url"])}" '
            f'target="_blank" rel="noopener noreferrer" '
            f'style="color:{ACCENT_GREEN};text-decoration:none;font-family:{FONT_EN};">'
            f'{_esc(link["title"])}</a></td></tr>'
            for link in links
        )
        source_note = (
            f"<p style=\"margin:6px 0 0;font-size:12px;color:{TEXT_META};\">"
            f"No full-text articles available today. "
            f"Click any link below to read the original story.</p>"
        )
        return f"""
    <!-- ── article: link list ── -->
    <tr><td style="padding:32px 40px 4px;">
      <h2 style="margin:0;font-size:20px;font-weight:700;color:{TEXT_PRIMARY};line-height:1.3;font-family:Georgia,'Times New Roman',Times,serif;">
        {_esc(title)}</h2>
      {source_note}
    </td></tr>
    <tr><td style="padding:12px 40px 32px;">
      <table width="100%" cellpadding="0" cellspacing="0">
        {links_html}
      </table>
    </td></tr>
    <tr><td style="padding:0 40px;">
      <table width="100%" cellpadding="0" cellspacing="0"><tr>
        <td style="height:1px;background:{BORDER_LIGHT};font-size:0;line-height:0;">&nbsp;</td>
      </tr></table>
    </td></tr>"""

    if is_summary:
        # ── Summary-only: full-width layout + "Read full article" button ──
        summary_text = _esc(paragraphs[0]["original"]) if paragraphs else ""

        return f"""
    <!-- ── article (summary): {title} ── -->
    <tr><td style="padding:32px 40px 4px;">
      <h2 style="margin:0;font-size:20px;font-weight:700;color:{TEXT_PRIMARY};line-height:1.3;font-family:Georgia,'Times New Roman',Times,serif;">
        <!-- 🔧 修改 3：加 target="_blank" -->
        <a href="{url}" target="_blank" rel="noopener noreferrer"
           style="color:{TEXT_PRIMARY};text-decoration:none;">{title}</a>
      </h2>
      <p style="margin:6px 0 0;font-size:12px;color:{TEXT_META};">{source}{" · " + author if author else ""}</p>
    </td></tr>

    <tr><td style="padding:20px 40px 32px;">
      <p style="margin:0;font-size:15px;line-height:1.7;color:{TEXT_BODY};font-family:{FONT_EN};">{summary_text}</p>
    </td></tr>

    <tr><td style="padding:0 40px 32px;">
      <table cellpadding="0" cellspacing="0" style="margin:0;">
        <tr>
          <!-- 🔧 修改 3：加 target="_blank" -->
          <td style="background:{BTN_GREEN};border-radius:6px;text-align:center;">
            <a href="{url}" target="_blank" rel="noopener noreferrer"
               style="display:inline-block;padding:10px 24px;font-size:14px;font-weight:600;color:#fff;text-decoration:none;font-family:{FONT_CN};">
              Read full article on {source} →
            </a>
          </td>
        </tr>
      </table>
    </td></tr>

    <tr><td style="padding:0 40px;">
      <table width="100%" cellpadding="0" cellspacing="0"><tr>
        <td style="height:1px;background:{BORDER_LIGHT};font-size:0;line-height:0;">&nbsp;</td>
      </tr></table>
    </td></tr>"""

    # ── Full article: two-column with vocabulary ────────────────────

    # ── left column: highlighted paragraphs ────────────────────────
    paragraphs_html = ""
    for p in paragraphs:
        orig = _esc(p.get("original", ""))
        trans = _esc(p.get("translation", ""))
        if not trans:
            trans = "[翻译生成失败]"

        highlighted = _highlight(orig, vocabulary)

        paragraphs_html += f"""
            <p style="margin:0 0 16px;font-size:15px;line-height:1.7;color:{TEXT_BODY};font-family:Georgia,'Times New Roman',Times,serif;">{highlighted}</p>
            <p style="margin:-12px 0 20px;font-size:14px;line-height:1.6;color:#999;font-style:italic;">{trans}</p>"""

    # ── right column: vocabulary cards ─────────────────────────────
    vocab_html = ""
    for v in vocabulary[:20]:
        word = _esc(v.get("word", ""))
        phonetic = _esc(v.get("phonetic", ""))
        pos = _esc(v.get("pos", "") or v.get("part_of_speech", ""))
        defin = _esc(v.get("meaning", "") or v.get("definition", ""))
        example_en = _esc(v.get("example_en", ""))
        example_cn = _esc(v.get("example_cn", ""))
        colloc_en = _esc(v.get("collocation_en", ""))
        colloc_cn = _esc(v.get("collocation_cn", ""))

        # Collocation: English in bold serif, Chinese in CN font
        colloc_parts = []
        if colloc_en:
            colloc_parts.append(f'<strong style="font-family:{FONT_EN};font-weight:700;">{colloc_en}</strong>')
        if colloc_cn:
            colloc_parts.append(f'<span style="font-family:{FONT_CN};">{colloc_cn}</span>')
        collocation_html = " ".join(colloc_parts)

        # Example: English in serif, Chinese in CN font (gray)
        example_parts = []
        if example_en:
            example_parts.append(f'<span style="font-family:{FONT_EN};">{example_en}</span>')
        if example_cn:
            example_parts.append(f'<span style="font-family:{FONT_CN};color:#999;">{example_cn}</span>')
        example_html = "<br>".join(example_parts)

        vocab_html += f"""
        <table width="100%" cellpadding="0" cellspacing="0" style="background:{CARD_BG};border:1px solid {CARD_BORDER};border-radius:8px;margin-bottom:10px;"><tr><td style="padding:12px;">
            <div style="font-family:{FONT_EN};font-size:16px;font-weight:700;color:{TEXT_PRIMARY};margin-bottom:2px;">{word}</div>
            <div style="font-size:12px;color:{TEXT_MUTED};margin-bottom:6px;">{phonetic} <span style="font-style:italic;color:#666;">{pos}</span></div>
            <div style="font-family:{FONT_CN};font-size:14px;color:#444;margin-bottom:5px;line-height:1.4;">{defin}</div>
            <div style="font-size:13px;color:{TEXT_PRIMARY};margin-bottom:4px;">{collocation_html}</div>
            <div style="font-size:11px;color:#666;line-height:1.5;border-top:1px solid {CARD_BORDER};padding-top:5px;">{example_html}</div>
        </td></tr></table>"""

    if not vocab_html:
        vocab_html = '<div style="font-size:13px;color:#999;">No vocabulary extracted.</div>'

    return f"""
    <!-- ── article: {title} ── -->
    <tr><td style="padding:32px 40px 4px;">
      <h2 style="margin:0;font-size:20px;font-weight:700;color:{TEXT_PRIMARY};line-height:1.3;font-family:Georgia,'Times New Roman',Times,serif;">
        <!-- 🔧 修改 3：加 target="_blank" -->
        <a href="{url}" target="_blank" rel="noopener noreferrer"
           style="color:{TEXT_PRIMARY};text-decoration:none;">{title}</a>
      </h2>
      <p style="margin:6px 0 0;font-size:12px;color:{TEXT_META};">{source}{" · " + author if author else ""}</p>
    </td></tr>

    <tr><td style="padding:16px 40px 32px;">
      <table width="100%" cellpadding="0" cellspacing="0"><tr>
        <!-- left 70% -->
        <td width="70%" valign="top" style="padding-right:24px;font-size:14px;line-height:1.5;">
          {paragraphs_html}
        </td>
        <!-- right 30% -->
        <td width="30%" valign="top">
          <div style="font-size:12px;font-weight:600;color:{TEXT_MUTED};text-transform:uppercase;letter-spacing:.5px;margin-bottom:12px;">Vocabulary</div>
          {vocab_html}
        </td>
      </tr></table>
    </td></tr>

    <tr><td style="padding:0 40px;">
      <table width="100%" cellpadding="0" cellspacing="0"><tr>
        <td style="height:1px;background:{BORDER_LIGHT};font-size:0;line-height:0;">&nbsp;</td>
      </tr></table>
    </td></tr>"""


# ═══════════════════════════════════════════════════════════════════
#  helpers
# ═══════════════════════════════════════════════════════════════════

def _esc(text: str) -> str:
    """HTML-escape *text*."""
    if not text:
        return ""
    return html_mod.escape(str(text), quote=True)


# 🔧 修改 1：新增 URL 专用转义函数
def _url_esc(url: str) -> str:
    """Prepare a URL for href attribute — preserves & as-is.

    Standard html.escape() converts & to &amp;, which breaks link
    parameters in QQ mail, 163 mail, Outlook for Windows, and Gmail.
    We keep & unescaped, which is universally supported by mail clients.
    """
    if not url:
        return ""
    url = str(url)
    # Only guard against characters that would break the HTML attribute
    url = url.replace('"', '%22')
    return url


def _highlight(text: str, vocabulary: List[Dict]) -> str:
    """Wrap vocabulary words in *text* with a pale-yellow ``<span>``.

    Uses word-boundary matching so that partial matches (e.g. "this"
    matching inside "thisism") are avoided.
    """
    if not vocabulary:
        return text

    # Sort longest-first to avoid nested / partial substitutions
    words = sorted(
        {v.get("word", "") for v in vocabulary if v.get("word")},
        key=len,
        reverse=True,
    )

    for word in words:
        # case-insensitive, word-boundary
        pattern = re.compile(
            rf"(?<!\w)({re.escape(word)})(?!\w)", re.IGNORECASE
        )
        replacement = (
            rf'<span style="background:{HIGHLIGHT};padding:1px 2px;'
            rf'border-radius:2px;">\1</span>'
        )
        text = pattern.sub(replacement, text)

    return text
