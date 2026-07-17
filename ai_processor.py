"""
AI processing via DeepSeek API (OpenAI-compatible SDK).
Handles translation and vocabulary extraction for each article.
"""
import logging
import re
import time
from typing import List, Dict, Any

from openai import OpenAI
import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared DeepSeek client
# ---------------------------------------------------------------------------
_client = OpenAI(
    api_key=config.DEEPSEEK_API_KEY,
    base_url=config.DEEPSEEK_BASE_URL,
)

DEFAULT_KWARGS = dict(
    model=config.DEEPSEEK_MODEL,
    temperature=0.3,
    max_tokens=16384,
)

_MAX_RETRIES = 1
_RETRY_DELAY_S = 1.0


def process_article(article: Dict) -> Dict:
    """Translate *article* into Chinese and extract useful vocabulary.

    Returns the original dict augmented with ``paragraphs`` (list of
    ``{"original": …, "translation": …}``) and ``vocabulary`` (list of
    word dicts).

    When the article is a summary-only (``is_summary``), AI translation
    and vocabulary extraction are skipped — the summary text is passed
    through as-is.
    """
    text = article["text"]

    # ── Summary-only: skip AI work ──────────────────────────────────
    if article.get("is_summary"):
        return {
            **article,
            "paragraphs": [{"original": text, "translation": ""}],
            "vocabulary": [],
        }

    # Split by double newline (real paragraph boundaries) and filter
    # out empty / tiny fragments
    raw_paragraphs = re.split(r"\n\s*\n", text.strip())
    paragraphs = []
    for p in raw_paragraphs:
        p = p.strip()
        if not p:
            continue
        # Collapse internal newlines into spaces
        p = re.sub(r"\s+", " ", p)
        if len(p) >= 30:  # skip very short fragments
            paragraphs.append(p)

    # Fallback: if no double-newline paragraphs found, split by single newline
    if not paragraphs:
        paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) >= 30]

    translated_paragraphs = _translate_paragraphs(paragraphs)
    vocabulary = _extract_vocabulary(text)

    return {
        **article,
        "paragraphs": translated_paragraphs,
        "vocabulary": vocabulary,
    }


# ---------------------------------------------------------------------------
#  Translation  (plain-text, line-based)
# ---------------------------------------------------------------------------

def _translate_paragraphs(paragraphs: List[str]) -> List[Dict[str, str]]:
    """Translate every paragraph into Chinese (one-to-one).

    Sends paragraphs with explicit [N] markers so the AI must respond
    with the same numbering — this guarantees strict 1:1 alignment
    between original and translation.
    """
    if not paragraphs:
        return []

    numbered_input = "\n".join(
        f"[{i + 1}] {p}" for i, p in enumerate(paragraphs)
    )
    prompt = (
        "请将以下各段英文逐段翻译成中文。\n"
        "对每一段，先原样输出原文（以[序号]开头），再输出对应的中文翻译。\n"
        f"格式示例：\n"
        f"[1] Original text...\n"
        f"[1] 中文翻译...\n"
        f"\n"
        f"待翻译内容：\n{numbered_input}"
    )
    system = (
        "你是一个专业翻译。严格按照格式输出：[序号] 原文 和 [序号] 翻译，"
        "每个[序号]一行。原文必须原样保留，不得修改。"
    )

    for attempt in range(1 + _MAX_RETRIES):
        raw = _chat(prompt, system)
        lines = [line.strip() for line in raw.strip().split("\n") if line.strip()]

        # Parse lines into a dict: {idx: {"original": ..., "translation": ...}}
        parsed: Dict[int, Dict[str, str]] = {}
        for line in lines:
            m = re.match(r'\[(\d+)\]\s*(.*)', line)
            if m:
                idx = int(m.group(1))
                text = m.group(2)
                if idx not in parsed:
                    parsed[idx] = {}
                # Decide whether it's original or translation by checking
                # whether it contains mostly ASCII characters
                ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
                if ascii_ratio > 0.7:
                    parsed[idx]["original"] = text
                else:
                    parsed[idx]["translation"] = text

        # Each paragraph must have BOTH an original AND a translation
        if len(parsed) >= len(paragraphs) and all(
            len(v) >= 2 for v in parsed.values()
        ):
            break

        logger.warning(
            "Translation parsed %d blocks, expected %d (attempt %d/%d), retrying…",
            len(parsed),
            len(paragraphs),
            attempt + 1,
            1 + _MAX_RETRIES,
        )
        if attempt < _MAX_RETRIES:
            time.sleep(_RETRY_DELAY_S)
    else:
        logger.error("Translation failed after %d attempts", 1 + _MAX_RETRIES)
        return [{"original": p, "translation": "[翻译生成失败]"} for p in paragraphs]

    result = []
    for i, para in enumerate(paragraphs):
        idx = i + 1
        entry = parsed.get(idx, {})
        trans = entry.get("translation", "[翻译失败]")
        result.append({"original": para, "translation": trans})
    return result


# ---------------------------------------------------------------------------
#  Vocabulary extraction  (pipe-delimited plain text)
# ---------------------------------------------------------------------------

def _extract_vocabulary(text: str) -> List[Dict[str, str]]:
    """Extract vocabulary items from *text* (requests 22, returns up to 20)."""
    prompt = (
        f"从下面文章中提取 22 个最有价值的单词。"
        f"对每个单词，按以下格式输出，每行一个词，用 | 分隔：\n\n"
        f"单词 | 音标 | 词性(英文) | 中文释义 | 固定搭配(英文短语) | 固定搭配中文释义 | 英文例句 | 中文例句\n\n"
        f"严格规则：\n"
        f"1. 固定搭配必须同时提供英文短语和中文释义\n"
        f"2. 英文例句必须是原文中的完整句子（不是短语），长度适中（10-25个单词）\n"
        f"3. 中文例句必须是与英文例句不同内容的独立句子，禁止中英互译同一句话\n"
        f"4. 不同单词必须使用不同的例句场景，禁止两个单词共用同一个句型模板\n"
        f"5. 连续 10 个单词至少覆盖 5 种不同的主题场景\n"
        f"6. 所有句子必须完整、包含具体信息、不要太长\n\n"
        f"文章：\n{text[:6000]}"
    )
    system = (
        "你是一个英语词汇老师。严格按照格式输出，每行一个单词，用 | 分隔字段。"
        "每行必须有8列。英文例句必须是原文完整句子（10-25词）。"
        "中文例句必须与英文例句是不同的独立句子，不是翻译关系。"
        "每个单词的例句场景必须独一无二，禁止重复句型。"
        "不要多余的文字。"
    )

    for attempt in range(1 + _MAX_RETRIES):
        raw = _chat(prompt, system)
        lines = [line.strip() for line in raw.strip().split("\n") if line.strip()]

        vocab = []
        for line in lines:
            parts = line.split("|")
            if len(parts) >= 8:
                example_en = parts[6].strip()
                example_cn = parts[7].strip()
                # Build collocation with both en and cn for email display
                colloc_en = parts[4].strip()
                colloc_cn = parts[5].strip()
                collocation = f"{colloc_en} {colloc_cn}" if colloc_en and colloc_cn else ""
                vocab.append({
                    "word": parts[0].strip(),
                    "phonetic": parts[1].strip(),
                    "pos": parts[2].strip(),
                    "meaning": parts[3].strip(),
                    "collocation": collocation,
                    "collocation_en": colloc_en,
                    "collocation_cn": colloc_cn,
                    "example_en": example_en,
                    "example_cn": example_cn,
                })
            elif len(parts) >= 6:
                # Fallback for old 6-column format
                vocab.append({
                    "word": parts[0].strip(),
                    "phonetic": parts[1].strip(),
                    "pos": parts[2].strip(),
                    "meaning": parts[3].strip(),
                    "collocation": parts[4].strip(),
                    "collocation_en": parts[4].strip(),
                    "collocation_cn": "",
                    "example_en": parts[5].strip(),
                    "example_cn": "",
                })

        if vocab:
            return vocab[:20]

        logger.warning(
            "Vocabulary parsing produced 0 items (attempt %d/%d), retrying…",
            attempt + 1,
            1 + _MAX_RETRIES,
        )
        if attempt < _MAX_RETRIES:
            time.sleep(_RETRY_DELAY_S)

    logger.error("Vocabulary extraction failed after %d attempts", 1 + _MAX_RETRIES)
    return []


# ---------------------------------------------------------------------------
# low-level chat
# ---------------------------------------------------------------------------

def _chat(prompt: str, system: str) -> str:
    """Low-level DeepSeek chat call – returns raw content string."""
    resp = _client.chat.completions.create(
        **DEFAULT_KWARGS,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content or ""
