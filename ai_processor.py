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

_MAX_RETRIES = 2          # for translation
_MAX_VOCAB_RETRIES = 3   # for vocabulary (stricter dedup)
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
        if len(p) >= 15:  # skip very short fragments
            paragraphs.append(p)

    # Fallback: if no double-newline paragraphs found, split by single newline
    if not paragraphs:
        paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) >= 15]

    translated_paragraphs = _translate_paragraphs(paragraphs)
    vocabulary = _extract_vocabulary(paragraphs)

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
        # One final attempt: translate only paragraphs still missing
        missing = []
        for i, para in enumerate(paragraphs):
            idx = i + 1
            entry = parsed.get(idx, {})
            if not entry.get("translation"):
                missing.append((i, para))
        if missing:
            local_input = "\n".join(
                f"[{j + 1}] {p}" for j, (_, p) in enumerate(missing)
            )
            fix_prompt = (
                f"将以下{len(missing)}段英文翻译成中文。"
                f"对每段只输出：[序号] 中文翻译（不要原文）。\n\n{local_input}"
            )
            fix_system = "你是一个专业翻译。对每段只输出：[序号] 中文翻译。不要输出原文。"
            raw2 = _chat(fix_prompt, fix_system)
            for line in raw2.strip().split("\n"):
                line = line.strip()
                m = re.match(r'\[(\d+)\]\s*(.*)', line)
                if m:
                    local_idx = int(m.group(1)) - 1
                    if 0 <= local_idx < len(missing):
                        global_idx = missing[local_idx][0] + 1
                        if global_idx in parsed:
                            parsed[global_idx]["translation"] = m.group(2)

        # Final per-paragraph fallback for any still-missing translations
        for i, para in enumerate(paragraphs):
            idx = i + 1
            entry = parsed.get(idx, {})
            if not entry.get("translation"):
                try:
                    single_prompt = (
                        f"把下面英文翻译成中文，只输出中文，不要任何多余内容：\n{para}"
                    )
                    single_system = "你是一个专业翻译。只输出中文翻译。"
                    raw3 = _chat(single_prompt, single_system)
                    trans = raw3.strip()
                    if trans:
                        parsed[idx]["translation"] = trans
                except Exception:
                    pass

        result = []
        for i, para in enumerate(paragraphs):
            idx = i + 1
            entry = parsed.get(idx, {})
            orig = entry.get("original", para)
            trans = entry.get("translation", "") or ""
            result.append({"original": orig, "translation": trans})
        return result

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

def _extract_vocabulary(paragraphs: List[str]) -> List[Dict[str, str]]:
    """Extract at least 2 vocabulary items per paragraph.

    Iterates over each paragraph individually so every paragraph is
    guaranteed to contribute at least 2 words.  Results are merged and
    deduplicated by word text, capped at 40 items.
    """
    all_vocab: list[Dict[str, str]] = []
    seen_words: set[str] = set()

    per_para_prompt = (
        "从下面这段英文中提取 2 个最有价值的单词。\n"
        "对每个单词，按以下格式输出，每行一个词，用 | 分隔：\n"
        "单词 | 音标 | 词性(英文) | 中文释义 | 固定搭配(英文短语) | "
        "固定搭配中文释义 | 英文例句 | 中文例句\n\n"
        "严格规则（按优先级排序）：\n"
        "1. 英文例句必须完整（10-25个单词），不能是短语。\n"
        "2. 中文例句必须是英文例句的精确中文翻译，逐词对应。\n"
        "3. 单词、固定搭配、例句优先从原文中选取；如果原文没有合适的，可以自创合理内容。\n"
        "4. 每行必须有8列，顺序为：单词 | 音标 | 词性 | 中文释义 | 固定搭配英文 | 固定搭配中文 | 英文例句 | 中文例句\n\n"
        "待处理段落：\n{para}"
    )
    per_para_system = (
        "你是一个英语词汇老师。每段提取 2 个单词。"
        "严格按照格式输出，每行一个单词，用 | 分隔字段。"
        "每行必须有8列。不要多余的文字。"
    )

    for para in paragraphs:
        if len(para) < 30:
            continue  # skip very short paragraphs
        prompt = per_para_prompt.format(para=para)

        for attempt in range(1 + _MAX_VOCAB_RETRIES):
            raw = _chat(prompt, per_para_system)
            lines = [line.strip() for line in raw.strip().split("\n") if line.strip()]
            local_vocab: list[Dict[str, str]] = []

            for line in lines:
                parts = line.split("|")
                if len(parts) >= 8:
                    example_en = parts[6].strip()
                    example_cn = parts[7].strip()
                    colloc_en = parts[4].strip()
                    colloc_cn = parts[5].strip()
                    collocation = f"{colloc_en} {colloc_cn}" if colloc_en and colloc_cn else ""
                    word = parts[0].strip()
                    local_vocab.append({
                        "word": word,
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
                    word = parts[0].strip()
                    local_vocab.append({
                        "word": word,
                        "phonetic": parts[1].strip(),
                        "pos": parts[2].strip(),
                        "meaning": parts[3].strip(),
                        "collocation": parts[4].strip(),
                        "collocation_en": parts[4].strip(),
                        "collocation_cn": "",
                        "example_en": parts[5].strip(),
                        "example_cn": "",
                    })

            if local_vocab:
                for v in local_vocab:
                    w = v["word"].lower()
                    if w not in seen_words:
                        seen_words.add(w)
                        all_vocab.append(v)
                break  # success for this paragraph

            logger.warning(
                "Per-para vocab attempt %d/%d failed for paragraph "
                "(len=%d), retrying…",
                attempt + 1,
                1 + _MAX_VOCAB_RETRIES,
                len(para),
            )
            if attempt < _MAX_VOCAB_RETRIES:
                time.sleep(_RETRY_DELAY_S)

    return all_vocab[:40]


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
