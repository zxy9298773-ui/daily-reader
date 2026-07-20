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

_MAX_RETRIES = 1          # for translation
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

    # ── Merge short adjacent paragraphs into real-sized paragraphs ─
    merged = []
    for p in paragraphs:
        if merged and len(merged[-1]) + len(p) < 300:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    paragraphs = merged

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

    Two-layer strategy:
      1. Bulk call — send all paragraphs with [Pn] markers, ask for 2 words
         per paragraph.  Tracks which paragraphs are covered.
      2. Per-paragraph fill — for any paragraph with < 2 words, make an
         individual call to fill the gap.
    """
    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def _parse_line(line: str) -> Dict[str, str] | None:
        """Parse a single pipe-delimited vocabulary line into a dict.

        Returns None when the line cannot be parsed.
        """
        parts = line.split("|")
        if len(parts) >= 8:
            example_en = parts[6].strip()
            example_cn = parts[7].strip()
            colloc_en = parts[4].strip()
            colloc_cn = parts[5].strip()
            collocation = f"{colloc_en} {colloc_cn}" if colloc_en and colloc_cn else ""
            return {
                "word":          parts[0].strip(),
                "phonetic":      parts[1].strip(),
                "pos":           parts[2].strip(),
                "meaning":       parts[3].strip(),
                "collocation":   collocation,
                "collocation_en": colloc_en,
                "collocation_cn": colloc_cn,
                "example_en":    example_en,
                "example_cn":    example_cn,
            }
        if len(parts) >= 6:
            return {
                "word":          parts[0].strip(),
                "phonetic":      parts[1].strip(),
                "pos":           parts[2].strip(),
                "meaning":       parts[3].strip(),
                "collocation":   parts[4].strip(),
                "collocation_en": parts[4].strip(),
                "collocation_cn": "",
                "example_en":    parts[5].strip(),
                "example_cn":    "",
            }
        return None

    def _dedup_append(vocab_list: list, item: dict) -> None:
        """Append *item* to *vocab_list* if its word hasn't been seen."""
        w = item["word"].lower()
        if w not in seen_words:
            seen_words.add(w)
            vocab_list.append(item)

    # ------------------------------------------------------------------
    # Phase 1 — Bulk call with paragraph markers
    # ------------------------------------------------------------------
    all_vocab: list[Dict[str, str]] = []
    seen_words: set[str] = set()

    # Track how many words per paragraph (by 0‑based index)
    para_count: dict[int, int] = {}

    # Build annotated input only for paragraphs long enough to be useful
    selected: list[tuple[int, str]] = []
    for i, p in enumerate(paragraphs):
        if len(p) >= 30:
            selected.append((i, p))

    if not selected:
        return []

    all_para_indices = [idx for idx, _ in selected]

    bulk_input = "\n".join(f"[P{idx + 1}] {p}" for idx, p in selected)
    bulk_system = (
        "你是一个英语词汇老师。严格按格式输出，每行一个单词，行首必须标注段落编号。"
        "每行必须有8列。不要多余的文字。"
    )
    bulk_prompt = (
        "从下面每段英文中各提取 2 个最有价值的单词。\n"
        "对每个单词，输出格式：\n"
        "[Pn] 单词 | 音标 | 词性(英文) | 中文释义 | "
        "固定搭配(英文短语) | 固定搭配中文释义 | 英文例句 | 中文例句\n\n"
        "规则：\n"
        "1. 每个段落 [Pn] 输出 2 行（每行 1 个单词），不要多也不要少。\n"
        "2. 英文例句必须完整（10-25个单词），不能是短语。\n"
        "3. 中文例句必须是英文例句的精确中文翻译。\n"
        "4. 禁止两个单词共用同一个英文例句。\n"
        "5. 优先使用原文句子；没有合适的可以自创。\n\n"
        "待处理内容：\n" + bulk_input
    )

    raw = _chat(bulk_prompt, bulk_system)
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Match [Pn] prefix
        m = re.match(r'\[P(\d+)\]\s*(.*)', line)
        if not m:
            continue
        para_idx = int(m.group(1)) - 1  # back to 0-based
        content = m.group(2)
        item = _parse_line(content)
        if item:
            _dedup_append(all_vocab, item)
            para_count[para_idx] = para_count.get(para_idx, 0) + 1

    # ------------------------------------------------------------------
    # Phase 2 — Fill paragraphs with < 2 words
    # ------------------------------------------------------------------
    for idx in all_para_indices:
        if para_count.get(idx, 0) >= 2:
            continue

        para_text = paragraphs[idx]
        fill_prompt = (
            "从下面这段英文中提取 2 个最有价值的单词。\n"
            "对每个单词，按以下格式输出，每行一个词，用 | 分隔：\n"
            "单词 | 音标 | 词性(英文) | 中文释义 | 固定搭配(英文短语) | "
            "固定搭配中文释义 | 英文例句 | 中文例句\n\n"
            "规则：\n"
            "1. 英文例句必须完整（10-25个单词），不能是短语。\n"
            "2. 中文例句必须是英文例句的精确中文翻译。\n"
            "3. 优先使用原文句子；没有合适的可以自创。\n"
            "4. 每行必须有8列。\n\n"
            "待处理段落：\n" + para_text
        )
        fill_system = (
            "你是一个英语词汇老师。每段提取 2 个单词。"
            "严格按照格式输出，每行一个单词，用 | 分隔字段。"
            "每行必须有8列。不要多余的文字。"
        )

        for attempt in range(1 + _MAX_VOCAB_RETRIES):
            raw2 = _chat(fill_prompt, fill_system)
            items = []
            for line in raw2.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                item = _parse_line(line)
                if item:
                    items.append(item)
            if items:
                for item in items:
                    _dedup_append(all_vocab, item)
                break
            logger.warning(
                "Vocab fill attempt %d/%d failed for paragraph %d, retrying…",
                attempt + 1, 1 + _MAX_VOCAB_RETRIES, idx,
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
