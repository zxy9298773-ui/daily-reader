"""
AI processing via DeepSeek API (OpenAI-compatible SDK).
Handles translation and vocabulary extraction for each article.
"""
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

from openai import OpenAI
import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Input/output length safety guards  (must be defined before DEFAULT_KWARGS)
# ---------------------------------------------------------------------------
_MAX_TRANSLATION_INPUT_CHARS = 30000   # max chars of paragraph text sent per batch for translation
_MAX_VOCAB_INPUT_CHARS = 30000        # max chars of paragraph text sent for vocab extraction
_MAX_OUTPUT_TOKENS = 8192             # DeepSeek models cap output at 8K tokens server-side

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
    max_tokens=_MAX_OUTPUT_TOKENS,
)

_MAX_RETRIES = 1          # for translation
_MAX_VOCAB_RETRIES = 3   # for vocabulary (stricter dedup)
_RETRY_DELAY_S = 1.0


def _truncate_paragraphs(paragraphs: list[str], max_chars: int) -> list[str]:
    """Drop paragraphs from the end until total length fits *max_chars*.

    Always keeps at least 1 paragraph.  Returns the truncated list.
    If a single paragraph exceeds *max_chars*, it is hard-truncated.
    """
    total = sum(len(p) for p in paragraphs)
    if total <= max_chars:
        return paragraphs
    truncated = list(paragraphs)
    while len(truncated) > 1 and sum(len(p) for p in truncated) > max_chars:
        truncated.pop()
    # Hard-truncate the last (and only) paragraph if it's still too long
    if truncated and sum(len(p) for p in truncated) > max_chars:
        truncated[0] = truncated[0][:max_chars]
    logger.warning(
        "Truncated %d paragraphs to %d (%.0f%% of original length)",
        len(paragraphs), len(truncated),
        sum(len(p) for p in truncated) / total * 100,
    )
    return truncated


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

    # ── Translate + extract vocabulary in parallel ────────────────
    # The two tasks are I/O-bound (DeepSeek API calls), so threading
    # cuts wall-clock time roughly in half.
    with ThreadPoolExecutor(max_workers=2) as executor:
        trans_future = executor.submit(_translate_paragraphs, paragraphs)
        vocab_future = executor.submit(_extract_vocabulary, paragraphs)
        translated_paragraphs = trans_future.result()
        vocabulary = vocab_future.result()

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

    Sends paragraphs in small batches so each API call's output stays
    well within the model's output token limit (~8K for DeepSeek).

    Each batch uses explicit [N] markers for 1:1 alignment.
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

    # ── Truncate if total input is too long ────────────────────────
    paragraphs = _truncate_paragraphs(paragraphs, _MAX_TRANSLATION_INPUT_CHARS)

    # ── Batch translate ───────────────────────────────────────────
    # Each paragraph pair (original + translation) needs ~450 output
    # tokens.  Batching ≤10 paragraphs keeps output within 8K.
    _BATCH_SIZE = 10
    parsed: Dict[int, Dict[str, str]] = {}
    prompt_template = (
        "请将以下各段英文逐段翻译成中文。\n"
        "对每一段，先原样输出原文（以[序号]开头），再输出对应的中文翻译。\n"
        "格式示例：\n"
        "[1] Original text...\n"
        "[1] 中文翻译...\n"
        "\n"
        "待翻译内容：\n{}"
    )
    system = (
        "你是一个专业翻译。严格按照格式输出：[序号] 原文 和 [序号] 翻译，"
        "每个[序号]一行。原文必须原样保留，不得修改。"
    )

    for batch_start in range(0, len(paragraphs), _BATCH_SIZE):
        batch = paragraphs[batch_start:batch_start + _BATCH_SIZE]
        numbered_input = "\n".join(
            f"[{i + 1}] {p}" for i, p in enumerate(batch)
        )
        prompt = prompt_template.format(numbered_input)

        batch_parsed: Dict[int, Dict[str, str]] = {}
        for attempt in range(1 + _MAX_RETRIES):
            raw = _chat(prompt, system)
            lines = [line.strip() for line in raw.strip().split("\n") if line.strip()]

            # Parse lines into batch_parsed keyed by global 1‑based index
            for line in lines:
                m = re.match(r'\[(\d+)\]\s*(.*)', line)
                if not m:
                    continue
                local_idx = int(m.group(1))               # 1‑based within batch
                global_idx = batch_start + local_idx       # 1‑based global
                text = m.group(2)
                if global_idx not in batch_parsed:
                    batch_parsed[global_idx] = {}
                ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
                if ascii_ratio > 0.7:
                    batch_parsed[global_idx]["original"] = text
                else:
                    batch_parsed[global_idx]["translation"] = text

            # Verify this batch is complete
            expected_in_batch = len(batch)
            batch_ok = len(batch_parsed) >= expected_in_batch and all(
                len(v) >= 2 for v in batch_parsed.values()
            )
            if batch_ok:
                break

            logger.warning(
                "Batch [%d-%d] parsed %d/%d blocks (attempt %d/%d)",
                batch_start + 1, batch_start + len(batch),
                len(batch_parsed), expected_in_batch,
                attempt + 1, 1 + _MAX_RETRIES,
            )
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY_S)

        parsed.update(batch_parsed)

    # ── One final attempt: translate only paragraphs still missing ─
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
        try:
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
        except Exception:
            pass

    # ── Final per-paragraph fallback for any still missing ────────
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

    # ── Build result ──────────────────────────────────────────────
    result = []
    for i, para in enumerate(paragraphs):
        idx = i + 1
        entry = parsed.get(idx, {})
        orig = entry.get("original", para)
        trans = entry.get("translation", "") or ""
        result.append({"original": orig, "translation": trans})
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

    # ── Truncate selected paragraphs if total input is too long ────
    total_selected = sum(len(t[1]) for t in selected)
    if total_selected > _MAX_VOCAB_INPUT_CHARS:
        truncated = list(selected)
        while len(truncated) > 1 and sum(len(t[1]) for t in truncated) > _MAX_VOCAB_INPUT_CHARS:
            truncated.pop()
        logger.warning(
            "Vocab bulk input truncated: %d paragraphs → %d (%.0f%% of original length)",
            len(selected), len(truncated),
            sum(len(t[1]) for t in truncated) / total_selected * 100,
        )
        selected = truncated

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
    """Low-level DeepSeek chat call – returns raw content string.

    Raises ``RuntimeError`` on API failure with the underlying error
    message attached.
    """
    # Estimate whether prompt is too long (rough: 1 token ≈ 4 chars)
    estimated_prompt_tokens = (len(system) + len(prompt)) // 4
    if estimated_prompt_tokens > 48000:
        logger.warning(
            "Prompt is very large (~%d tokens), may exceed context window",
            estimated_prompt_tokens,
        )

    try:
        resp = _client.chat.completions.create(
            **DEFAULT_KWARGS,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
    except Exception as e:
        err_msg = str(e)
        logger.error("DeepSeek API call failed: %s", err_msg)
        raise RuntimeError(f"DeepSeek API error: {err_msg}") from e

    content = resp.choices[0].message.content or ""
    if not content:
        logger.warning("DeepSeek returned empty content (prompt ~%d tokens)", estimated_prompt_tokens)
    return content
