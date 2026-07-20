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
    """Extract vocabulary items from *paragraphs*, at least 2 per paragraph.

    Builds a structured prompt with paragraph markers so the model
    distributes words evenly across the article.
    """
    # Build paragraph-annotated input (capped to avoid token overflow)
    para_lines: list[str] = []
    total_chars = 0
    for i, p in enumerate(paragraphs):
        prefix = f"[P{i + 1}] "
        line = prefix + p
        total_chars += len(line)
        if total_chars > 5_500:
            break
        para_lines.append(line)

    num_paras = len(para_lines)
    target_words = min(num_paras * 2, 40)

    prompt = (
        f"下面文章有 {num_paras} 段。从每一段中提取至少 2 个最有价值的单词，"
        f"共提取约 {target_words} 个单词。分布要求：每段至少 2 个，不允许任何段为 0 个。\n\n"
        f"文章段落：\n" + "\n".join(para_lines) + "\n\n"
        f"对每个单词，按以下格式输出，每行一个词，用 | 分隔：\n"
        f"单词 | 音标 | 词性(英文) | 中文释义 | 固定搭配(英文短语) | 固定搭配中文释义 | 英文例句 | 中文例句\n\n"
        f"严格规则（按优先级排序）：\n"
        f"1. 【最重要】禁止任何两个单词共用同一个英文例句。每个单词的英文例句必须是独一无二的，\n"
        f"   即使原文中同一个句子包含多个目标单词，每个单词也只能使用该句子一次，其他单词必须换用不同的例句。\n"
        f"2. 中文例句必须是英文例句的精确中文翻译，逐词对应，不能意译、不能自己编造不同内容。\n"
        f"   例：\n"
        f"     英文: Smartphones are prevalent among young people.\n"
        f"     中文: 智能手机在年轻人中普遍存在。  ✅（精确翻译）\n"
        f"     中文: 智能手机很流行。             ❌（意译，不精确）\n"
        f"3. 英文例句可以来自原文完整句子，也可以是针对该单词自创的合理句子。\n"
        f"   优先使用原文句子，但如果原文句子质量差或长度不合适，可以自创更自然、更清晰的例句。\n"
        f"4. 所有例句必须是信息完整的句子（10-25个单词），不能是短语或片段。\n"
        f"5. 【分布要求】输出时，不要求按段落顺序排列，但要确保最终列表涵盖所有段落。\n"
        f"   即：每一段至少有 2 个单词被选中，整体分布均匀。\n\n"
        f"每行必须有8列，顺序为：单词 | 音标 | 词性 | 中文释义 | 固定搭配英文 | 固定搭配中文 | 英文例句 | 中文例句"
    )
    system = (
        "你是一个英语词汇老师。严格按照格式输出，每行一个单词，用 | 分隔字段。"
        "每行必须有8列，顺序为：单词 | 音标 | 词性 | 中文释义 | 固定搭配英文 | 固定搭配中文 | 英文例句 | 中文例句。"
        "中文例句必须是英文例句的精确中文翻译，不能编造不同内容。"
        "绝对禁止两个单词共用同一个英文例句。"
        "不要多余的文字。"
    )

    for attempt in range(1 + _MAX_VOCAB_RETRIES):
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
            # Dedup check: reject if any two vocab items share the same English example
            seen_examples: set[str] = set()
            has_dup = False
            for v in vocab:
                ex = v.get("example_en", "")
                if ex in seen_examples:
                    has_dup = True
                    break
                seen_examples.add(ex)

            if has_dup:
                logger.warning(
                    "Vocab has duplicate English examples (attempt %d/%d), retrying…",
                    attempt + 1,
                    1 + _MAX_VOCAB_RETRIES,
                )
                if attempt < _MAX_VOCAB_RETRIES:
                    time.sleep(_RETRY_DELAY_S)
                    continue
                # Retries exhausted → deduplicate by keeping only the first
                # occurrence of each unique English example
                seen: set[str] = set()
                deduped = []
                for v in vocab:
                    ex = v.get("example_en", "")
                    if ex not in seen:
                        seen.add(ex)
                        deduped.append(v)
                logger.warning(
                    "Vocab dedup fallback: reduced %d → %d items",
                    len(vocab),
                    len(deduped),
                )
                return deduped[:40]

            return vocab[:40]

        logger.warning(
            "Vocabulary parsing produced 0 items (attempt %d/%d), retrying…",
            attempt + 1,
            1 + _MAX_VOCAB_RETRIES,
        )
        if attempt < _MAX_VOCAB_RETRIES:
            time.sleep(_RETRY_DELAY_S)

    logger.error("Vocabulary extraction failed after %d attempts", 1 + _MAX_VOCAB_RETRIES)
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
