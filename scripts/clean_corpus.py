#!/usr/bin/env python3
"""Clean a curated corpus file before generating training data.

Anything left in the corpus ends up in training targets, and the model
learns it as the author's style. This removes:

- Gutenberg markup (_italics_, =AB=, [Math: ...]) and "--" dashes
- footnote markers, footnote bodies and inline page citations
- ALL-CAPS runs that were italics in the source
- chapter numbers stuck to the start of a paragraph
- paragraphs with formulas stripped out, or cut off mid-sentence

A badly scanned book can be rebuilt from its raw OCR text with --rebuild:
running headers, page numbers and footnotes are dropped, page breaks and
hyphenated words rejoined, and DeepSeek fixes spelling and OCR errors
(--ocr-fix). The model may only fix OCR damage; replies that change more
than that are rejected and the original is kept.

    python scripts/clean_corpus.py data/corpus/curated/russell.txt \\
        --rebuild data/corpus/russell/sceptical_essays.txt --ocr-fix
"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MIN_WORDS = 40

_ROMAN = re.compile(r"^[IVXLCDM]+$")

# Paragraphs whose formulas were stripped leave these holes behind.
_FORMULA_GAP = re.compile(
    r"\[Math"
    r"|\b[Ll]et be\b"
    r"|\b(?:Let|let|of|is|and|when|where|than|to|by|that|as|form)\s[,.;:)]"
    r"|\(\w\)\s;"
    r"|\s,\s,"
    r"|\bof 's\b"
)

_FOOTNOTE_BODY = re.compile(r"^(?:\*+|\[\d+\]|†|‡)\s*")
_SECTION_HEADING = re.compile(r"^\(\w\)\s+[A-Z][A-Z ,'’-]+\.\s*(?:--|—)\s*")
_CHAPTER_PREFIX = re.compile(r"^(?:[IVXL]{2,}|Ill|\d{1,2})\s+(?=[A-Z][a-z]|I\s)")
_SENTENCE_END = re.compile(r"[.!?:;\"”’')\]]$")


def _lower_caps(text: str) -> str:
    """Lowercase ALL-CAPS emphasis, keeping Roman numerals and short labels."""
    def repl(match):
        word = match.group(0)
        if len(word) < 3 or _ROMAN.match(word):
            return word
        lower = word.lower()
        before = text[:match.start()].rstrip()
        # Capitalize again where the word opens a sentence or a quote.
        if not before or before[-1] in '.!?"“(':
            lower = lower[0].upper() + lower[1:]
        return lower

    return re.sub(r"\b[A-Z][A-Z'’]+\b", repl, text)


def clean_paragraph(text: str, min_words: int = MIN_WORDS) -> Optional[str]:
    """Clean one paragraph, or return None if it should be dropped."""
    text = text.strip()
    # A lowercase start continues something that was removed (usually a
    # displayed formula: "where is the frequency, ...").
    if _FOOTNOTE_BODY.match(text) or _FORMULA_GAP.search(text) or text[:1].islower():
        return None

    text = _SECTION_HEADING.sub("", text)
    text = _CHAPTER_PREFIX.sub("", text)

    # Markup
    text = re.sub(r"(?<![\w_])_([^_]+?)_(?![\w_])", r"\1", text)
    text = re.sub(r"(?<!\w)_(?=\w)|(?<=\w)_(?!\w)", "", text)
    text = re.sub(r"=([A-Z][A-Za-z′']*)=", r"\1", text)

    # Footnote markers and page citations
    text = re.sub(r"\[(?:\d+|[A-Z])\]", "", text)
    text = re.sub(r"(?<=[\w.,;:!?\"”’')])\*+", "", text)
    text = re.sub(r"\s\*+(?=\s|$)", "", text)
    text = re.sub(r"\s*\([^()]*\bpp?\.\s*\d[^()]*\)", "", text)

    # Dashes: ":--" introduces a list or quote; other "--" is an em-dash.
    text = re.sub(r":\s*--", ":", text)
    text = re.sub(r"\s*--\s*", "—", text)

    text = _lower_caps(text)

    # Lone scanner debris: "bias. ^ It is", "■", "|"
    text = re.sub(r"(?:(?<=\s)|^)[\^|■~«»•<>\\]+(?=\s)", "", text)
    text = re.sub(r"(?<=\w)[|■]+(?=\s)", "", text)

    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text).strip()

    # Cut off mid-sentence: the rest was a formula. Keep the whole sentences.
    if text.endswith("—") or not _SENTENCE_END.search(text):
        ends = list(re.finditer(r"[.!?][\"”’')]?(?=\s)", text))
        if not ends:
            return None
        text = text[:ends[-1].end()]
    if len(text.split()) < min_words:
        return None
    return text


# ---------------------------------------------------------------------------
# Rebuilding an OCR-scanned book
# ---------------------------------------------------------------------------

# Page furniture: "IDEALS OF HAPPINESS 103", "1 68", "1 8a", "*54", "III"
_HEADER_LINE = re.compile(r"^\s*(?:\d+\s+)?[A-Z][A-Z ,:'’-]{2,}(?:\s+\d+)?\s*$")
_PAGE_NUMBER_LINE = re.compile(r"^\s*[*«•]?\s*\d[\d ]{0,4}[a-z]?\s*[*]?\s*$")
_FOOTNOTE_BLOCK = re.compile(r"^\s*(?:\d{1,2}|[•*†>])\s+\S")
# Footnote line inside a block: "1 See The Freeman, February 15, 1922, p. 532."
_FOOTNOTE_LINE = re.compile(r"^\s*(?:\d{1,2}|[•*†])\s+[A-Z]")
# Section numbers as OCR reads them: "i7 What", "rv The", "ii Men's", "in What"
_OCR_SECTION = re.compile(
    r"^(?:[ivxl]{1,4}\d?|[IVXL]{2,4}|rv|\d{1,2}|in(?=\s(?:What|The|It|I|We|There|This|If|When|A|One|In)\b))\s+(?=[A-Z])"
)
_BLOCK_END = re.compile(r"[.!?:\"”’')\]]$")


def _is_furniture(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _PAGE_NUMBER_LINE.match(stripped) or _ROMAN.match(stripped):
        return True
    # All-caps line: running header or chapter title. A lone "I" is a word.
    return bool(_HEADER_LINE.match(stripped)) and stripped not in {"I", "A"}


def rebuild_ocr_book(raw: str, min_words: int = MIN_WORDS) -> List[str]:
    """Turn raw OCR text (one line per printed line) into clean paragraphs."""
    blocks = []
    for block in re.split(r"\n\s*\n", raw):
        lines = [ln for ln in block.split("\n") if ln.strip() and not _is_furniture(ln)]
        # Footnote lines mid-block; a footnote opening a block is handled below.
        lines = lines[:1] + [ln for ln in lines[1:] if not _FOOTNOTE_LINE.match(ln)]
        if not lines:
            continue
        text = ""
        for line in lines:
            line = line.strip()
            if text.endswith("-") and not text.endswith("--") and line[:1].islower():
                text = text[:-1] + line  # hyphenated line break
            else:
                text = f"{text} {line}" if text else line
        blocks.append(text)

    # Footnotes sit wherever the page ended, often mid-sentence.
    blocks = [b for b in blocks if not (_FOOTNOTE_BLOCK.match(b) and len(b.split()) < 80)]
    blocks = [_OCR_SECTION.sub("", b) for b in blocks]

    # A block that doesn't finish its sentence continues on the next page.
    # A lowercase block continues the last unfinished paragraph, even when a
    # quote or footnote block sits between them.
    paragraphs: List[str] = []
    for block in blocks:
        target = None
        if paragraphs and not _BLOCK_END.search(paragraphs[-1]):
            target = len(paragraphs) - 1
        elif paragraphs and block[:1].islower():
            unfinished = [i for i in range(max(0, len(paragraphs) - 3), len(paragraphs))
                          if not _BLOCK_END.search(paragraphs[i])]
            target = unfinished[-1] if unfinished else len(paragraphs) - 1
        if target is None:
            paragraphs.append(block)
            continue
        prev = paragraphs[target]
        if prev.endswith("-") and not prev.endswith("--") and block[:1].islower():
            paragraphs[target] = prev[:-1] + block
        else:
            paragraphs[target] = f"{prev} {block}"

    cleaned = []
    for para in paragraphs:
        para = re.sub(r"\s+([.,;:!?])", r"\1", para)
        para = re.sub(r"\s{2,}", " ", para).strip()
        if len(para.split()) >= min_words:
            cleaned.append(para)
    return cleaned


def replace_book(paragraphs: List[str], raw: str, new_paragraphs: List[str]) -> List[str]:
    """Swap the contiguous run of paragraphs taken from ``raw`` for new ones."""
    norm = re.sub(r"\s+", " ", raw)
    hits = [i for i, p in enumerate(paragraphs) if re.sub(r"\s+", " ", p)[20:80] in norm]
    if not hits:
        raise ValueError("No paragraphs from this book found in the corpus")
    start, end = hits[0], hits[-1]
    return paragraphs[:start] + new_paragraphs + paragraphs[end + 1:]


# ---------------------------------------------------------------------------
# OCR repair via DeepSeek
# ---------------------------------------------------------------------------

OCR_SYSTEM_PROMPT = """You repair OCR errors in a scanned book by Bertrand Russell.

Fix ONLY damage done by the scanner:
- misread letters ("sdence" -> "science", "pladng" -> "placing", "stem" -> "stern" where the context demands it)
- words split by stray spaces ("amic able adjustm ent" -> "amicable adjustment", "T he" -> "The")
- words run together ("ofprobability" -> "of probability")
- stray page numbers, footnote marks or header words left inside a sentence

Do NOT reword, modernize spelling or period word forms (Marxianism, to-day, connexion), change British spellings, fix grammar, change punctuation style, or add, remove or reorder any words that were printed. If a passage is already correct, return it unchanged.

Return only the repaired text, with no commentary."""

# A repair may touch this many characters per 100 of the original.
MAX_OCR_EDIT_RATE = 4.0
MAX_OCR_WORD_DRIFT = 0.03
# Short passages get a fixed budget so a couple of fixes aren't over the rate.
MIN_OCR_EDIT_BUDGET = 12


def ocr_fix_acceptable(original: str, fixed: str) -> bool:
    """True if ``fixed`` only repairs characters and doesn't reword."""
    if not fixed.strip():
        return False
    orig_words, fixed_words = len(original.split()), len(fixed.split())
    # Rejoining split words ("adjustm ent") lowers the count, so allow for those.
    allowed_drift = max(3, orig_words * MAX_OCR_WORD_DRIFT) + _split_word_count(original)
    if abs(orig_words - fixed_words) > allowed_drift:
        return False
    matcher = SequenceMatcher(None, original, fixed, autojunk=False)
    changed = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal")
    return changed <= max(MIN_OCR_EDIT_BUDGET, len(original) * MAX_OCR_EDIT_RATE / 100)


def _split_word_count(text: str) -> int:
    """Rough count of OCR word splits ("T he", "adjustm ent") in ``text``."""
    return len(re.findall(r"\b(?:[A-Za-z]|[a-z]{2,}) (?=[a-z]{1,3}\b)", text))


def fix_ocr(paragraphs: List[str], call: Callable[[str], str], cache_path: Path,
            workers: int = 8, log=print) -> List[str]:
    """Repair OCR errors paragraph by paragraph, caching replies on disk."""
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [p for p in paragraphs if p not in cache]

    def work(para: str):
        try:
            return para, call(para).strip()
        except Exception as e:  # keep going; the paragraph stays as it was
            log(f"  OCR fix failed: {e}")
            return para, None

    if todo:
        log(f"OCR repair: {len(todo)} paragraphs ({len(paragraphs) - len(todo)} cached)")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, (para, reply) in enumerate(pool.map(work, todo), 1):
                if reply is not None:
                    cache[para] = reply
                if i % 50 == 0:
                    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
                    log(f"  {i}/{len(todo)}")
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1))

    out, rejected, changed = [], 0, 0
    for para in paragraphs:
        reply = cache.get(para)
        if reply and reply != para and ocr_fix_acceptable(para, reply):
            out.append(reply)
            changed += 1
        else:
            if reply and reply != para:
                rejected += 1
            out.append(para)
    log(f"OCR repair: {changed} paragraphs fixed, {rejected} replies rejected as rewrites")
    return out


def deepseek_caller() -> Callable[[str], str]:
    from src.config import LLMProviderConfig
    from src.llm.deepseek import DeepSeekProvider
    import os

    provider = DeepSeekProvider(LLMProviderConfig(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        max_tokens=4000,
        temperature=0.0,
        timeout=180,
    ))
    return lambda text: provider.call(system_prompt=OCR_SYSTEM_PROMPT, user_prompt=text,
                                      temperature=0.0, max_tokens=max(512, len(text.split()) * 3))


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Clean a curated corpus file")
    parser.add_argument("input", type=Path, help="Curated corpus (paragraphs separated by blank lines)")
    parser.add_argument("-o", "--output", type=Path, help="Output path (default: overwrite input)")
    parser.add_argument("--rebuild", type=Path, action="append", default=[],
                        help="Raw OCR text of a book to rebuild in place (repeatable)")
    parser.add_argument("--ocr-fix", action="store_true", help="Repair OCR errors in rebuilt books with DeepSeek")
    parser.add_argument("--min-words", type=int, default=MIN_WORDS)
    args = parser.parse_args()

    paragraphs = [p.strip() for p in args.input.read_text(encoding="utf-8").split("\n\n") if p.strip()]
    print(f"Loaded {len(paragraphs)} paragraphs")

    rebuilt_ids = set()
    for raw_path in args.rebuild:
        raw = raw_path.read_text(encoding="utf-8")
        rebuilt = rebuild_ocr_book(raw, min_words=args.min_words)
        print(f"Rebuilt {raw_path.name}: {len(rebuilt)} paragraphs")
        if args.ocr_fix:
            cache = raw_path.with_suffix(".ocrfix.json")
            rebuilt = fix_ocr(rebuilt, deepseek_caller(), cache)
        rebuilt_ids.update(rebuilt)
        paragraphs = replace_book(paragraphs, raw, rebuilt)

    cleaned, seen, dropped = [], set(), 0
    for para in paragraphs:
        result = clean_paragraph(para, min_words=args.min_words)
        if result is None or result in seen:
            dropped += 1
        else:
            seen.add(result)
            cleaned.append(result)

    out = args.output or args.input
    out.write_text("\n\n".join(cleaned) + "\n", encoding="utf-8")
    words = sum(len(p.split()) for p in cleaned)
    print(f"Wrote {len(cleaned)} paragraphs ({words:,} words) to {out}; dropped {dropped}")


if __name__ == "__main__":
    main()
