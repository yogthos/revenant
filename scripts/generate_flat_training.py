#!/usr/bin/env python3
"""Generate training data using LLM-based lossless neutralization.

Pipeline:
1. Curate corpus: Clean text, extract quality paragraphs
2. Topic variation: Create topic-varied versions via DeepSeek (optional)
3. Overlapping chunks: Create sliding window chunks across paragraph boundaries
4. Lossless neutralization: LLM paraphrase preserving ALL facts
5. Training pairs: (neutral paraphrase) → (styled text)

Key insight: The neutral input must contain ALL facts from the output.
OpenIE triple extraction is lossy - it teaches memorization, not style transfer.
LLM-based neutralization preserves every fact in simple neutral language.

Usage:
    python scripts/generate_flat_training.py \
        --corpus data/corpus/lovecraft.txt \
        --author "H.P. Lovecraft" \
        --output data/training/lovecraft

    # Resume from chunks file
    python scripts/generate_flat_training.py \
        --resume-from-chunks data/training/lovecraft_chunks.json \
        --author "H.P. Lovecraft" \
        --output data/training/lovecraft
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CurationConfig:
    """Configuration for corpus curation."""
    min_words: int = 100
    max_words: int = 650
    min_sentences: int = 2
    max_special_char_ratio: float = 0.10
    max_word_repetition_ratio: float = 0.55


@dataclass
class OverlapConfig:
    """Configuration for overlapping chunks.

    Based on research showing "style lives in transitions" - stylistic markers
    concentrate at chunk boundaries. Using smaller chunks with overlap exposes
    the model to more beginning/ending patterns.

    See: https://muratcankoylan.com/projects/gertrude-stein-style-training
    """
    min_words: int = 150  # Minimum words per chunk
    max_words: int = 400  # Maximum words per chunk
    overlap_sentences: int = 2  # Sentences to overlap between chunks


# =============================================================================
# DeepSeek API (for fact variation)
# =============================================================================

def call_deepseek(prompt: str, system: str = "", max_retries: int = 3) -> str:
    """Call DeepSeek API."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable not set")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 2048
                },
                timeout=90
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise


# =============================================================================
# Step 1: Corpus Curation
# =============================================================================

_nlp = None

def get_nlp():
    """Get or load spaCy model (singleton pattern)."""
    global _nlp
    if _nlp is None:
        import spacy
        logger.info("Loading spaCy model...")
        _nlp = spacy.load("en_core_web_sm")
        logger.info("spaCy model loaded")
    return _nlp


def clean_text(text: str) -> str:
    """Clean raw corpus text."""
    text = re.sub(r'^Chapter \d+.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^Page \d+.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.replace(''', "'").replace(''', "'")
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace('—', '-').replace('–', '-')
    text = text.replace('…', '...')
    return text.strip()


def split_into_sentences(text: str, nlp=None) -> List[str]:
    """Split text into sentences using spaCy."""
    if nlp is None:
        nlp = get_nlp()
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def count_special_chars(text: str) -> float:
    if not text:
        return 0.0
    special = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return special / len(text)


def count_word_repetition(text: str) -> float:
    words = text.lower().split()
    if len(words) < 10:
        return 0.0
    unique = len(set(words))
    return 1 - (unique / len(words))


def is_quality_paragraph(para: str, config: CurationConfig, nlp=None) -> Tuple[bool, str]:
    """Check if paragraph meets quality criteria."""
    words = para.split()
    word_count = len(words)

    if word_count < config.min_words:
        return False, f"Too short ({word_count} < {config.min_words} words)"
    if word_count > config.max_words:
        return False, f"Too long ({word_count} > {config.max_words} words)"

    sentences = split_into_sentences(para, nlp)
    if len(sentences) < config.min_sentences:
        return False, f"Too few sentences ({len(sentences)} < {config.min_sentences})"

    special_ratio = count_special_chars(para)
    if special_ratio > config.max_special_char_ratio:
        return False, f"Too many special chars ({special_ratio:.1%})"

    repetition_ratio = count_word_repetition(para)
    if repetition_ratio > config.max_word_repetition_ratio:
        return False, f"Too much repetition ({repetition_ratio:.1%})"

    return True, "OK"


def extract_paragraphs(text: str, config: CurationConfig) -> List[str]:
    """Extract quality paragraphs from text."""
    start_time = time.time()
    logger.info("Cleaning text...")
    text = clean_text(text)
    raw_paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    logger.info(f"Found {len(raw_paragraphs)} raw paragraphs")

    nlp = get_nlp()
    quality_paragraphs = []

    for i, para in enumerate(raw_paragraphs):
        is_good, reason = is_quality_paragraph(para, config, nlp)
        if is_good:
            quality_paragraphs.append(para)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            logger.info(f"Curation progress: {i + 1}/{len(raw_paragraphs)} | Kept: {len(quality_paragraphs)} | Rate: {rate:.0f}/s")

    elapsed = time.time() - start_time
    pct = len(quality_paragraphs) / len(raw_paragraphs) * 100 if raw_paragraphs else 0
    logger.info(f"Curation complete: {len(quality_paragraphs)}/{len(raw_paragraphs)} paragraphs kept ({pct:.1f}%) in {elapsed:.1f}s")
    return quality_paragraphs


# =============================================================================
# Step 2: Topic Variation (1-to-N Expansion)
# =============================================================================
# For each original paragraph, generate variations on different topics while
# preserving the author's style. This prevents "content overfitting" - the model
# learns to transform structure/style rather than memorizing specific content.
#
# Categories:
# - Concrete: Physical objects (mechanical, tangible verbs)
# - Abstract: Philosophical concepts (intellectual verbs)
# - Action: Physical activities (motion, process verbs)

# Topic pools for variation generation
CONCRETE_TOPICS = [
    "an old clocktower", "a lighthouse on a cliff", "a rusted locomotive",
    "an ancient library", "a abandoned factory", "a Victorian greenhouse",
    "a cathedral's pipe organ", "a deep-sea submarine", "a medieval castle",
    "a decaying mansion", "a stone bridge", "a forgotten well",
    "an antique telescope", "a printing press", "a grandfather clock",
]

ABSTRACT_TOPICS = [
    "the nature of time", "the concept of justice", "the illusion of free will",
    "the meaning of consciousness", "the paradox of infinity", "the weight of memory",
    "the architecture of dreams", "the entropy of civilization", "the geometry of thought",
    "the mathematics of beauty", "the philosophy of darkness", "the essence of fear",
    "the structure of chaos", "the fabric of reality", "the limits of knowledge",
]

ACTION_TOPICS = [
    "descending into a cave", "navigating a storm at sea", "climbing a mountain",
    "exploring ancient ruins", "crossing a frozen lake", "wandering through fog",
    "escaping a collapsing building", "pursuing something through shadows",
    "excavating a forgotten tomb", "assembling a complex machine", "decoding an ancient text",
    "tracking something through a forest", "mapping an uncharted territory",
]


def validate_variation(original: str, varied: str, nlp=None) -> Tuple[bool, str]:
    """Validate that topic variation preserved approximate structure."""
    orig_words = len(original.split())
    varied_words = len(varied.split())

    # Allow 20% word count variance for topic changes
    if abs(orig_words - varied_words) > orig_words * 0.20:
        return False, f"Word count too different: {orig_words} vs {varied_words}"

    if nlp is None:
        nlp = get_nlp()
    orig_sentences = len(split_into_sentences(original, nlp))
    varied_sentences = len(split_into_sentences(varied, nlp))

    # Sentence count should be close
    if abs(orig_sentences - varied_sentences) > 1:
        return False, f"Sentence count mismatch: {orig_sentences} vs {varied_sentences}"

    # Check it's not just the original with minor changes
    orig_words_set = set(original.lower().split())
    varied_words_set = set(varied.lower().split())
    overlap = len(orig_words_set & varied_words_set) / len(orig_words_set) if orig_words_set else 0

    if overlap > 0.85:
        return False, f"Too similar to original ({overlap:.0%} overlap)"

    return True, "OK"


def create_topic_variation(
    paragraph: str,
    author: str,
    topic: str,
    category: str,
    max_attempts: int = 2
) -> Optional[str]:
    """Create a topic variation that maintains the author's style.

    The variation should:
    1. Be about the new topic (completely different subject matter)
    2. Maintain the author's sentence structure, rhythm, and vocabulary
    3. Be FACTUALLY ACCURATE about the new topic (no false statements)
    """
    system = f"""You are a literary style transfer assistant specializing in {author}'s writing style.

Your task: Rewrite the given passage to be about a completely different topic while preserving:
- The EXACT sentence structure and rhythm
- The author's characteristic vocabulary and word choices
- The same emotional tone and atmosphere
- The same level of descriptive complexity

CRITICAL RULE: All statements about the new topic MUST be factually accurate. If the rhythm
demands an adjective that would be false, choose a different true adjective. Factual accuracy
is more important than perfect rhythm preservation."""

    prompt = f"""Rewrite this passage by {author} to be about "{topic}" ({category} topic).

Original passage:
{paragraph}

Requirements:
1. The new passage must be ENTIRELY about "{topic}" - no traces of the original subject
2. Preserve the sentence structure, rhythm, and {author}'s distinctive style
3. ALL facts about "{topic}" must be TRUE - verify your statements are accurate
4. Match approximately the same word count ({len(paragraph.split())} words)

Output only the rewritten passage, nothing else."""

    for attempt in range(max_attempts):
        try:
            varied = call_deepseek(prompt, system, max_retries=2)
            varied = varied.strip('`"\' \n')
            if varied.startswith('```'):
                varied = re.sub(r'^```\w*\n?', '', varied)
                varied = re.sub(r'\n?```$', '', varied)

            # Validate the variation
            is_valid, reason = validate_variation(paragraph, varied)
            if is_valid:
                return varied
            else:
                logger.debug(f"Variation rejected: {reason}")
        except Exception as e:
            logger.debug(f"Variation attempt {attempt+1} failed: {e}")

    return None


def save_intermediate(items: List[Tuple[str, str]], path: Path, stage: str = "items") -> None:
    """Save intermediate data to JSON file.

    Items are tuples of (text, variation_type) where variation_type is:
    - 'original': Original author text
    - 'concrete': Concrete topic variation
    - 'abstract': Abstract topic variation
    - 'action': Action topic variation
    """
    data = [{"text": text, "variation_type": vtype} for text, vtype in items]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(items)} {stage} to {path}")


def load_intermediate(path: Path, stage: str = "items") -> List[Tuple[str, str]]:
    """Load intermediate data from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Handle both old format (is_varied bool) and new format (variation_type str)
    items = []
    for item in data:
        if "variation_type" in item:
            items.append((item["text"], item["variation_type"]))
        else:
            # Old format compatibility
            vtype = "varied" if item.get("is_varied", False) else "original"
            items.append((item["text"], vtype))
    logger.info(f"Loaded {len(items)} {stage} from {path}")
    return items


def expand_corpus_with_variations(
    paragraphs: List[str],
    author: str,
    workers: int = 4,
    skip_variation: bool = False,
    variations_per_paragraph: int = 3
) -> List[Tuple[str, str]]:
    """Expand corpus with topic variations using 1-to-N expansion.

    For each original paragraph, generates variations on different topics
    (concrete, abstract, action) while preserving the author's style.

    This prevents content overfitting - the model learns style transformation
    rather than memorizing specific content.

    Args:
        paragraphs: Original author paragraphs
        author: Author name for style preservation
        workers: Number of parallel workers
        skip_variation: If True, skip variation generation
        variations_per_paragraph: Number of variations per original (max 3)

    Returns:
        List of (text, variation_type) tuples
    """
    # Always include originals
    result = [(para, "original") for para in paragraphs]

    if skip_variation:
        logger.info("Skipping topic variation (--skip-variation flag)")
        return result

    # Determine which categories to use
    categories = [
        ("concrete", CONCRETE_TOPICS),
        ("abstract", ABSTRACT_TOPICS),
        ("action", ACTION_TOPICS),
    ][:variations_per_paragraph]

    total_variations = len(paragraphs) * len(categories)
    logger.info(f"Creating {len(categories)} topic variations per paragraph...")
    logger.info(f"Total variations to generate: {total_variations}")

    # Prepare all variation tasks
    tasks = []
    for idx, para in enumerate(paragraphs):
        for cat_name, topic_pool in categories:
            # Select a random topic from the pool
            topic = random.choice(topic_pool)
            tasks.append((idx, para, topic, cat_name))

    variation_count = 0
    failed_count = 0
    start_time = time.time()

    def process_variation(task):
        idx, para, topic, category = task
        varied = create_topic_variation(para, author, topic, category)
        return idx, varied, category

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_variation, task): task for task in tasks}

        for future in as_completed(futures):
            try:
                idx, varied, category = future.result()
                if varied:
                    result.append((varied, category))
                    variation_count += 1
                else:
                    failed_count += 1

                total_processed = variation_count + failed_count
                if total_processed % 20 == 0:
                    elapsed = time.time() - start_time
                    rate = total_processed / elapsed if elapsed > 0 else 0
                    success_rate = variation_count / total_processed * 100 if total_processed > 0 else 0
                    logger.info(
                        f"Variations: {variation_count}/{total_variations} | "
                        f"Failed: {failed_count} | "
                        f"Success: {success_rate:.0f}% | "
                        f"Rate: {rate:.1f}/s"
                    )
            except Exception as e:
                failed_count += 1
                logger.debug(f"Variation task failed: {e}")

    elapsed = time.time() - start_time
    logger.info(
        f"Variation complete: {variation_count} created, {failed_count} failed "
        f"in {elapsed:.1f}s ({variation_count/elapsed:.1f}/s)"
    )

    # Log category breakdown
    category_counts = {}
    for _, vtype in result:
        category_counts[vtype] = category_counts.get(vtype, 0) + 1
    logger.info(f"Category breakdown: {category_counts}")

    return result


# =============================================================================
# Step 3: Overlapping Chunks (Style Lives in Transitions)
# =============================================================================
# Research shows stylistic markers concentrate at chunk boundaries.
# By creating smaller, overlapping chunks that cross paragraph boundaries,
# we expose the model to more beginning/ending patterns where style lives.
#
# Key insight: Same source material → more training examples by restructuring,
# not by adding content.

def create_overlapping_chunks(paragraphs: List[Tuple[str, str]], config: OverlapConfig) -> List[Tuple[str, str]]:
    """Create overlapping chunks that cross paragraph boundaries.

    IMPORTANT: Overlapping only applies to ORIGINALS (continuous narrative).
    Variations are kept as separate paragraphs because each was rewritten
    to a DIFFERENT topic - combining them creates "Frankenstein" narratives.

    Args:
        paragraphs: List of (text, variation_type) tuples
        config: Overlap configuration (min/max words, overlap sentences)

    Returns:
        List of (chunk_text, variation_type) tuples
    """
    start_time = time.time()
    nlp = get_nlp()

    # Group paragraphs by variation_type
    by_type = {}
    for text, vtype in paragraphs:
        if vtype not in by_type:
            by_type[vtype] = []
        by_type[vtype].append(text)

    logger.info(f"Processing {len(paragraphs)} paragraphs into chunks...")
    logger.info(f"Variation types: {list(by_type.keys())}")
    logger.info(f"Target chunk size: {config.min_words}-{config.max_words} words")
    logger.info(f"NOTE: Overlapping only for 'original' type (continuous narrative)")

    all_chunks = []

    for vtype, texts in by_type.items():
        # For VARIATIONS: Keep paragraphs separate (each is about a different topic)
        # Combining them would create "Frankenstein" narratives
        if vtype != "original":
            chunks_for_type = []
            for para_text in texts:
                word_count = len(para_text.split())
                # Only include if it meets minimum size
                if word_count >= config.min_words:
                    chunks_for_type.append((para_text, vtype))
                elif word_count >= config.min_words * 0.7:
                    # Include slightly smaller ones too
                    chunks_for_type.append((para_text, vtype))

            all_chunks.extend(chunks_for_type)
            logger.info(f"  {vtype}: {len(texts)} paragraphs -> {len(chunks_for_type)} chunks (no overlap)")
            continue

        # For ORIGINALS: Use sliding window with overlap (continuous narrative)
        sentences = []
        for para_idx, para_text in enumerate(texts):
            para_sentences = split_into_sentences(para_text, nlp)
            for sent_idx, sent in enumerate(para_sentences):
                word_count = len(sent.split())
                is_para_start = (sent_idx == 0)
                is_para_end = (sent_idx == len(para_sentences) - 1)
                sentences.append({
                    'text': sent,
                    'words': word_count,
                    'para_start': is_para_start,
                    'para_end': is_para_end,
                    'para_idx': para_idx,
                })

        if not sentences:
            continue

        logger.info(f"  {vtype}: {len(texts)} paragraphs -> {len(sentences)} sentences")

        # Create chunks using sliding window
        chunks_for_type = []
        i = 0
        while i < len(sentences):
            # Build a chunk starting at sentence i
            chunk_sentences = []
            chunk_words = 0
            j = i

            # Add sentences until we reach max_words or run out
            while j < len(sentences) and chunk_words < config.max_words:
                sent = sentences[j]
                # Don't exceed max_words by too much
                if chunk_words + sent['words'] > config.max_words * 1.1 and chunk_words >= config.min_words:
                    break
                chunk_sentences.append(sent)
                chunk_words += sent['words']
                j += 1

            # Only keep chunk if it meets minimum size
            if chunk_words >= config.min_words:
                chunk_text = ' '.join(s['text'] for s in chunk_sentences)
                chunks_for_type.append((chunk_text, vtype))

                # Count paragraph transitions in this chunk (for stats)
                transitions = sum(1 for s in chunk_sentences if s['para_start'] and chunk_sentences.index(s) > 0)

            # Move start position: advance by (sentences_used - overlap)
            # This creates the overlap where style lives
            sentences_used = len(chunk_sentences)
            step = max(1, sentences_used - config.overlap_sentences)
            i += step

            # If we're near the end and would create a tiny chunk, just stop
            remaining_words = sum(s['words'] for s in sentences[i:])
            if remaining_words < config.min_words * 0.5:
                break

        all_chunks.extend(chunks_for_type)
        logger.info(f"  {vtype}: created {len(chunks_for_type)} chunks (with {config.overlap_sentences}-sentence overlap)")

    elapsed = time.time() - start_time

    # Stats
    if all_chunks:
        word_counts = [len(c[0].split()) for c in all_chunks]
        avg_words = sum(word_counts) / len(word_counts)
        min_words = min(word_counts)
        max_words = max(word_counts)
        logger.info(f"Chunking complete: {len(all_chunks)} chunks in {elapsed:.1f}s")
        logger.info(f"Chunk sizes: avg={avg_words:.0f}, min={min_words}, max={max_words} words")

    return all_chunks


# =============================================================================
# Step 4: Round-Trip Translation Neutralization (The "Linguistic Laundromat")
# =============================================================================
# Round-Trip Translation through Mandarin is the ultimate style scrubber:
#
# 1. English → Mandarin: Grammar distance forces syntax flattening
#    - Mandarin's Topic-Prominent structure can't support nested clauses
#    - HSK3 vocabulary constraint strips literary words
#    - No cognates means no fancy word preservation
#
# 2. Mandarin → English: Restores natural English but without style
#    - Produces "natural but plain" text (not robotic graph output)
#    - Perfect for teaching model to "elevate" prose
#
# Uses local MLX model (Qwen2.5-3B-Instruct) for fast inference without API calls.
# Configuration in config.json under llm.providers.mlx_rtt.

# Global RTT neutralizer (shared across threads)
_rtt_neutralizer = None


def get_rtt_neutralizer():
    """Get or create shared RTT neutralizer (singleton pattern)."""
    global _rtt_neutralizer
    if _rtt_neutralizer is None:
        from src.llm.mlx_provider import RTTNeutralizer
        logger.info("Initializing RTT neutralizer (local MLX model)...")
        _rtt_neutralizer = RTTNeutralizer()
        logger.info("RTT neutralizer ready")
    return _rtt_neutralizer


def neutralize_text(styled_text: str, max_retries: int = 2) -> Optional[str]:
    """Round-Trip Translation neutralization via Mandarin pivot.

    Step 1 (Scrub): English → Mandarin (HSK3 vocabulary)
    Step 2 (Rinse): Mandarin → Plain English

    Uses local MLX model (Qwen2.5-3B-Instruct) for fast inference.

    Args:
        styled_text: The styled text to neutralize
        max_retries: Number of retry attempts

    Returns:
        Neutral English with all facts preserved, or None if failed
    """
    try:
        neutralizer = get_rtt_neutralizer()
        return neutralizer.neutralize(styled_text, max_retries=max_retries)
    except Exception as e:
        logger.error(f"RTT neutralization failed: {e}")
        return None


# =============================================================================
# Step 5: Training Data Generation
# =============================================================================

def format_training_example(neutral_text: str, styled_text: str, author: str, word_count: int) -> dict:
    """Format a training example for BASE model (not instruct).

    Base models need raw text completion format, not chat roles.
    The model learns: given neutral text + instruction → produce styled text.

    Format:
        Rewrite in {author}'s style (~N words):
        [neutral text]
        ###
        [styled text]
    """
    # Simple prompt format for base model
    prompt = f"Rewrite in {author}'s style (~{word_count} words):\n{neutral_text}\n###\n"

    # For mlx_lm.lora with base models: {"text": "prompt + completion"}
    # With mask_prompt=true, only the completion (after prompt) is trained
    return {
        "text": prompt + styled_text,
        "prompt": prompt,  # For mask_prompt to know where to split
        "word_count": word_count
    }


def generate_training_data(
    chunks: List[Tuple[str, str]],
    author: str,
    output_path: Path,
    workers: int = 4,
) -> int:
    """Generate training data using RTT neutralization, writing sequentially.

    Processes chunks in parallel but writes results immediately as they complete.
    This ensures progress is saved even if the process is interrupted.

    Args:
        chunks: List of (styled_text, variation_type) tuples
        author: Author name for system prompt
        output_path: Output JSONL file path
        workers: Number of parallel workers for neutralization

    Returns:
        Number of examples written
    """
    import threading

    total = len(chunks)
    logger.info(f"Generating training data for {total} chunks with {workers} workers...")
    logger.info(f"Writing sequentially to {output_path} (progress saved on interrupt)...")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failed_count = 0
    type_counts = {}
    start_time = time.time()
    write_lock = threading.Lock()

    def process_and_write(item, f):
        nonlocal success_count, failed_count
        idx, (styled_text, vtype) = item

        # RTT neutralization (2 API calls)
        neutral = neutralize_text(styled_text)

        if neutral:
            word_count = len(styled_text.split())
            example = format_training_example(
                neutral_text=neutral,
                styled_text=styled_text,
                author=author,
                word_count=word_count
            )
            example["variation_type"] = vtype
            example["source_idx"] = idx

            # Write immediately with lock
            with write_lock:
                f.write(json.dumps(example) + '\n')
                f.flush()  # Ensure written to disk
                success_count += 1
                type_counts[vtype] = type_counts.get(vtype, 0) + 1
        else:
            with write_lock:
                failed_count += 1

        return neutral is not None

    with open(output_path, 'w', encoding='utf-8') as f:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_and_write, (i, chunk), f): i
                       for i, chunk in enumerate(chunks)}

            for future in as_completed(futures):
                try:
                    future.result()

                    total_processed = success_count + failed_count
                    if total_processed % 50 == 0:
                        elapsed = time.time() - start_time
                        rate = total_processed / elapsed if elapsed > 0 else 0
                        pct = total_processed / total * 100
                        eta = (total - total_processed) / rate if rate > 0 else 0
                        logger.info(
                            f"Progress: {total_processed}/{total} ({pct:.0f}%) | "
                            f"Success: {success_count} | Failed: {failed_count} | "
                            f"Rate: {rate:.1f}/s | ETA: {eta/60:.1f}m"
                        )
                except Exception as e:
                    with write_lock:
                        failed_count += 1
                    logger.debug(f"Task failed: {e}")

    elapsed = time.time() - start_time
    logger.info(
        f"Complete: {success_count}/{total} examples written "
        f"({success_count/total*100:.1f}%) in {elapsed:.1f}s"
    )
    logger.info(f"By variation type: {type_counts}")
    return success_count


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate training data using OpenIE-based flattening (fast, no GPU)"
    )
    parser.add_argument("--corpus", required=False, help="Path to author corpus file")
    parser.add_argument("--author", required=True, help="Author name")
    parser.add_argument("--output", required=True, help="Output directory for training data")
    parser.add_argument("--workers", type=int, default=4, help="Workers for fact variation")
    parser.add_argument("--skip-variation", action="store_true", help="Skip topic variation step")
    parser.add_argument("--variations", type=int, default=3, choices=[1, 2, 3],
                        help="Number of topic variations per paragraph (1-3, default: 3)")
    parser.add_argument("--max-paragraphs", type=int, default=None, help="Max paragraphs to process")
    parser.add_argument("--min-para-words", type=int, default=100, help="Min words per paragraph (curation)")
    parser.add_argument("--max-para-words", type=int, default=650, help="Max words per paragraph (curation)")
    parser.add_argument("--min-chunk-words", type=int, default=150, help="Min words per chunk (default: 150)")
    parser.add_argument("--max-chunk-words", type=int, default=400, help="Max words per chunk (default: 400)")
    parser.add_argument("--overlap-sentences", type=int, default=2, help="Sentence overlap between chunks")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from intermediate JSON (skips Steps 1-2)")
    parser.add_argument("--resume-from-chunks", type=str, default=None, help="Resume from chunks JSON (skips Steps 1-3)")
    parser.add_argument("--save-intermediate", type=str, default=None, help="Save intermediate paragraphs")

    args = parser.parse_args()

    if not args.resume_from and not args.resume_from_chunks and not args.corpus:
        parser.error("--corpus is required unless using --resume-from or --resume-from-chunks")

    overall_start = time.time()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    variation_counts = {}

    # Check if resuming from chunks file (skips Steps 1-3)
    if args.resume_from_chunks:
        logger.info("=" * 60)
        logger.info("RESUMING FROM CHUNKS FILE (skipping Steps 1-3)")
        logger.info("=" * 60)

        chunks = load_intermediate(Path(args.resume_from_chunks), stage="chunks")
        for _, vtype in chunks:
            variation_counts[vtype] = variation_counts.get(vtype, 0) + 1
        logger.info(f"Chunks: {len(chunks)} total | Breakdown: {variation_counts}")

    # Check if resuming from intermediate file (skips Steps 1-2)
    elif args.resume_from:
        logger.info("=" * 60)
        logger.info("RESUMING FROM INTERMEDIATE FILE (skipping Steps 1-2)")
        logger.info("=" * 60)

        expanded = load_intermediate(Path(args.resume_from), stage="paragraphs")
        for _, vtype in expanded:
            variation_counts[vtype] = variation_counts.get(vtype, 0) + 1
        logger.info(f"Corpus: {len(expanded)} total | Breakdown: {variation_counts}")

        # Step 3: Overlapping chunks
        logger.info("=" * 60)
        logger.info("STEP 3: Creating overlapping chunks")
        logger.info("=" * 60)

        overlap_config = OverlapConfig(
            min_words=args.min_chunk_words,
            max_words=args.max_chunk_words,
            overlap_sentences=args.overlap_sentences
        )
        chunks = create_overlapping_chunks(expanded, overlap_config)
        logger.info(f"Created {len(chunks)} overlapping chunks")

        chunks_path = output_dir / "chunks.json"
        save_intermediate(chunks, chunks_path, stage="chunks")

    else:
        # Full pipeline from corpus
        logger.info(f"Loading corpus: {args.corpus}")
        with open(args.corpus, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        # Step 1: Curate corpus
        logger.info("=" * 60)
        logger.info("STEP 1: Curating corpus")
        logger.info("=" * 60)

        curation_config = CurationConfig(min_words=args.min_para_words, max_words=args.max_para_words)
        paragraphs = extract_paragraphs(raw_text, curation_config)

        if args.max_paragraphs:
            paragraphs = paragraphs[:args.max_paragraphs]

        logger.info(f"Extracted {len(paragraphs)} quality paragraphs")

        # Step 2: Topic variation (1-to-N expansion)
        logger.info("=" * 60)
        logger.info("STEP 2: Creating topic variations (1-to-N expansion)")
        logger.info("=" * 60)

        expanded = expand_corpus_with_variations(
            paragraphs,
            author=args.author,
            workers=args.workers,
            skip_variation=args.skip_variation,
            variations_per_paragraph=args.variations
        )

        for _, vtype in expanded:
            variation_counts[vtype] = variation_counts.get(vtype, 0) + 1
        logger.info(f"Corpus: {len(expanded)} total | Breakdown: {variation_counts}")

        if args.save_intermediate:
            save_intermediate(expanded, Path(args.save_intermediate), stage="paragraphs")
        else:
            intermediate_path = output_dir / "paragraphs.json"
            save_intermediate(expanded, intermediate_path, stage="paragraphs")

        # Step 3: Overlapping chunks
        logger.info("=" * 60)
        logger.info("STEP 3: Creating overlapping chunks")
        logger.info("=" * 60)

        overlap_config = OverlapConfig(
            min_words=args.min_chunk_words,
            max_words=args.max_chunk_words,
            overlap_sentences=args.overlap_sentences
        )
        chunks = create_overlapping_chunks(expanded, overlap_config)
        logger.info(f"Created {len(chunks)} overlapping chunks")

        chunks_path = output_dir / "chunks.json"
        save_intermediate(chunks, chunks_path, stage="chunks")

    # Step 4: LLM-based lossless neutralization + training data generation
    logger.info("=" * 60)
    logger.info("STEP 4: Lossless neutralization and training data generation")
    logger.info("=" * 60)

    all_output_path = output_dir / "all.jsonl"
    num_examples = generate_training_data(chunks, args.author, all_output_path, workers=args.workers)

    # Create train/valid/test split for mlx_lm.lora
    logger.info("=" * 60)
    logger.info("STEP 5: Creating train/valid/test split")
    logger.info("=" * 60)

    # Load all examples from output file
    all_examples = []
    with open(all_output_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_examples.append(json.loads(line))

    # 80% train, 10% valid, 10% test
    random.seed(42)  # Reproducible splits

    shuffled = all_examples.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * 0.8)
    n_valid = int(n * 0.1)

    train_examples = shuffled[:n_train]
    valid_examples = shuffled[n_train:n_train + n_valid]
    test_examples = shuffled[n_train + n_valid:]

    # Write train/valid/test files (messages format only, no metadata)
    for split_name, examples in [("train", train_examples), ("valid", valid_examples), ("test", test_examples)]:
        split_path = output_dir / f"{split_name}.jsonl"
        with open(split_path, 'w', encoding='utf-8') as f:
            for ex in examples:
                # Only keep messages, strip metadata
                f.write(json.dumps({"messages": ex["messages"]}) + '\n')
        logger.info(f"{split_name}: {len(examples)} examples -> {split_path}")

    # Summary
    total_time = time.time() - overall_start
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Author: {args.author}")
    logger.info(f"Variation breakdown: {variation_counts}")
    logger.info(f"Overlapping chunks: {len(chunks)}")
    logger.info(f"Training examples: {num_examples}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"  all.jsonl: {num_examples} examples (with metadata)")
    logger.info(f"  train.jsonl: {len(train_examples)}, valid.jsonl: {len(valid_examples)}, test.jsonl: {len(test_examples)}")
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")


if __name__ == "__main__":
    main()
