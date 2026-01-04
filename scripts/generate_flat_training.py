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

import threading

_nlp = None
_nlp_lock = threading.Lock()

def get_nlp():
    """Get or load spaCy model (thread-safe singleton)."""
    global _nlp
    if _nlp is None:
        with _nlp_lock:
            # Double-check after acquiring lock
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
# Step 2: Topic Variation - The "Triad" Strategy
# =============================================================================
# For each original paragraph, generate exactly 2 variations for a 1:3 ratio:
#
# Entry 1 (Anchor): Original author text
#   - Purpose: Teaches vocabulary and how author describes their actual subjects
#   - Input: Neutral summary of original content
#   - Output: Real author text
#
# Entry 2 (Snowflake): Topic swap to mundane subject
#   - Purpose: Teaches that sentence STRUCTURE applies to everything
#   - Method: Rewrite about everyday topic (making toast, filing taxes) keeping structure
#   - Input: Neutral summary of mundane topic
#   - Output: "Author-style Toast Making" (Synthetic)
#
# Entry 3 (Robustness): Input perturbation (NEFTune simulation)
#   - Purpose: Prevents overfitting to specific input words
#   - Method: Take Entry 1 or 2, heavily corrupt the INPUT only
#   - Input: "The enginer fixd machine qickly." (Messy)
#   - Output: Clean author/synthetic text
#
# This gives 3 entries per paragraph (1:3 ratio) - optimal for training volume
# without drowning out the real author signal (33% real vs 67% synthetic)

# Topic pools for variation generation - MUNDANE topics to isolate style from content
# Using everyday activities forces the model to learn structure, not subject matter
MUNDANE_TOPICS = [
    # Domestic activities
    "making toast for breakfast", "doing the weekly laundry", "organizing a closet",
    "washing dishes after dinner", "vacuuming the living room", "folding clean towels",
    "watering houseplants", "making the bed", "cleaning the bathroom mirror",
    "sorting through old mail", "replacing a lightbulb", "taking out the trash",
    # Office/bureaucratic
    "filing tax returns", "attending a staff meeting", "writing a work email",
    "waiting in line at the DMV", "filling out insurance forms", "updating a spreadsheet",
    "scheduling a dentist appointment", "renewing a driver's license", "balancing a checkbook",
    # Routine errands
    "grocery shopping on Saturday", "pumping gas at the station", "returning library books",
    "picking up dry cleaning", "waiting for a bus", "walking to the mailbox",
    "parallel parking downtown", "choosing produce at the market", "standing in the checkout line",
    # Simple activities
    "brewing morning coffee", "tying shoelaces", "checking the weather forecast",
    "setting an alarm clock", "microwaving leftovers", "charging a phone overnight",
    "brushing teeth before bed", "packing a lunch", "feeding the cat",
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
    max_attempts: int = 2
) -> Optional[str]:
    """Create a topic variation (Snowflake) that maintains the author's style.

    The Snowflake variation teaches that sentence STRUCTURE applies to any topic.
    By using mundane everyday topics, the model can't rely on subject matter
    similarity - it must learn the structural patterns.

    The variation should:
    1. Be about the mundane topic (completely different subject matter)
    2. Maintain the EXACT sentence structure and rhythm
    3. Use the author's characteristic vocabulary patterns
    """
    system = f"""You are a literary style transfer assistant specializing in {author}'s writing style.

Your task: Rewrite the given passage to be about a mundane everyday topic while preserving:
- The EXACT sentence structure (same number of sentences, same clause patterns)
- The author's characteristic rhythm and cadence
- Similar vocabulary complexity and word choices
- The same punctuation patterns (semicolons, dashes, parentheticals)

The goal is to prove that {author}'s STYLE can make even mundane activities sound distinctive."""

    prompt = f"""Rewrite this passage by {author} to be about "{topic}".

Original passage:
{paragraph}

Requirements:
1. The new passage must be ENTIRELY about "{topic}" - a mundane everyday activity
2. Preserve the EXACT sentence structure: {len(paragraph.split('.'))} sentences, same clause patterns
3. Use {author}'s distinctive vocabulary and phrasing style
4. Match the word count closely (~{len(paragraph.split())} words)
5. Keep the same punctuation patterns and rhythm

The result should sound unmistakably like {author} writing about {topic}.

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


def create_heavy_perturbation(text: str, perturbation_rate: float = 0.15) -> str:
    """Apply heavy perturbations for Robustness entries (Entry 3 of Triad).

    This is stronger than the light perturbation applied to all examples.
    Simulates NEFTune by heavily corrupting the input while keeping output clean.

    Applies ~15% random changes:
    - Synonym swap: Replace word with synonym
    - Word drop: Remove articles and filler words
    - Typo: Swap adjacent characters
    - Case errors: Random case changes

    Args:
        text: Input text to perturb
        perturbation_rate: Probability of perturbing each word (default 15%)

    Returns:
        Heavily perturbed text
    """
    words = text.split()
    result = []
    droppable = {'the', 'a', 'an', 'very', 'really', 'just', 'quite', 'some', 'this', 'that'}

    for word in words:
        if random.random() > perturbation_rate:
            result.append(word)
            continue

        # Choose perturbation type
        choice = random.random()

        if choice < 0.30:
            # Synonym swap (30% of perturbations)
            word_lower = word.lower().rstrip('.,!?;:')
            if word_lower in SYNONYMS:
                synonym = random.choice(SYNONYMS[word_lower])
                # Preserve case
                if word[0].isupper():
                    synonym = synonym.capitalize()
                result.append(synonym + word[len(word_lower):])
            else:
                result.append(word)

        elif choice < 0.50:
            # Word drop (20% of perturbations)
            if word.lower() in droppable:
                pass  # Drop the word
            else:
                result.append(word)

        elif choice < 0.75:
            # Typo - swap two adjacent chars (25% of perturbations)
            if len(word) > 3:
                i = random.randint(1, len(word) - 2)
                word = word[:i] + word[i+1] + word[i] + word[i+2:]
            result.append(word)

        elif choice < 0.90:
            # Double letter typo (15% of perturbations)
            if len(word) > 2:
                i = random.randint(0, len(word) - 1)
                word = word[:i] + word[i] + word[i:]
            result.append(word)

        else:
            # Case error (10% of perturbations)
            if random.random() < 0.5:
                result.append(word.lower())
            else:
                result.append(word.upper() if len(word) <= 4 else word)

    return ' '.join(result)


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
) -> List[Tuple[str, str]]:
    """Expand corpus using the Triad strategy (1:3 ratio).

    For each original paragraph, creates exactly 2 variations:
    - Entry 1 (Anchor): Original author text
    - Entry 2 (Snowflake): Topic swap to mundane activity
    - Entry 3 (Robustness): Marked for heavy input perturbation later

    The Robustness entry reuses the original text but will have heavy
    perturbation applied to its INPUT during training data generation.
    This is more efficient than generating another LLM variation.

    Args:
        paragraphs: Original author paragraphs
        author: Author name for style preservation
        workers: Number of parallel workers
        skip_variation: If True, skip variation generation (originals only)

    Returns:
        List of (text, variation_type) tuples where variation_type is:
        - 'original': Real author text (Entry 1 - Anchor)
        - 'snowflake': Mundane topic swap (Entry 2 - Snowflake)
        - 'robustness': Same as original, marked for heavy perturbation (Entry 3)
    """
    # Entry 1: Anchor (original author text)
    result = [(para, "original") for para in paragraphs]

    if skip_variation:
        logger.info("Skipping topic variation (--skip-variation flag)")
        return result

    # Entry 3: Robustness (same text, will get heavy input perturbation)
    # Add these now - they use original text but will be processed differently
    robustness_entries = [(para, "robustness") for para in paragraphs]

    logger.info(f"Triad Strategy: 1 original + 1 snowflake + 1 robustness per paragraph")
    logger.info(f"Creating {len(paragraphs)} snowflake (topic swap) variations...")
    logger.info(f"Robustness entries: {len(paragraphs)} (will use heavy input perturbation)")

    # Entry 2: Snowflake (mundane topic swap)
    # Prepare all variation tasks
    tasks = []
    for idx, para in enumerate(paragraphs):
        topic = random.choice(MUNDANE_TOPICS)
        tasks.append((idx, para, topic))

    snowflake_count = 0
    failed_count = 0
    start_time = time.time()

    def process_variation(task):
        idx, para, topic = task
        varied = create_topic_variation(para, author, topic)
        return idx, varied

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_variation, task): task for task in tasks}

        for future in as_completed(futures):
            try:
                idx, varied = future.result()
                if varied:
                    result.append((varied, "snowflake"))
                    snowflake_count += 1
                else:
                    failed_count += 1

                total_processed = snowflake_count + failed_count
                if total_processed % 20 == 0:
                    elapsed = time.time() - start_time
                    rate = total_processed / elapsed if elapsed > 0 else 0
                    success_rate = snowflake_count / total_processed * 100 if total_processed > 0 else 0
                    logger.info(
                        f"Snowflake: {snowflake_count}/{len(paragraphs)} | "
                        f"Failed: {failed_count} | "
                        f"Success: {success_rate:.0f}% | "
                        f"Rate: {rate:.1f}/s"
                    )
            except Exception as e:
                failed_count += 1
                logger.debug(f"Variation task failed: {e}")

    # Add robustness entries
    result.extend(robustness_entries)

    elapsed = time.time() - start_time
    logger.info(
        f"Snowflake complete: {snowflake_count} created, {failed_count} failed "
        f"in {elapsed:.1f}s"
    )

    # Log Triad breakdown
    category_counts = {}
    for _, vtype in result:
        category_counts[vtype] = category_counts.get(vtype, 0) + 1
    logger.info(f"Triad breakdown: {category_counts}")

    # Calculate actual ratio
    total = len(result)
    original_pct = category_counts.get('original', 0) / total * 100
    logger.info(f"Real author signal: {original_pct:.1f}% (target: ~33%)")

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
        # Triad Strategy handling:
        # - 'original': Overlapping chunks (continuous narrative, style in transitions)
        # - 'snowflake': Keep separate (each is about different mundane topic)
        # - 'robustness': Keep separate (will get heavy input perturbation later)
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
# Uses DeepSeek API by default for fast bulk processing.
# Configuration in config.json under llm.provider.rtt and llm.providers.deepseek_rtt.
# Set llm.provider.rtt to "mlx" for local processing (slower but free).

# Global RTT neutralizer (shared across threads)
_rtt_neutralizer = None
_rtt_lock = None  # Lock for thread-safe access


def get_rtt_neutralizer(provider: str = None, batch_size: int = None):
    """Get or create shared RTT neutralizer (singleton pattern).

    Args:
        provider: 'mlx' or 'deepseek'. If None, reads from config.json.
        batch_size: Batch size for DeepSeek (ignored for MLX).
    """
    global _rtt_neutralizer, _rtt_lock
    if _rtt_neutralizer is None:
        import threading
        from src.llm.mlx_provider import create_rtt_neutralizer
        _rtt_neutralizer = create_rtt_neutralizer(provider=provider, batch_size=batch_size)
        _rtt_lock = threading.Lock()
        logger.info(f"RTT neutralizer ready: {type(_rtt_neutralizer).__name__}")
    return _rtt_neutralizer, _rtt_lock


def neutralize_text(styled_text: str, max_retries: int = 2, monotone: bool = True) -> Optional[str]:
    """Round-Trip Translation neutralization via Mandarin pivot.

    Step 1 (Scrub): English → Mandarin (HSK3 vocabulary)
    Step 2 (Rinse): Mandarin → Plain English
    Step 3 (Flatten): Break into short SVO sentences (if monotone=True)

    Uses provider from config.json (default: DeepSeek API for speed).
    Thread-safe via lock.

    Args:
        styled_text: The styled text to neutralize
        max_retries: Number of retry attempts
        monotone: If True, flatten to uniform short sentences (default True for training)

    Returns:
        Neutral English with all facts preserved, or None if failed
    """
    try:
        neutralizer, lock = get_rtt_neutralizer()
        with lock:
            return neutralizer.neutralize(styled_text, max_retries=max_retries, monotone=monotone)
    except Exception as e:
        logger.error(f"RTT neutralization failed: {e}")
        return None


def neutralize_batch(texts: list, monotone: bool = True, on_progress=None) -> list:
    """Neutralize multiple texts in batched API calls.

    DeepSeek: Uses parallel batch processing (no lock needed, thread-safe).
    MLX: Falls back to individual calls with locking (single-threaded).

    Args:
        texts: List of styled texts to neutralize.
        monotone: If True, flatten to uniform short sentences.
        on_progress: Optional callback (processed, total).

    Returns:
        List of neutralized texts (None for failures).
    """
    from src.llm.mlx_provider import DeepSeekRTTNeutralizer

    neutralizer, lock = get_rtt_neutralizer()

    # DeepSeek is thread-safe with internal parallelization - no lock needed
    if isinstance(neutralizer, DeepSeekRTTNeutralizer):
        return neutralizer.neutralize_batch(texts, monotone=monotone, on_progress=on_progress)

    # MLX requires locking (single-threaded)
    if hasattr(neutralizer, 'neutralize_batch'):
        with lock:
            return neutralizer.neutralize_batch(texts, monotone=monotone, on_progress=on_progress)
    else:
        # Fall back to individual processing with locking
        results = []
        for i, text in enumerate(texts):
            with lock:
                result = neutralizer.neutralize(text, monotone=monotone)
            results.append(result)
            if on_progress:
                on_progress(i + 1, len(texts))
        return results


# =============================================================================
# Step 5: Structural Analysis and Control Tokens
# =============================================================================
# Style tags tell the model WHAT structural pattern to produce.
# This turns "vibe" into concrete instructions.

def analyze_structure(text: str, nlp=None) -> dict:
    """Analyze structural features of text for style tagging.

    Returns dict with:
        - sentence_lengths: list of word counts per sentence
        - avg_length: average sentence length
        - length_variance: how varied the lengths are
        - has_complex_syntax: semicolons, em-dashes, nested clauses
        - connectives: conjunctions used (and, but, yet, for, etc.)
    """
    if nlp is None:
        nlp = get_nlp()

    sentences = split_into_sentences(text, nlp)
    if not sentences:
        return {"tag": "[STYLE: Simple]"}

    lengths = [len(s.split()) for s in sentences]
    avg_length = sum(lengths) / len(lengths)

    # Calculate variance
    if len(lengths) > 1:
        variance = sum((x - avg_length) ** 2 for x in lengths) / len(lengths)
        std_dev = variance ** 0.5
    else:
        std_dev = 0

    # Detect complex syntax markers
    complex_markers = [';', '—', '--', ':', '(', ')']
    has_complex = any(m in text for m in complex_markers)

    # Detect connectives
    connective_words = ['however', 'although', 'yet', 'moreover', 'furthermore',
                        'nevertheless', 'whilst', 'whereas']
    has_literary_connectives = any(w in text.lower() for w in connective_words)

    return {
        "avg_length": avg_length,
        "std_dev": std_dev,
        "has_complex": has_complex,
        "has_literary_connectives": has_literary_connectives,
        "sentence_count": len(sentences),
    }


def generate_style_tag(styled_text: str, nlp=None) -> str:
    """Generate a structural style tag based on text analysis.

    Tags describe the structural features the model should produce:
    - Length pattern: Short & Punchy, Varied Lengths, Long & Flowing
    - Complexity: Simple Syntax, Complex Syntax, Baroque Syntax

    Example: [STYLE: Varied Lengths | Complex Syntax]
    """
    analysis = analyze_structure(styled_text, nlp)

    # Determine length pattern
    avg = analysis.get("avg_length", 15)
    std = analysis.get("std_dev", 0)

    if avg < 12:
        length_tag = "Short & Punchy"
    elif avg > 25:
        length_tag = "Long & Flowing"
    elif std > 8:
        length_tag = "Varied Lengths"
    else:
        length_tag = "Medium Length"

    # Determine complexity
    if analysis.get("has_complex") and analysis.get("has_literary_connectives"):
        complexity_tag = "Baroque Syntax"
    elif analysis.get("has_complex"):
        complexity_tag = "Complex Syntax"
    else:
        complexity_tag = "Simple Syntax"

    return f"[STYLE: {length_tag} | {complexity_tag}]"


# =============================================================================
# Step 6: Prompt Jitter and Input Perturbation (NEFTune Simulation)
# =============================================================================
# NEFTune adds noise to embeddings. We simulate this by:
# 1. Randomizing the instruction prefix (Prompt Jitter)
# 2. Adding typos/synonyms to neutral input (Input Perturbation)
# This forces the model to learn robust patterns, not memorize strings.

# Prompt templates for jitter - randomly select one per example
PROMPT_TEMPLATES = [
    "Rewrite in {author}'s style (~{word_count} words):",
    "Transform this text into {author}'s voice (~{word_count} words):",
    "Convert to {author}'s style (~{word_count} words):",
    "Apply {author}'s style (~{word_count} words):",
    "Rewrite the following in {author}'s voice (~{word_count} words):",
    "Style transfer to {author} (~{word_count} words):",
]

# Simple synonym map for input perturbation
SYNONYMS = {
    "big": ["large", "huge", "great"],
    "small": ["little", "tiny", "minor"],
    "old": ["ancient", "aged", "elderly"],
    "new": ["fresh", "recent", "modern"],
    "good": ["fine", "nice", "great"],
    "bad": ["poor", "awful", "terrible"],
    "house": ["building", "home", "dwelling"],
    "said": ["stated", "spoke", "remarked"],
    "walked": ["went", "moved", "traveled"],
    "looked": ["appeared", "seemed", "gazed"],
    "very": ["quite", "rather", "extremely"],
    "really": ["truly", "actually", "indeed"],
}


def perturb_text(text: str, perturbation_rate: float = 0.08) -> str:
    """Apply random perturbations to text (Poor Man's NEFTune).

    Applies 5-10% random changes:
    - Synonym swap: Replace word with synonym
    - Word drop: Remove non-essential words (the, a, an)
    - Typo: Swap adjacent characters

    Args:
        text: Input text to perturb
        perturbation_rate: Probability of perturbing each word (default 8%)

    Returns:
        Perturbed text
    """
    words = text.split()
    result = []
    droppable = {'the', 'a', 'an', 'very', 'really', 'just', 'quite'}

    for word in words:
        if random.random() > perturbation_rate:
            result.append(word)
            continue

        # Choose perturbation type
        choice = random.random()

        if choice < 0.4:
            # Synonym swap (40% of perturbations)
            word_lower = word.lower().rstrip('.,!?;:')
            if word_lower in SYNONYMS:
                synonym = random.choice(SYNONYMS[word_lower])
                # Preserve case
                if word[0].isupper():
                    synonym = synonym.capitalize()
                result.append(synonym + word[len(word_lower):])
            else:
                result.append(word)

        elif choice < 0.7:
            # Word drop (30% of perturbations)
            if word.lower() in droppable:
                pass  # Drop the word
            else:
                result.append(word)

        else:
            # Typo - swap two adjacent chars (30% of perturbations)
            if len(word) > 3:
                i = random.randint(1, len(word) - 2)
                word = word[:i] + word[i+1] + word[i] + word[i+2:]
            result.append(word)

    return ' '.join(result)


def format_training_example(
    neutral_text: str,
    styled_text: str,
    author: str,
    word_count: int,
    style_tag: str = None,
    variation_type: str = "original",
    use_jitter: bool = True,
) -> dict:
    """Format a training example for BASE model (not instruct).

    Base models need raw text completion format, not chat roles.
    The model learns: given neutral text + instruction → produce styled text.

    Applies NEFTune simulation based on variation_type:
    - Prompt Jitter: Random instruction prefix (all types)
    - Light Perturbation: 8% rate for 'original' and 'snowflake'
    - Heavy Perturbation: 15% rate for 'robustness' (Entry 3 of Triad)

    Format:
        Rewrite in {author}'s style (~N words):
        [STYLE: Varied Lengths | Complex Syntax]
        [neutral text with perturbations]
        ###
        [styled text]
    """
    # Select random prompt template (Prompt Jitter)
    if use_jitter:
        template = random.choice(PROMPT_TEMPLATES)
    else:
        template = PROMPT_TEMPLATES[0]

    instruction = template.format(author=author, word_count=word_count)

    # Apply perturbation based on variation type
    # Robustness entries get HEAVY perturbation (Entry 3 of Triad)
    # Other entries get light perturbation
    if variation_type == "robustness":
        perturbed_input = create_heavy_perturbation(neutral_text)
    else:
        perturbed_input = perturb_text(neutral_text)

    # Build prompt with optional style tag
    if style_tag:
        prompt = f"{instruction}\n{style_tag}\n{perturbed_input}\n###\n"
    else:
        prompt = f"{instruction}\n{perturbed_input}\n###\n"

    # For mlx_lm.lora with base models: {"text": "prompt + completion"}
    # With mask_prompt=true, only the completion (after prompt) is trained
    return {
        "text": prompt + styled_text,
        "prompt": prompt,  # For mask_prompt to know where to split
        "word_count": word_count,
        "style_tag": style_tag,
    }


def generate_training_data(
    chunks: List[Tuple[str, str]],
    author: str,
    output_path: Path,
    workers: int = 1,
    monotone: bool = False,
    resume: bool = False,
) -> int:
    """Generate training data using RTT neutralization, writing progressively.

    Processes chunks sequentially (MLX requires single-threaded access) and
    writes each result immediately. Progress is saved on interrupt.

    Args:
        chunks: List of (styled_text, variation_type) tuples
        author: Author name for system prompt
        output_path: Output JSONL file path
        workers: Unused (kept for API compatibility)
        monotone: If True, apply extra flattening step (slower, +50% time)
        resume: If True, skip already-processed items and append to file

    Returns:
        Number of examples written
    """
    total = len(chunks)
    mode = "monotone" if monotone else "standard"

    # Check for existing progress if resuming
    processed_indices = set()
    if resume and output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if 'source_idx' in entry:
                        processed_indices.add(entry['source_idx'])
                except:
                    pass
        logger.info(f"Resuming: found {len(processed_indices)} already processed items")

    remaining = total - len(processed_indices)
    logger.info(f"Generating training data for {remaining}/{total} chunks ({mode} RTT)...")
    logger.info(f"Writing progressively to {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    success_count = len(processed_indices)
    failed_count = 0
    type_counts = {}
    start_time = time.time()

    # Get batch size from neutralizer
    neutralizer, _ = get_rtt_neutralizer()
    batch_size = getattr(neutralizer, 'batch_size', 1)
    use_batching = batch_size > 1 and hasattr(neutralizer, 'neutralize_batch')

    if use_batching:
        logger.info(f"Using batched RTT with batch_size={batch_size}")
    else:
        logger.info("Using sequential RTT (batch_size=1)")

    # Filter out already processed chunks
    pending_chunks = [(idx, styled_text, vtype) for idx, (styled_text, vtype) in enumerate(chunks)
                      if idx not in processed_indices]

    # Append if resuming, otherwise overwrite
    file_mode = 'a' if resume and processed_indices else 'w'
    with open(output_path, file_mode, encoding='utf-8') as f:
        if use_batching:
            # Process in batches
            for batch_start in range(0, len(pending_chunks), batch_size):
                batch_end = min(batch_start + batch_size, len(pending_chunks))
                batch = pending_chunks[batch_start:batch_end]

                # Extract texts for batch neutralization
                batch_texts = [styled_text for _, styled_text, _ in batch]

                # Progress update
                elapsed = time.time() - start_time
                processed = len(processed_indices) + batch_start
                if processed > 0:
                    rate = processed / elapsed
                    eta = (total - processed) / rate if rate > 0 else 0
                    logger.info(
                        f"[{processed}/{total}] ({processed*100//total}%) | "
                        f"✓{success_count} ✗{failed_count} | "
                        f"{rate:.2f}/s | ETA: {eta/60:.1f}m | batch {batch_start//batch_size + 1}"
                    )

                # Batch neutralization
                try:
                    neutrals = neutralize_batch(batch_texts, monotone=monotone)
                except Exception as e:
                    logger.warning(f"Batch RTT error: {e}")
                    neutrals = [None] * len(batch)

                # Process results
                for (idx, styled_text, vtype), neutral in zip(batch, neutrals):
                    if neutral:
                        word_count = len(styled_text.split())
                        style_tag = generate_style_tag(styled_text)

                        example = format_training_example(
                            neutral_text=neutral,
                            styled_text=styled_text,
                            author=author,
                            word_count=word_count,
                            style_tag=style_tag,
                            variation_type=vtype,  # Controls perturbation level
                        )
                        example["variation_type"] = vtype
                        example["source_idx"] = idx

                        f.write(json.dumps(example) + '\n')
                        success_count += 1
                        type_counts[vtype] = type_counts.get(vtype, 0) + 1
                    else:
                        failed_count += 1
                        logger.warning(f"  [{idx}] ✗ Failed to neutralize ({len(styled_text.split())}w)")

                f.flush()  # Flush after each batch
        else:
            # Sequential processing (original behavior)
            for batch_idx, (idx, styled_text, vtype) in enumerate(pending_chunks):
                # Progress header every 10 items
                if batch_idx % 10 == 0:
                    elapsed = time.time() - start_time
                    processed = len(processed_indices) + batch_idx
                    if processed > 0:
                        rate = processed / elapsed
                        eta = (total - processed) / rate if rate > 0 else 0
                        logger.info(
                            f"[{processed}/{total}] ({processed*100//total}%) | "
                            f"✓{success_count} ✗{failed_count} | "
                            f"{rate:.2f}/s | ETA: {eta/60:.1f}m"
                        )
                    else:
                        logger.info(f"[{processed}/{total}] Starting...")

                # RTT neutralization
                try:
                    neutral = neutralize_text(styled_text, monotone=monotone)
                except Exception as e:
                    logger.warning(f"  [{idx}] RTT error: {e}")
                    neutral = None

                if neutral:
                    word_count = len(styled_text.split())
                    style_tag = generate_style_tag(styled_text)

                    example = format_training_example(
                        neutral_text=neutral,
                        styled_text=styled_text,
                        author=author,
                        word_count=word_count,
                        style_tag=style_tag,
                        variation_type=vtype,  # Controls perturbation level
                    )
                    example["variation_type"] = vtype
                    example["source_idx"] = idx

                    f.write(json.dumps(example) + '\n')
                    f.flush()
                    success_count += 1
                    type_counts[vtype] = type_counts.get(vtype, 0) + 1
                else:
                    failed_count += 1
                    logger.warning(f"  [{idx}] ✗ Failed to neutralize ({len(styled_text.split())}w)")

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
    parser.add_argument("--workers", type=int, default=1,
                        help="Workers for parallel processing (MLX neutralization is serialized, so 1 is optimal)")
    parser.add_argument("--skip-variation", action="store_true",
                        help="Skip topic variation step (originals only, no Triad)")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=None,
                        help="Run specific phase: 1=Anchors (RTT), 2=Noise (script), 3=Snowflakes (LLM)")
    parser.add_argument("--no-monotone", action="store_true",
                        help="Disable monotone flattening (faster but less effective for burstiness)")
    parser.add_argument("--max-paragraphs", type=int, default=None, help="Max paragraphs to process")
    parser.add_argument("--min-para-words", type=int, default=100, help="Min words per paragraph (curation)")
    parser.add_argument("--max-para-words", type=int, default=650, help="Max words per paragraph (curation)")
    parser.add_argument("--min-chunk-words", type=int, default=150, help="Min words per chunk (default: 150)")
    parser.add_argument("--max-chunk-words", type=int, default=400, help="Max words per chunk (default: 400)")
    parser.add_argument("--overlap-sentences", type=int, default=2, help="Sentence overlap between chunks")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from intermediate JSON (skips Steps 1-2)")
    parser.add_argument("--resume-from-chunks", type=str, default=None, help="Resume from chunks JSON (skips Steps 1-3)")
    parser.add_argument("--resume", action="store_true", help="Resume from last processed item (checks all.jsonl)")
    parser.add_argument("--save-intermediate", type=str, default=None, help="Save intermediate paragraphs")
    parser.add_argument("--from-selected", type=str, default=None,
                        help="Load from selected paragraphs JSON (from select_diverse_paragraphs.py)")

    args = parser.parse_args()

    if not args.resume_from and not args.resume_from_chunks and not args.corpus and not args.from_selected:
        parser.error("--corpus or --from-selected is required unless using --resume-from or --resume-from-chunks")

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

    elif args.from_selected:
        # Load from pre-selected paragraphs JSON (from select_diverse_paragraphs.py)
        logger.info("=" * 60)
        logger.info("LOADING FROM SELECTED PARAGRAPHS (skipping Step 1)")
        logger.info("=" * 60)

        with open(args.from_selected, 'r', encoding='utf-8') as f:
            selected_data = json.load(f)

        paragraphs = [p["text"] for p in selected_data["paragraphs"]]
        metadata = selected_data.get("metadata", {})
        logger.info(f"Loaded {len(paragraphs)} pre-selected paragraphs")
        logger.info(f"  Source: {metadata.get('source', 'unknown')}")
        logger.info(f"  Tokens: {metadata.get('actual_tokens', 'unknown'):,}")
        logger.info(f"  Mean quality: {metadata.get('mean_quality', 'unknown'):.3f}")

        if args.max_paragraphs:
            paragraphs = paragraphs[:args.max_paragraphs]

        # Step 2: Triad Strategy (1 original + 1 snowflake + 1 robustness)
        logger.info("=" * 60)
        logger.info("STEP 2: Triad Strategy (1:3 expansion)")
        logger.info("=" * 60)

        expanded = expand_corpus_with_variations(
            paragraphs,
            author=args.author,
            workers=args.workers,
            skip_variation=args.skip_variation,
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

        # Step 2: Triad Strategy (1 original + 1 snowflake + 1 robustness)
        logger.info("=" * 60)
        logger.info("STEP 2: Triad Strategy (1:3 expansion)")
        logger.info("=" * 60)

        expanded = expand_corpus_with_variations(
            paragraphs,
            author=args.author,
            workers=args.workers,
            skip_variation=args.skip_variation,
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
    num_examples = generate_training_data(
        chunks, args.author, all_output_path,
        workers=args.workers, monotone=not args.no_monotone,
        resume=args.resume
    )

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
                # Keep text field for base model format, strip other metadata
                f.write(json.dumps({"text": ex["text"]}) + '\n')
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
