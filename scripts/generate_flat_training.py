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
from difflib import SequenceMatcher
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
# Includes ACTIONS, DESCRIPTIONS, and OPINIONS to cover all structural modes
MUNDANE_TOPICS = [
    # ACTIONS - Domestic activities
    "making toast for breakfast", "doing the weekly laundry", "organizing a closet",
    "washing dishes after dinner", "vacuuming the living room", "folding clean towels",
    "watering houseplants", "making the bed", "cleaning the bathroom mirror",
    "sorting through old mail", "replacing a lightbulb", "taking out the trash",
    # ACTIONS - Office/bureaucratic
    "filing tax returns", "attending a staff meeting", "writing a work email",
    "waiting in line at the DMV", "filling out insurance forms", "updating a spreadsheet",
    "scheduling a dentist appointment", "renewing a driver's license", "balancing a checkbook",
    # ACTIONS - Routine errands
    "grocery shopping on Saturday", "pumping gas at the station", "returning library books",
    "picking up dry cleaning", "waiting for a bus", "walking to the mailbox",
    "parallel parking downtown", "choosing produce at the market", "standing in the checkout line",
    # ACTIONS - Simple activities
    "brewing morning coffee", "tying shoelaces", "checking the weather forecast",
    "setting an alarm clock", "microwaving leftovers", "charging a phone overnight",
    "brushing teeth before bed", "packing a lunch", "feeding the cat",
    # DESCRIPTIONS - Atmospheres and places (crucial for atmospheric styles)
    "the smell of a clean hospital waiting room", "the appearance of a disorganized office desk",
    "the texture of stale bread", "the quiet atmosphere of an empty parking lot at noon",
    "the feeling of a warm laundromat on a cold day", "the sound of a refrigerator humming",
    "the view from a suburban kitchen window", "the clutter of an old garage",
    "the sterile brightness of a fluorescent-lit hallway", "the mustiness of an attic",
    "the mundane geometry of a parking garage", "the particular silence of a library",
    "the way dust collects on a computer monitor", "the peeling paint on an old fence",
    # DESCRIPTIONS - Objects and textures
    "the worn leather of an old office chair", "a slightly rusted garden hose",
    "the scratched surface of a kitchen table", "faded wallpaper in a hallway",
    "the particular weight of a full laundry basket", "cracked tiles in a bathroom",
    # OPINIONS/ARGUMENTS - For essayist styles
    "why coffee is better than tea", "a complaint about slow internet speeds",
    "a review of a mediocre sandwich", "the case against reply-all emails",
    "why waiting rooms should have better magazines", "the problem with self-checkout machines",
    "an argument for alphabetizing spice racks", "why voicemail is obsolete",
    "a defense of generic brand cereal", "the annoyance of printer paper jams",
    "why takeout containers never stack properly", "the unfairness of parking meters",
]

# =============================================================================
# Perspective Pools for Variation Generation
# =============================================================================
# The LoRA is trained on first-person singular narrative ("I saw", "I found").
# To make inference work with ANY input perspective, we generate training variations
# in different perspectives. This teaches the model that style is independent of POV.

PERSPECTIVE_TRANSFORMS = {
    "first_person_plural": {
        "description": "Convert to first person plural (we/us/our)",
        "instruction": """Convert this first-person singular narrative to first-person PLURAL.
Change all instances of:
- "I" → "we"
- "me" → "us"
- "my" → "our"
- "myself" → "ourselves"
- "mine" → "ours"

The narrator is now speaking for a group who experienced this together.
Preserve ALL other aspects: sentence structure, vocabulary, tone, rhythm.""",
    },
    "third_person": {
        "description": "Convert to third person (he/she/they)",
        "instruction": """Convert this first-person narrative to THIRD PERSON.
The narrator becomes "the observer", "the researcher", "the chronicler", or similar.
Change all instances of:
- "I" → "the observer" / "he" / "she" / "they"
- "me" → "him" / "her" / "them"
- "my" → "his" / "her" / "their"
- "myself" → "himself" / "herself" / "themselves"

Use a consistent third-person subject throughout.
Preserve ALL other aspects: sentence structure, vocabulary, tone, rhythm.""",
    },
    "impersonal": {
        "description": "Convert to impersonal exposition (one/it is/passive voice)",
        "instruction": """Convert this first-person narrative to IMPERSONAL EXPOSITION.
Remove the personal narrator entirely. Use:
- "One observes that..." / "It is observed that..."
- "The evidence suggests..." / "It becomes clear that..."
- Passive constructions: "was discovered", "can be seen", "is known"
- "We" in the academic sense (the reader and author together)

The text should read like academic or encyclopedic prose.
Preserve ALL factual content and the logical flow of ideas.""",
    },
}


# Snowflake thresholds. The variant must keep the author's sentences and
# wording and change only the topic words.
SNOWFLAKE_MIN_PUNCT_SIMILARITY = 0.8
SNOWFLAKE_MIN_SKELETON_SIMILARITY = 0.75
SNOWFLAKE_MAX_SENTENCE_LENGTH_DRIFT = 0.25
SNOWFLAKE_MAX_CONTENT_OVERLAP = 0.8

_PUNCT_CHARS = set(',;:—–-()?!"')


def _punctuation_signature(text: str) -> List[str]:
    return [c for c in text if c in _PUNCT_CHARS or c == '.']


def _function_skeleton(text: str) -> List[str]:
    """Tokens with content words blanked out: "the _ of the _ ;"."""
    from spacy.lang.en.stop_words import STOP_WORDS
    tokens = re.findall(r"[\w']+|[^\w\s]", text.lower())
    return [t if (t in STOP_WORDS or not t[0].isalnum()) else '_' for t in tokens]


def _content_words(text: str) -> set:
    from spacy.lang.en.stop_words import STOP_WORDS
    return {w for w in re.findall(r"[a-z']+", text.lower()) if w not in STOP_WORDS and len(w) > 2}


def _similarity(a: list, b: list) -> float:
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def validate_variation(original: str, varied: str, nlp=None) -> Tuple[bool, str]:
    """Check a snowflake variant kept the original's structure and wording.

    Sentence count must match, each sentence stays about as long, and the
    punctuation and function-word skeleton match closely. The content words
    must still differ enough to show the topic actually changed.
    """
    orig_words = len(original.split())
    varied_words = len(varied.split())
    if abs(orig_words - varied_words) > orig_words * 0.20:
        return False, f"Word count too different: {orig_words} vs {varied_words}"

    if nlp is None:
        nlp = get_nlp()
    orig_sents = split_into_sentences(original, nlp)
    varied_sents = split_into_sentences(varied, nlp)
    if len(orig_sents) != len(varied_sents):
        return False, f"Sentence count mismatch: {len(orig_sents)} vs {len(varied_sents)}"

    for o, v in zip(orig_sents, varied_sents):
        o_len, v_len = len(o.split()), len(v.split())
        if abs(o_len - v_len) > max(3, o_len * SNOWFLAKE_MAX_SENTENCE_LENGTH_DRIFT):
            return False, f"Sentence length changed: {o_len} vs {v_len}"

    punct = _similarity(_punctuation_signature(original), _punctuation_signature(varied))
    if punct < SNOWFLAKE_MIN_PUNCT_SIMILARITY:
        return False, f"Punctuation pattern changed ({punct:.0%} similar)"

    skeleton = _similarity(_function_skeleton(original), _function_skeleton(varied))
    if skeleton < SNOWFLAKE_MIN_SKELETON_SIMILARITY:
        return False, f"Sentence structure changed ({skeleton:.0%} similar)"

    orig_content = _content_words(original)
    if orig_content:
        overlap = len(orig_content & _content_words(varied)) / len(orig_content)
        if overlap > SNOWFLAKE_MAX_CONTENT_OVERLAP:
            return False, f"Topic not changed ({overlap:.0%} content overlap)"

    return True, "OK"


def create_topic_variation(
    paragraph: str,
    author: str,
    topic: str,
    max_attempts: int = 2
) -> Optional[str]:
    """Create a topic variation (Snowflake) that keeps the author's own wording.

    Snowflake rows teach the model to write about subjects the corpus never
    covers. The target is still mostly the author's text: every sentence,
    clause, connective and non-topical word stays as written, and only the
    words tied to the subject are swapped for ones about ``topic``. A free
    rewrite would train the model on the LLM's imitation of the author.
    """
    system = f"""You are editing a passage by {author}. You change what it is about, never how it is written.

Replace only the words tied to the topic (the subject nouns, the names, the examples, and the verbs and adjectives that only make sense for that subject) with words about the new topic.

Keep every other word exactly as written: connectives, qualifiers, evaluative words, idioms, pronouns, sentence openings and endings. Keep every sentence, in the same order, with the same clauses and the same punctuation. Do not add, merge, split or drop sentences. Do not add new ideas, jokes or flourishes."""

    prompt = f"""Change the topic of this passage by {author} to "{topic}".

Original passage:
{paragraph}

Rules:
1. Swap only the words tied to the topic for words about "{topic}".
2. Keep every other word as it is, in place.
3. Same {len(split_into_sentences(paragraph))} sentences in the same order, each about the same length, with the same punctuation.
4. About {len(paragraph.split())} words.

Output only the edited passage, nothing else."""

    for attempt in range(max_attempts):
        try:
            varied = call_deepseek(prompt, system, max_retries=2)
            varied = varied.strip('`"\' \n')
            if varied.startswith('```'):
                varied = re.sub(r'^```\w*\n?', '', varied)
                varied = re.sub(r'\n?```$', '', varied)

            is_valid, reason = validate_variation(paragraph, varied)
            if is_valid:
                return varied
            else:
                logger.debug(f"Variation rejected: {reason}")
        except Exception as e:
            logger.debug(f"Variation attempt {attempt+1} failed: {e}")

    return None


# A perspective variant that keeps more than this share of the original's
# first-person-singular pronouns is treated as a copy.
MAX_KEPT_PRONOUN_FRACTION = 0.25

_FIRST_PERSON_SINGULAR = re.compile(r"\b(I|me|my|mine|myself)\b")


def validate_perspective_variation(original: str, varied: str, nlp=None) -> Tuple[bool, str]:
    """Validate that perspective variation preserved content and structure."""
    orig_words = len(original.split())
    varied_words = len(varied.split())

    # Allow 15% word count variance for perspective changes (pronouns change count)
    if abs(orig_words - varied_words) > orig_words * 0.15:
        return False, f"Word count too different: {orig_words} vs {varied_words}"

    if nlp is None:
        nlp = get_nlp()
    orig_sentences = len(split_into_sentences(original, nlp))
    varied_sentences = len(split_into_sentences(varied, nlp))

    # Sentence count should be exactly the same (perspective change doesn't add sentences)
    if orig_sentences != varied_sentences:
        return False, f"Sentence count mismatch: {orig_sentences} vs {varied_sentences}"

    # A copy of the original teaches nothing and duplicates the target, so
    # most of the first-person-singular pronouns must actually be gone.
    before = len(_FIRST_PERSON_SINGULAR.findall(original))
    after = len(_FIRST_PERSON_SINGULAR.findall(varied))
    if varied.strip() == original.strip() or after > before * MAX_KEPT_PRONOUN_FRACTION:
        return False, f"Perspective unchanged ({after}/{before} first-person pronouns kept)"

    return True, "OK"


def create_perspective_variation(
    paragraph: str,
    author: str,
    perspective_key: str,
    max_attempts: int = 2
) -> Optional[str]:
    """Create a perspective variation that changes POV while maintaining style.

    This teaches the LoRA that style is independent of perspective. By training on
    the same content in different perspectives (first-person plural, third-person,
    impersonal), the model learns to apply style regardless of input POV.

    Args:
        paragraph: Original author paragraph (assumed first-person singular).
        author: Author name for style context.
        perspective_key: Key from PERSPECTIVE_TRANSFORMS dict.
        max_attempts: Maximum retry attempts.

    Returns:
        Paragraph in new perspective, or None if failed.
    """
    if perspective_key not in PERSPECTIVE_TRANSFORMS:
        logger.warning(f"Unknown perspective key: {perspective_key}")
        return None

    # The transforms rewrite first-person singular text. Without it there is
    # nothing to change and the model just returns the original.
    if not _FIRST_PERSON_SINGULAR.search(paragraph):
        return None

    transform = PERSPECTIVE_TRANSFORMS[perspective_key]

    system = f"""You are a literary perspective transformation assistant.

Your task: Change the grammatical perspective of a passage while preserving:
- The EXACT sentence structure and rhythm
- ALL vocabulary (except pronouns and verb conjugations)
- The author's characteristic style and tone
- ALL factual content

You are NOT rewriting or paraphrasing. You are ONLY changing the grammatical person."""

    prompt = f"""Transform this passage from first-person singular to {transform['description']}.

{transform['instruction']}

Original passage:
{paragraph}

Requirements:
1. Change ONLY the perspective - preserve everything else
2. Keep the same sentence count and structure
3. Match the word count closely (~{len(paragraph.split())} words)
4. Maintain the author's distinctive vocabulary and rhythm

Output only the transformed passage, nothing else."""

    for attempt in range(max_attempts):
        try:
            varied = call_deepseek(prompt, system, max_retries=2)
            varied = varied.strip('`"\' \n')
            if varied.startswith('```'):
                varied = re.sub(r'^```\w*\n?', '', varied)
                varied = re.sub(r'\n?```$', '', varied)

            # Validate the variation
            is_valid, reason = validate_perspective_variation(paragraph, varied)
            if is_valid:
                return varied
            else:
                logger.debug(f"Perspective variation rejected: {reason}")
        except Exception as e:
            logger.debug(f"Perspective variation attempt {attempt+1} failed: {e}")

    return None


# Items flowing through the pipeline are (text, variation_type, source_paragraphs):
# source_paragraphs holds the indices of the corpus paragraphs the text came
# from, so the train/val split can hold out whole paragraphs.
Item = Tuple[str, str, Tuple[int, ...]]


def save_intermediate(items: List[Item], path: Path, stage: str = "items") -> None:
    """Save intermediate data to JSON file.

    variation_type is one of:
    - 'original': Original author text
    - 'snowflake': Topic swap of an original
    - 'robustness': Original, marked for heavy input perturbation
    - 'perspective_*': Original in another grammatical person
    """
    data = [
        {"text": text, "variation_type": vtype, "source_paragraphs": list(src)}
        for text, vtype, src in items
    ]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(items)} {stage} to {path}")


def load_intermediate(path: Path, stage: str = "items") -> List[Item]:
    """Load intermediate data from JSON file.

    Files written before source ids existed load with an empty id tuple.
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    items = []
    for item in data:
        if "variation_type" in item:
            vtype = item["variation_type"]
        else:
            # Old format compatibility
            vtype = "varied" if item.get("is_varied", False) else "original"
        items.append((item["text"], vtype, tuple(item.get("source_paragraphs", ()))))
    logger.info(f"Loaded {len(items)} {stage} from {path}")
    return items


def load_snowflake_topics(topics_file: Optional[str] = None) -> List[str]:
    """Load snowflake topics from an external file or fall back to MUNDANE_TOPICS.

    The external file should be a Python file with a list variable (BOOK_TOPICS
    or MUNDANE_TOPICS). This allows per-author topic customization — e.g. conceptual
    topics for Russell vs mundane activities for Lovecraft.

    Args:
        topics_file: Path to a Python file containing a topic list variable.

    Returns:
        List of topic strings.
    """
    if topics_file:
        topics_path = Path(topics_file)
        if topics_path.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("topics", topics_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            # Look for BOOK_TOPICS first, then MUNDANE_TOPICS, then any list
            for attr in ["BOOK_TOPICS", "MUNDANE_TOPICS", "TOPICS"]:
                if hasattr(mod, attr):
                    topics = getattr(mod, attr)
                    logger.info(f"Loaded {len(topics)} snowflake topics from {topics_file} ({attr})")
                    return topics
            logger.warning(f"No topic list found in {topics_file}, using defaults")
        else:
            logger.warning(f"Topics file not found: {topics_file}, using defaults")

    logger.info(f"Using default MUNDANE_TOPICS ({len(MUNDANE_TOPICS)} topics)")
    return MUNDANE_TOPICS


def expand_corpus_with_variations(
    paragraphs: List[str],
    author: str,
    workers: int = 4,
    skip_variation: bool = False,
    skip_perspective: bool = False,
    snowflake_topics: Optional[List[str]] = None,
) -> List[Item]:
    """Expand corpus using enhanced Triad strategy with perspective variations.

    For each original paragraph, creates variations:
    - Entry 1 (Anchor): Original author text (first-person singular)
    - Entry 2 (Snowflake): Topic swap to mundane activity
    - Entry 3 (Robustness): Marked for heavy input perturbation later
    - Entry 4-6 (Perspective): Same content in different perspectives

    Perspective variations teach the LoRA that style is independent of POV:
    - first_person_plural: "we saw" instead of "I saw"
    - third_person: "the observer saw" instead of "I saw"
    - impersonal: "it was observed" instead of "I saw"

    Args:
        paragraphs: Original author paragraphs
        author: Author name for style preservation
        workers: Number of parallel workers
        skip_variation: If True, skip topic variation (originals only)
        skip_perspective: If True, skip perspective variations

    Returns:
        List of (text, variation_type, (paragraph_index,)) tuples where variation_type is:
        - 'original': Real author text (Entry 1 - Anchor)
        - 'snowflake': Mundane topic swap (Entry 2 - Snowflake)
        - 'robustness': Same as original, marked for heavy perturbation (Entry 3)
        - 'perspective_plural': First person plural version
        - 'perspective_third': Third person version
        - 'perspective_impersonal': Impersonal/passive version
    """
    # Entry 1: Anchor (original author text)
    result = [(para, "original", (idx,)) for idx, para in enumerate(paragraphs)]

    if skip_variation:
        logger.info("Skipping all variations (--skip-variation flag)")
        return result

    # Entry 3: Robustness (same text, will get heavy input perturbation)
    # Add these now - they use original text but will be processed differently
    robustness_entries = [(para, "robustness", (idx,)) for idx, para in enumerate(paragraphs)]

    logger.info(f"Enhanced Triad Strategy: 1 original + 1 snowflake + 1 robustness + 3 perspective per paragraph")
    logger.info(f"Creating {len(paragraphs)} snowflake (topic swap) variations...")
    logger.info(f"Robustness entries: {len(paragraphs)} (will use heavy input perturbation)")

    # Entry 2: Snowflake (topic swap)
    # Prepare all variation tasks
    topics_pool = snowflake_topics or MUNDANE_TOPICS
    tasks = []
    for idx, para in enumerate(paragraphs):
        topic = random.choice(topics_pool)
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
                    result.append((varied, "snowflake", (idx,)))
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

    # =========================================================================
    # Entry 4-6: Perspective Variations
    # =========================================================================
    # These teach the LoRA that style is independent of POV. By training on
    # the same content in different perspectives, the model learns to apply
    # style regardless of input perspective (first person, third person, etc.)

    if not skip_perspective:
        logger.info("=" * 60)
        logger.info("Creating perspective variations (first_person_plural, third_person, impersonal)...")

        perspective_counts = {key: 0 for key in PERSPECTIVE_TRANSFORMS.keys()}
        perspective_failed = 0
        perspective_start = time.time()

        # Prepare all perspective tasks (3 perspectives per paragraph)
        perspective_tasks = []
        for idx, para in enumerate(paragraphs):
            for perspective_key in PERSPECTIVE_TRANSFORMS.keys():
                perspective_tasks.append((idx, para, perspective_key))

        logger.info(f"Generating {len(perspective_tasks)} perspective variations...")

        def process_perspective(task):
            idx, para, perspective_key = task
            varied = create_perspective_variation(para, author, perspective_key)
            return idx, perspective_key, varied

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_perspective, task): task for task in perspective_tasks}

            for future in as_completed(futures):
                try:
                    idx, perspective_key, varied = future.result()
                    if varied:
                        result.append((varied, f"perspective_{perspective_key}", (idx,)))
                        perspective_counts[perspective_key] += 1
                    else:
                        perspective_failed += 1

                    total_processed = sum(perspective_counts.values()) + perspective_failed
                    if total_processed % 30 == 0:
                        elapsed_p = time.time() - perspective_start
                        rate = total_processed / elapsed_p if elapsed_p > 0 else 0
                        success_rate = sum(perspective_counts.values()) / total_processed * 100 if total_processed > 0 else 0
                        logger.info(
                            f"Perspective: {sum(perspective_counts.values())}/{len(perspective_tasks)} | "
                            f"Failed: {perspective_failed} | "
                            f"Success: {success_rate:.0f}% | "
                            f"Rate: {rate:.1f}/s"
                        )
                except Exception as e:
                    perspective_failed += 1
                    logger.debug(f"Perspective task failed: {e}")

        perspective_elapsed = time.time() - perspective_start
        logger.info(
            f"Perspective complete: {sum(perspective_counts.values())} created, {perspective_failed} failed "
            f"in {perspective_elapsed:.1f}s"
        )
        logger.info(f"  Per perspective: {perspective_counts}")
    else:
        logger.info("Skipping perspective variations (--skip-perspective flag)")

    # Log final breakdown
    category_counts = {}
    for _, vtype, _ in result:
        category_counts[vtype] = category_counts.get(vtype, 0) + 1
    logger.info(f"Final breakdown: {category_counts}")

    # Calculate actual ratio
    total = len(result)
    original_pct = category_counts.get('original', 0) / total * 100
    logger.info(f"Real author signal: {original_pct:.1f}%")

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

def create_overlapping_chunks(paragraphs: List[Item], config: OverlapConfig) -> List[Item]:
    """Create overlapping chunks that cross paragraph boundaries.

    IMPORTANT: Overlapping only applies to ORIGINALS (continuous narrative).
    Variations are kept as separate paragraphs because each was rewritten
    to a DIFFERENT topic - combining them creates "Frankenstein" narratives.

    Args:
        paragraphs: List of (text, variation_type, source_paragraphs) tuples
        config: Overlap configuration (min/max words, overlap sentences)

    Returns:
        List of (chunk_text, variation_type, source_paragraphs) tuples. An
        original chunk lists every paragraph it spans.
    """
    start_time = time.time()
    nlp = get_nlp()

    # Group paragraphs by variation_type
    by_type = {}
    for text, vtype, src in paragraphs:
        by_type.setdefault(vtype, []).append((text, tuple(src)))

    logger.info(f"Processing {len(paragraphs)} paragraphs into chunks...")
    logger.info(f"Variation types: {list(by_type.keys())}")
    logger.info(f"Target chunk size: {config.min_words}-{config.max_words} words")
    if len(by_type) > 1 or "original" not in by_type:
        logger.info(f"NOTE: Overlapping only for 'original' type (continuous narrative)")

    all_chunks = []

    for vtype, texts in by_type.items():
        # Enhanced Triad Strategy handling:
        # - 'original': Overlapping chunks (continuous narrative, style in transitions)
        # - 'snowflake': Keep separate (each is about different mundane topic)
        # - 'robustness': Keep separate (will get heavy input perturbation later)
        # - 'perspective_*': Keep separate (each is same content in different POV)
        if vtype != "original":
            chunks_for_type = []
            for para_text, src in texts:
                # Include slightly smaller ones too
                if len(para_text.split()) >= config.min_words * 0.7:
                    chunks_for_type.append((para_text, vtype, src))

            all_chunks.extend(chunks_for_type)
            logger.info(f"  {vtype}: {len(texts)} paragraphs -> {len(chunks_for_type)} chunks (no overlap)")
            continue

        # For ORIGINALS: Use sliding window with overlap (continuous narrative)
        sentences = []
        for para_idx, (para_text, src) in enumerate(texts):
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
                    'src': src,
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
                chunk_src = tuple(sorted({p for s in chunk_sentences for p in s['src']}))
                chunks_for_type.append((chunk_text, vtype, chunk_src))

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
#    - HSK 5 vocabulary constraint strips literary words
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


def clean_neutral_text(text: str) -> str:
    """Clean RTT output artifacts.

    Fixes:
    - Leading punctuation (". " prefix from RTT chunk boundaries)
    - Trailing whitespace
    - Double spaces
    """
    # Strip leading punctuation and whitespace
    text = re.sub(r'^[\s.,;:!?\-–—]+', '', text)
    # Fix double spaces
    text = re.sub(r'  +', ' ', text)
    return text.strip()


def neutralize_text(styled_text: str, max_retries: int = 2, monotone: bool = True) -> Optional[str]:
    """Round-Trip Translation neutralization via Mandarin pivot.

    Step 1 (Scrub): English → Mandarin (HSK 5 vocabulary)
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
# Lexical Bleed Filter
# =============================================================================
# Prevents the model from learning copy-paste by rejecting training pairs
# where the neutral input retains too much distinctive vocabulary.

# Common words to ignore in lexical analysis
COMMON_WORDS = {
    'the', 'and', 'of', 'to', 'a', 'in', 'is', 'that', 'for', 'it', 'was',
    'with', 'as', 'be', 'on', 'not', 'this', 'but', 'by', 'from', 'or',
    'have', 'an', 'they', 'which', 'one', 'you', 'were', 'her', 'all',
    'she', 'there', 'would', 'their', 'we', 'him', 'been', 'has', 'when',
    'who', 'will', 'more', 'if', 'no', 'out', 'so', 'said', 'what', 'up',
    'its', 'about', 'into', 'than', 'them', 'can', 'only', 'other', 'new',
    'some', 'could', 'time', 'these', 'two', 'may', 'then', 'do', 'first',
    'any', 'my', 'now', 'such', 'like', 'our', 'over', 'man', 'me', 'even',
    'most', 'made', 'after', 'also', 'did', 'many', 'before', 'must', 'through',
    'back', 'years', 'where', 'much', 'your', 'way', 'well', 'down', 'should',
    'because', 'each', 'just', 'those', 'people', 'how', 'too', 'little', 'state',
    'good', 'very', 'make', 'world', 'still', 'own', 'see', 'men', 'work', 'long',
    'get', 'here', 'between', 'both', 'life', 'being', 'under', 'never', 'day',
    'same', 'another', 'know', 'while', 'last', 'might', 'us', 'great', 'old', 'year',
}


def _bleed_words(text: str) -> List[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def check_lexical_bleed(neutral: str, styled: str, max_overlap: float = 0.50) -> Tuple[bool, float]:
    """Check if neutral input retains too much distinctive vocabulary from styled output.

    If the neutral text already contains most of the distinctive words from the output,
    the model learns copy-paste rather than style transfer.

    Args:
        neutral: Neutralized input text
        styled: Original styled text
        max_overlap: Maximum allowed overlap ratio (default 0.6)

    Returns:
        Tuple of (is_valid, overlap_ratio)
    """
    # Tokenize on letters so "cruelty," and "cruelty" count as the same word.
    neutral_words = set(_bleed_words(neutral))
    styled_words = set(_bleed_words(styled))

    # Find distinctive words in styled text (not common)
    distinctive_styled = styled_words - COMMON_WORDS

    if not distinctive_styled:
        return True, 0.0

    # Find shared distinctive words
    shared = (neutral_words & styled_words) - COMMON_WORDS

    overlap_ratio = len(shared) / len(distinctive_styled)

    # Reject if overlap is too high
    return overlap_ratio < max_overlap, overlap_ratio


# =============================================================================
# Persona-Based Training (Acting Directions, not Translation Instructions)
# =============================================================================
# Instead of "Rewrite X as Y", we put the model inside a scene.
# Formula: [ROLE] + [CONTEXT] + [EMOTIONAL STATE] + [CONSTRAINT]
#
# CRITICAL: Frames are split into NARRATIVE vs CONCEPTUAL to avoid
# instruction-content mismatch. A narrative about "Jervas Dudley" should not
# use "Explain the concept..." prompts, or the model learns that "concepts"
# are stories about people.
#
# Uses shared classifier from src/utils/content_classifier.py to ensure
# training and inference use identical classification logic.

# Import through the src package (PROJECT_ROOT is on sys.path). Importing
# "utils.*" as a top-level package breaks its relative imports.
from src.utils.content_classifier import ContentType, classify_content_type
from src.utils.perturbation import heavy_perturb_text as create_heavy_perturbation, perturb_text


# Persona frames split by content type to avoid instruction-content mismatch
PERSONA_FRAMES = {
    "default": {
        "narrative": [
            "You are recounting events you witnessed firsthand. Describe what happened as if confessing to a close friend.",
            "You are a chronicler recording history. Narrate the sequence of events with weight and significance.",
            "Tell this story as if you're sitting by a fire, speaking to someone who needs to understand what happened.",
        ],
        "conceptual": [
            "You are reverse-engineering an alien device. Describe the hidden logic as 'invisible machinery'.",
            "You are a coroner analyzing a system crash. Treat the failure as the universe reclaiming order.",
            "Describe this complex system as a mindless 'Leviathan' made of billions of dumb parts.",
            "State these facts with the absolute, pitiless precision of a machine.",
        ],
    },
    "H.P. Lovecraft": {
        "narrative": [
            # The "Journal" Frame (narrative)
            "You are writing in a diary by candlelight. Your hand is shaking. You must record what happened, but you are terrified to write it down. Do not summarize; confess.",
            # The "Testimony" Frame
            "You are giving testimony about events you witnessed. Narrate what occurred, but let your dread seep through the clinical details.",
            # The "Letter to a Friend" Frame
            "You are writing a letter to a trusted friend, recounting a sequence of disturbing events. You need them to understand what happened, even if they won't believe you.",
        ],
        "conceptual": [
            # The "Warning" Frame (conceptual)
            "You are writing a desperate letter to a colleague, urging them to destroy their research. Explain this forbidden knowledge as dangerous, something that should not be known.",
            # The "Scholarly Discovery" Frame
            "You are a researcher in the Miskatonic University archives who has found something unsettling. Document this discovery as if the knowledge itself is a threat.",
            # The "Forbidden Text" Frame
            "You are translating an ancient text that reveals terrible truths. Explain the mechanism or principle, but frame it as knowledge that corrupts the knower.",
        ],
    },
    "Douglas Hofstadter": {
        "narrative": [
            # The "Anecdote" Frame
            "You are telling a story to illustrate a point. Narrate the sequence of events, but let your playful voice shine through.",
            # The "Personal Discovery" Frame
            "You are recounting how you personally came to understand something. Tell the story of your realization.",
        ],
        "conceptual": [
            # The "Skeptical Professor" Frame
            "You are a cognitive scientist arguing with a stubborn student during office hours. Explain this concept, but show your frustration at how counter-intuitive it is. Use analogies. Do not lecture; converse.",
            # The "Margin Note" Frame
            "You are scribbling notes in the margins of a dry textbook. Criticize the text for being too rigid. Rephrase the core idea with wit, playfulness, and self-referential humor.",
            # The "Dinner Party" Frame
            "You are explaining a complex idea to a friend at a loud dinner party. Be vivid and punchy. Avoid academic jargon. Use physical objects on the table as metaphors.",
        ],
    },
    "Bertrand Russell": {
        "narrative": [
            # The "Witness to History" Frame
            "You are recounting a historical event you observed firsthand. Describe the facts with clarity and precision, but let your moral conviction show through the selection of details.",
            # The "Personal Reflection" Frame
            "You are writing an autobiographical passage about an experience that shaped your thinking. Be direct, candid, and unsparing in your self-assessment.",
            # The "Cautionary Account" Frame
            "You are describing a sequence of events that illustrates a larger principle about human nature or society. Let the facts speak, but arrange them so the conclusion is inescapable.",
        ],
        "conceptual": [
            # The "Philosophical Argument" Frame
            "You are constructing a philosophical argument for an educated general audience. State the position clearly, consider the strongest objection, and dismantle it with precision. Do not hedge or equivocate.",
            # The "Debunking" Frame
            "You are dismantling a widely held belief that you consider to be nonsense. Be direct and incisive. Use concrete examples to expose the absurdity of the position. Allow yourself dry wit.",
            # The "Exposition" Frame
            "You are explaining a difficult concept to an intelligent reader who lacks specialist knowledge. Be clear and precise without being condescending. Use analogies drawn from common experience.",
            # The "Materialist Case" Frame
            "You are making the case that a phenomenon commonly attributed to mysterious or supernatural causes has a straightforward material explanation. Be methodical, assertive, and spare no sacred cows.",
            # The "Sceptical Inquiry" Frame
            "You are examining a claim with rigorous scepticism. Distinguish what is known from what is merely assumed. Demand evidence and reject appeals to authority or tradition.",
        ],
    },
}

# =============================================================================
# Negative Constraints (Anti-AI Writing Markers)
# =============================================================================
# Tiered constraint system to force human-like writing patterns.

# ALWAYS included (100%) - these are clear AI tells
ALWAYS_CONSTRAINTS = [
    "Do not use: 'Moreover', 'Furthermore', 'Therefore', 'Thus', 'Hence', 'In conclusion', 'It is important to note', 'It is worth noting', 'This highlights', 'This underscores', 'In essence', 'Ultimately'.",
    "Do not hedge. Avoid: 'arguably', 'it could be said', 'one might argue', 'perhaps it is', 'it seems that'. State things directly.",
]

# FREQUENT (70%) - strong anti-pattern
FREQUENT_CONSTRAINTS = [
    "Do not start with a topic sentence. Start with a sensory detail, a question, or mid-thought.",
    "Do not use numbered lists or 'Firstly/Secondly/Thirdly' structures.",
]

# ROTATING (one random, 40%) - stylistic variety
ROTATING_CONSTRAINTS = [
    "Use fragments. Interrupt yourself with dashes (—).",
    "Let ideas collide without transition words.",
    "Do not explain. Imply.",
    "Use at least one rhetorical question.",
    "Interrupt yourself with a parenthetical thought.",
    "Start the paragraph with a conjunction (But, And, Yet, So).",
    "Be biased. Be opinionated. Do not balance your argument.",
    "Vary sentence lengths dramatically. Follow a long sentence with a short one.",
    "Use concrete nouns instead of abstractions. Not 'the concept' but the thing itself.",
    "End on an image or action, not a summary.",
]

# Common concrete nouns to replace with placeholders in Abstract Summary
CONCRETE_NOUNS = {
    'house', 'car', 'tree', 'door', 'window', 'table', 'chair', 'book', 'phone',
    'computer', 'screen', 'keyboard', 'mouse', 'desk', 'floor', 'wall', 'ceiling',
    'room', 'building', 'street', 'road', 'city', 'town', 'village', 'country',
    'mountain', 'river', 'lake', 'ocean', 'sea', 'forest', 'field', 'garden',
    'hand', 'face', 'eye', 'head', 'body', 'arm', 'leg', 'foot', 'finger',
    'sun', 'moon', 'star', 'sky', 'cloud', 'rain', 'snow', 'wind', 'fire',
    'water', 'earth', 'stone', 'rock', 'metal', 'wood', 'glass', 'paper',
    'food', 'bread', 'meat', 'fruit', 'vegetable', 'drink', 'wine', 'beer',
    'man', 'woman', 'child', 'person', 'people', 'animal', 'dog', 'cat', 'bird',
    'machine', 'engine', 'wheel', 'tool', 'weapon', 'knife', 'gun', 'sword',
}


# =============================================================================
# Rhetorical Structure Analysis
# =============================================================================

def classify_rhetorical_move(sentence: str) -> str:
    """Classify the rhetorical move of a sentence.

    Returns one of: Metaphor, Personal Observation, Technical Definition,
    Rhetorical Question, Contrast, Consequence, or Observation (default).
    """
    sent_lower = sentence.lower()

    # Detect patterns
    if any(w in sent_lower for w in ['like', 'as if', 'resembles', 'similar to', 'as though']):
        return "Metaphor"
    if sent_lower.startswith(('i ', 'we ', 'one ', "i'm ", "we're ")):
        return "Personal Observation"
    if any(w in sent_lower for w in ['defined as', 'refers to', 'means that', 'is called']):
        return "Technical Definition"
    if '?' in sentence:
        return "Rhetorical Question"
    if any(w in sent_lower.split()[:3] for w in ['however', 'but', 'yet', 'although', 'while']):
        return "Contrast"
    if any(w in sent_lower for w in ['because', 'therefore', 'thus', 'hence', 'consequently']):
        return "Consequence"

    return "Observation"


def extract_rhetorical_skeleton(text: str) -> str:
    """Extract rhetorical structure from text.

    Returns skeleton like: [Observation] -> [Metaphor] -> [Technical Definition]
    """
    nlp = get_nlp()
    doc = nlp(text)
    sentences = list(doc.sents)

    if not sentences:
        return "[Observation]"

    skeleton_parts = []
    for sent in sentences:
        move = classify_rhetorical_move(sent.text)
        skeleton_parts.append(f"[{move}]")

    return " -> ".join(skeleton_parts)


# =============================================================================
# Many-to-One Input Variants
# =============================================================================

# Modifiers that carry the claim itself. Dropping "never", "only" or "always"
# flips or changes what a sentence says, so information dropout keeps them.
MEANING_MODIFIERS = frozenset({
    'not', "n't", 'never', 'no', 'only', 'always', 'often', 'sometimes', 'rarely',
    'seldom', 'hardly', 'scarcely', 'barely', 'almost', 'nearly', 'even', 'also',
    'too', 'still', 'yet', 'already', 'again', 'ever', 'once', 'just', 'merely',
    'solely', 'mostly', 'usually', 'generally', 'entirely', 'wholly', 'partly',
    'less', 'more', 'most', 'least', 'very', 'so', 'as', 'then', 'now', 'here',
    'there', 'perhaps', 'probably', 'possibly', 'certainly', 'necessarily',
    'many', 'much', 'few', 'several', 'all', 'some', 'any', 'each', 'every',
    'other', 'same', 'such', 'own', 'first', 'last', 'next', 'false', 'true',
    'possible', 'impossible', 'necessary', 'certain', 'uncertain',
})


def strip_modifiers(text: str) -> str:
    """Drop decorative modifiers (Information Dropout).

    Only plain attributive adjectives ("the clever student") and -ly manner
    adverbs go. Negation, frequency, degree and quantity words, comparatives,
    superlatives and predicate adjectives ("the problem is hard") carry the
    claim and are kept.

    Uses token.text_with_ws to preserve original whitespace and contractions
    (spaCy splits "doesn't" into ["does", "n't"] — using text_with_ws keeps
    the original spacing so contractions re-join correctly).
    """
    nlp = get_nlp()
    doc = nlp(text)
    parts = []

    for token in doc:
        lower = token.text.lower()
        droppable = lower not in MEANING_MODIFIERS and token.dep_ != 'neg' and (
            (token.pos_ == 'ADJ' and token.tag_ == 'JJ' and token.dep_ == 'amod')
            or (token.pos_ == 'ADV' and token.tag_ == 'RB' and lower.endswith('ly'))
        )
        if not droppable:
            parts.append(token.text_with_ws)
        elif parts and token.whitespace_ == '' :
            # Keep the space before punctuation that followed the dropped word.
            parts[-1] = parts[-1].rstrip()

    result = ''.join(parts)
    result = re.sub(r'  +', ' ', result)
    return result.strip()


def is_concrete_noun(word: str) -> bool:
    """Check if a word is a concrete noun."""
    return word.lower() in CONCRETE_NOUNS


def remove_concrete_nouns(text: str) -> str:
    """Remove specific concrete nouns to force metaphor hallucination (Abstract Summary).

    Replaces concrete nouns with [THING] placeholder, forcing the model
    to generate the author's characteristic metaphors and imagery.
    Uses token.text_with_ws to preserve contractions and original spacing.
    """
    nlp = get_nlp()
    doc = nlp(text)
    parts = []

    for token in doc:
        # Replace concrete nouns with generic placeholders
        if token.pos_ == 'NOUN' and token.ent_type_ == '' and is_concrete_noun(token.text):
            parts.append('[THING]' + token.whitespace_)
        else:
            parts.append(token.text_with_ws)

    result = ''.join(parts)
    result = re.sub(r'  +', ' ', result)
    return result.strip()


def create_input_variants(styled_text: str, standard_neutral: str) -> List[Tuple[str, str]]:
    """Create 3 input variants for Many-to-One mapping.

    Prevents the model from memorizing 1:1 input→output mappings by providing
    multiple different inputs that all map to the same styled output.

    Returns:
        List of (neutral_input, variant_type) tuples:
        - ("standard neutral text", "standard")
        - ("text without adjectives/adverbs", "info_dropout")
        - ("text with [THING] placeholders", "abstract")
    """
    variants = []

    # Input A: Standard Neutralization (existing RTT)
    variants.append((standard_neutral, "standard"))

    # Input B: Information Dropout (strip adjectives/adverbs)
    stripped = strip_modifiers(standard_neutral)
    if stripped and len(stripped.split()) >= 10:  # Ensure not too short
        variants.append((stripped, "info_dropout"))

    # Input C: Abstract Summary (remove concrete nouns)
    abstract = remove_concrete_nouns(standard_neutral)
    if abstract and len(abstract.split()) >= 10:  # Ensure not too short
        variants.append((abstract, "abstract"))

    return variants


# =============================================================================
# Persona Instruction Generation
# =============================================================================

def get_persona_instruction(
    author: str,
    word_count: int,
    styled_text: str = None,
    use_skeleton: bool = True,
    input_text: str = None,
) -> str:
    """Generate persona-based instruction with tiered constraints.

    Formula: [ROLE] + [CONTEXT] + [EMOTIONAL STATE] + [CONSTRAINTS]

    Instead of "Rewrite X as Y" (translation), this puts the model inside a scene
    (acting direction), forcing it to learn the PERSONA not just a prompt string.

    CRITICAL: Classifies content as NARRATIVE or CONCEPTUAL to avoid instruction-
    content mismatch. A story about "Jervas Dudley" uses narrative frames like
    "Narrate the sequence of events...", while explanatory content uses conceptual
    frames like "Explain the mechanism...".

    Constraint tiers:
    - ALWAYS_CONSTRAINTS: 100% - ban clear AI tells (Moreover, hedging, etc.)
    - FREQUENT_CONSTRAINTS: 70% each - strong anti-patterns (topic sentences, lists)
    - ROTATING_CONSTRAINTS: 40% - one random stylistic constraint

    Args:
        author: Author name to get persona frames for
        word_count: Target word count
        styled_text: Original styled text (for skeleton extraction)
        use_skeleton: If True, 50% chance to include rhetorical skeleton
        input_text: Neutral input the model sees (for content classification)

    Returns:
        Persona-based instruction string
    """
    # Classify content type (NARRATIVE vs CONCEPTUAL). Inference only sees the
    # neutral input, so classify that, not the styled target.
    classify_source = input_text or styled_text
    if classify_source:
        content_type = classify_content_type(classify_source)
    else:
        content_type = ContentType.NARRATIVE  # Default for missing text

    # Get author-specific frames or fall back to default
    author_frames = PERSONA_FRAMES.get(author, PERSONA_FRAMES["default"])

    # Select frames matching content type
    type_key = content_type.value  # "narrative" or "conceptual"
    frames = author_frames.get(type_key, author_frames.get("narrative", []))

    if not frames:
        # Fallback to default frames
        frames = PERSONA_FRAMES["default"].get(type_key, PERSONA_FRAMES["default"]["narrative"])

    persona_frame = random.choice(frames)

    # Add word count constraint
    instruction = f"{persona_frame}\n\nWrite approximately {word_count} words."

    # Add structural skeleton 50% of the time (Grafting Prep)
    if use_skeleton and styled_text and random.random() < 0.50:
        skeleton = extract_rhetorical_skeleton(styled_text)
        instruction = f"{instruction}\n\nFollow this structure: {skeleton}"

    # Build constraints section
    constraints = []

    # ALWAYS constraints (100%) - clear AI tells
    constraints.extend(ALWAYS_CONSTRAINTS)

    # FREQUENT constraints (70% each)
    for constraint in FREQUENT_CONSTRAINTS:
        if random.random() < 0.70:
            constraints.append(constraint)

    # ROTATING constraints (one random, 40%)
    if random.random() < 0.40:
        constraints.append(random.choice(ROTATING_CONSTRAINTS))

    # Add constraints to instruction
    if constraints:
        constraints_text = "\n".join(f"[CONSTRAINT]: {c}" for c in constraints)
        instruction = f"{instruction}\n\n{constraints_text}"

    return instruction


def format_training_example(
    neutral_text: str,
    styled_text: str,
    author: str,
    word_count: int,
    variation_type: str = "original",
    output_format: str = "llama_factory",
) -> dict:
    """Format a training example for LoRA training.

    Supports two output formats:
    - llama_factory: {"instruction": "...", "input": "...", "output": "..."}
    - mlx: {"text": "prompt + completion"} for base models

    Uses Acting Directions (not Translation Instructions):
    - Persona Frame: Random scenario trigger from PERSONA_FRAMES
    - Structural Skeleton: 50% chance to include rhetorical structure
    - Negative Constraints: 30% chance to add ONE anti-AI-writing rule
    """
    # Persona-based instruction (Acting Direction)
    instruction = get_persona_instruction(
        author=author,
        word_count=word_count,
        styled_text=styled_text,
        input_text=neutral_text,
    )

    # Apply perturbation based on variation type
    # info_dropout and abstract variants are already processed
    if variation_type == "robustness":
        perturbed_input = create_heavy_perturbation(neutral_text)
    elif variation_type in ("info_dropout", "abstract"):
        # These variants are already processed, just apply light perturbation
        perturbed_input = perturb_text(neutral_text, perturbation_rate=0.05)
    else:
        perturbed_input = perturb_text(neutral_text)

    if output_format == "llama_factory":
        # LLaMA-Factory SFT format: {"instruction": "...", "input": "...", "output": "..."}
        return {
            "instruction": instruction,
            "input": perturbed_input,
            "output": styled_text,
        }
    else:
        # MLX format: {"text": "prompt + completion"} for base models
        # With mask_prompt=true, only the completion (after prompt) is trained
        prompt = f"{instruction}\n\n{perturbed_input}\n###\n"
        return {
            "text": prompt + styled_text,
            "prompt": prompt,
            "word_count": word_count,
            "variation_type": variation_type,
        }


def _read_processed_indices(output_path: Path) -> set:
    """Chunk indices already written to ``output_path``.

    Exits rather than returning an empty set when the file has rows but no
    source ids: resuming would then rewrite it from scratch and lose them.
    """
    processed = set()
    rows = 0
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows += 1
            if 'source_idx' in entry:
                processed.add(entry['source_idx'])
    if rows and not processed:
        logger.error(
            f"{output_path} has {rows} rows without source_idx, so --resume can't tell "
            "which chunks are done. Move the file aside or run without --resume."
        )
        sys.exit(1)
    return processed


def generate_training_data(
    chunks: List[Item],
    author: str,
    output_path: Path,
    workers: int = 1,
    monotone: bool = False,
    resume: bool = False,
    output_format: str = "llama_factory",
) -> int:
    """Generate training data using RTT neutralization, writing progressively.

    Processes chunks sequentially (MLX requires single-threaded access) and
    writes each result immediately. Progress is saved on interrupt.

    Uses persona-based training:
    - Acting Directions (not Translation Instructions)
    - Many-to-One mapping: 3 variants per anchor (standard, info_dropout, abstract)
    - Structural skeletons: 50% chance to include rhetorical structure
    - Negative constraints: 30% chance to add ONE anti-AI-writing rule

    Every row carries ``source_idx`` (its chunk) and ``source_paragraphs``
    (corpus paragraphs it came from), in both formats. Resume uses the first,
    the grouped train/val split the second.

    Args:
        chunks: List of (styled_text, variation_type, source_paragraphs) tuples
        author: Author name for system prompt
        output_path: Output JSONL file path
        workers: Unused (kept for API compatibility)
        monotone: If True, apply extra flattening step (slower, +50% time)
        resume: If True, skip already-processed items and append to file
        output_format: Output format - 'llama_factory' or 'mlx'

    Returns:
        Number of examples written
    """
    total = len(chunks)
    mode = "monotone" if monotone else "standard"

    processed_indices = set()
    if resume and output_path.exists():
        processed_indices = _read_processed_indices(output_path)
        logger.info(f"Resuming: found {len(processed_indices)} already processed items")

    remaining = total - len(processed_indices)
    logger.info(f"Generating training data for {remaining}/{total} chunks ({mode} RTT)...")
    logger.info(f"Writing progressively to {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failed_count = 0
    type_counts = {}
    start_time = time.time()

    # Get batch size and concurrent batches from neutralizer
    neutralizer, _ = get_rtt_neutralizer()
    batch_size = getattr(neutralizer, 'batch_size', 1)
    concurrent_batches = getattr(neutralizer, 'concurrent_batches', 1)
    use_batching = isinstance(batch_size, int) and batch_size > 1 and hasattr(neutralizer, 'neutralize_batch')

    # Super-batch = batch_size * concurrent_batches (to fully utilize parallelism)
    # e.g., batch_size=10, concurrent_batches=4 → send 40 texts at once
    super_batch_size = batch_size * concurrent_batches if use_batching else 1

    if use_batching:
        logger.info(f"Using batched RTT with batch_size={batch_size}, concurrent={concurrent_batches}, super_batch={super_batch_size}")
    else:
        logger.info("Using sequential RTT (batch_size=1)")

    pending_chunks = [
        (idx, styled_text, vtype, tuple(src))
        for idx, (styled_text, vtype, src) in enumerate(chunks)
        if idx not in processed_indices
    ]

    def write_rows(f, idx, styled_text, vtype, src, neutral) -> None:
        nonlocal success_count, failed_count
        neutral = clean_neutral_text(neutral)
        word_count = len(styled_text.split())

        # Many-to-One: 3 input variants per anchor (standard, info_dropout, abstract)
        if vtype == "original":
            variants = create_input_variants(styled_text, neutral)
        else:
            variants = [(neutral, vtype)]

        for variant_neutral, variant_type in variants:
            # Lexical bleed filter: reject if neutral retains too much distinctive vocabulary
            is_valid, overlap_ratio = check_lexical_bleed(variant_neutral, styled_text)
            if not is_valid:
                failed_count += 1
                logger.debug(f"  [{idx}] ✗ Lexical bleed ({overlap_ratio:.0%} overlap) for {variant_type}")
                continue

            example = format_training_example(
                neutral_text=variant_neutral,
                styled_text=styled_text,
                author=author,
                word_count=word_count,
                variation_type=variant_type,
                output_format=output_format,
            )
            example["source_idx"] = idx
            example["source_paragraphs"] = list(src)
            example["variation_type"] = variant_type
            if output_format == "mlx":
                example["many_to_one"] = len(variants) > 1

            f.write(json.dumps(example, ensure_ascii=False) + '\n')
            success_count += 1
            type_counts[variant_type] = type_counts.get(variant_type, 0) + 1

    # Append if resuming, otherwise overwrite
    file_mode = 'a' if resume and processed_indices else 'w'
    with open(output_path, file_mode, encoding='utf-8') as f:
        for batch_start in range(0, len(pending_chunks), super_batch_size):
            batch = pending_chunks[batch_start:batch_start + super_batch_size]

            if batch_start % max(super_batch_size, 10) == 0:
                elapsed = time.time() - start_time
                processed = len(processed_indices) + batch_start
                rate = batch_start / elapsed if elapsed > 0 and batch_start else 0
                eta = (total - processed) / rate if rate > 0 else 0
                logger.info(
                    f"[{processed}/{total}] ({processed*100//max(total, 1)}%) | "
                    f"✓{success_count} ✗{failed_count} | "
                    f"{rate:.2f}/s | ETA: {eta/60:.1f}m"
                )

            if use_batching:
                # Retries are handled inside the queue-based pipeline
                try:
                    neutrals = neutralize_batch([styled for _, styled, _, _ in batch], monotone=monotone)
                except Exception as e:
                    logger.warning(f"Batch RTT error: {e}")
                    neutrals = [None] * len(batch)
            else:
                neutrals = []
                for idx, styled_text, _, _ in batch:
                    neutral = None
                    for retry in range(3):
                        try:
                            neutral = neutralize_text(styled_text, monotone=monotone)
                        except Exception as e:
                            logger.debug(f"  [{idx}] RTT attempt {retry + 1} error: {e}")
                        if neutral:
                            break
                    if not neutral:
                        logger.warning(f"  [{idx}] ✗ All retries exhausted ({len(styled_text.split())}w)")
                    neutrals.append(neutral)

            for (idx, styled_text, vtype, src), neutral in zip(batch, neutrals):
                if neutral:
                    write_rows(f, idx, styled_text, vtype, src, neutral)
                else:
                    failed_count += 1

            f.flush()

    elapsed = time.time() - start_time
    logger.info(
        f"Complete: {success_count} examples written from {remaining} chunks "
        f"({failed_count} rejected or failed) in {elapsed:.1f}s"
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
    parser.add_argument("--skip-perspective", action="store_true",
                        help="Skip perspective variation step (no first_person_plural, third_person, impersonal)")
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
    parser.add_argument("--resume", action="store_true", help="Resume from last processed item (checks train.jsonl)")
    parser.add_argument("--save-intermediate", type=str, default=None, help="Save intermediate paragraphs")
    parser.add_argument("--from-selected", type=str, default=None,
                        help="Load from selected paragraphs JSON (from select_diverse_paragraphs.py)")
    parser.add_argument("--format", choices=["llama_factory", "mlx"], default="llama_factory",
                        help="Output format: llama_factory (default) or mlx")
    parser.add_argument("--skip-curation", action="store_true",
                        help="Skip curation step (corpus is already curated, one paragraph per double-newline)")
    parser.add_argument("--val-fraction", type=float, default=0.05,
                        help="Share of source paragraphs held out for validation (default: 0.05)")
    parser.add_argument("--no-nli", action="store_true",
                        help="Skip the two-way entailment filter when writing LlamaFactory splits")
    parser.add_argument("--snowflake-topics", type=str, default=None,
                        help="Path to Python file with custom snowflake topics list "
                             "(e.g., data/training/russell/snowflake_topics.py). "
                             "Falls back to built-in MUNDANE_TOPICS if not specified.")

    args = parser.parse_args()

    if not args.resume_from and not args.resume_from_chunks and not args.corpus and not args.from_selected:
        parser.error("--corpus or --from-selected is required unless using --resume-from or --resume-from-chunks")

    overall_start = time.time()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load custom snowflake topics if provided
    snowflake_topics = load_snowflake_topics(args.snowflake_topics)

    variation_counts = {}

    # Check if resuming from chunks file (skips Steps 1-3)
    if args.resume_from_chunks:
        logger.info("=" * 60)
        logger.info("RESUMING FROM CHUNKS FILE (skipping Steps 1-3)")
        logger.info("=" * 60)

        chunks = load_intermediate(Path(args.resume_from_chunks), stage="chunks")
        for _, vtype, _ in chunks:
            variation_counts[vtype] = variation_counts.get(vtype, 0) + 1
        logger.info(f"Chunks: {len(chunks)} total | Breakdown: {variation_counts}")

    # Check if resuming from intermediate file (skips Steps 1-2)
    elif args.resume_from:
        logger.info("=" * 60)
        logger.info("RESUMING FROM INTERMEDIATE FILE (skipping Steps 1-2)")
        logger.info("=" * 60)

        expanded = load_intermediate(Path(args.resume_from), stage="paragraphs")
        for _, vtype, _ in expanded:
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
        actual_tokens = metadata.get('actual_tokens')
        logger.info(f"  Tokens: {actual_tokens:,}" if isinstance(actual_tokens, (int, float)) else "  Tokens: unknown")
        mean_quality = metadata.get('mean_quality')
        logger.info(f"  Mean quality: {mean_quality:.3f}" if isinstance(mean_quality, (int, float)) else "  Mean quality: unknown")

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
            skip_perspective=args.skip_perspective,
            snowflake_topics=snowflake_topics,
        )

        for _, vtype, _ in expanded:
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

        if args.skip_curation:
            # Skip curation - corpus is already curated (one paragraph per double-newline)
            logger.info("=" * 60)
            logger.info("STEP 1: Skipping curation (--skip-curation flag)")
            logger.info("=" * 60)

            # Just split by double-newlines and strip
            paragraphs = [p.strip() for p in raw_text.split('\n\n') if p.strip()]
            logger.info(f"Loaded {len(paragraphs)} pre-curated paragraphs")
        else:
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
            skip_perspective=args.skip_perspective,
            snowflake_topics=snowflake_topics,
        )

        for _, vtype, _ in expanded:
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

    # Output directly to train.jsonl
    train_output_path = output_dir / "train.jsonl"
    logger.info(f"Using PERSONA-based training (Acting Directions + Many-to-One)")
    logger.info(f"Output format: {args.format}")
    num_examples = generate_training_data(
        chunks, args.author, train_output_path,
        workers=args.workers, monotone=not args.no_monotone,
        resume=args.resume, output_format=args.format
    )

    # Step 5: filter and split by source paragraph for LlamaFactory
    if args.format == "llama_factory":
        logger.info("=" * 60)
        logger.info("STEP 5: Filtering rows and writing train/val splits")
        logger.info("=" * 60)
        from filter_training_data import finalize
        finalize(
            train_output_path,
            output_dir / "LlamaFactory",
            dataset_name=output_dir.name,
            val_fraction=args.val_fraction,
            nli=not args.no_nli,
            log=logger.info,
        )

    # Summary
    total_time = time.time() - overall_start
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Author: {args.author}")
    logger.info(f"Format: {args.format}")
    logger.info(f"Variation breakdown: {variation_counts}")
    logger.info(f"Overlapping chunks: {len(chunks)}")
    logger.info(f"Training examples: {num_examples}")
    logger.info(f"Output: {train_output_path}")
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")


if __name__ == "__main__":
    main()
