#!/usr/bin/env python3
"""Generate training data using the 50/50 Split Strategy.

Phased approach for efficient training data generation:
- Phase 1: RTT all chunks → Anchors (core vocabulary mapping)
- Phase 2: Even chunks → Noise variants (input perturbation, zero LLM cost)
- Phase 3: Odd chunks → Snowflakes (topic swap via LLM)

Total: ~8,600 examples from 4,300 chunks (~1.7M tokens)

Usage:
    # Phase 1: Generate Anchors (RTT all chunks)
    python scripts/generate_phased_training.py \
        --chunks data/training/lovecraft/chunks_150.json \
        --author "H.P. Lovecraft" \
        --output data/training/lovecraft_150 \
        --phase 1 \
        --workers 4

    # Phase 2: Add Noise variants (script only, fast)
    python scripts/generate_phased_training.py \
        --chunks data/training/lovecraft/chunks_150.json \
        --author "H.P. Lovecraft" \
        --output data/training/lovecraft_150 \
        --phase 2

    # Phase 3: Add Snowflakes (LLM topic swap)
    python scripts/generate_phased_training.py \
        --chunks data/training/lovecraft/chunks_150.json \
        --author "H.P. Lovecraft" \
        --output data/training/lovecraft_150 \
        --phase 3 \
        --workers 8

    # Merge all phases into train/valid/test
    python scripts/generate_phased_training.py \
        --chunks data/training/lovecraft/chunks_150.json \
        --author "H.P. Lovecraft" \
        --output data/training/lovecraft_150 \
        --phase 1 --merge
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path

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
# Progress Tracking
# =============================================================================

class ProgressTracker:
    """Track progress with rate and ETA calculations."""

    def __init__(self, total: int, name: str = "Progress"):
        self.total = total
        self.name = name
        self.start_time = time.time()
        self.processed = 0
        self.success = 0
        self.failed = 0
        self.last_log_time = 0
        self.log_interval = 5  # Log every 5 seconds minimum

    def update(self, success: bool = True):
        """Update progress counters."""
        self.processed += 1
        if success:
            self.success += 1
        else:
            self.failed += 1

    def should_log(self) -> bool:
        """Check if we should log (rate limited)."""
        now = time.time()
        if now - self.last_log_time >= self.log_interval:
            self.last_log_time = now
            return True
        return False

    def log(self, force: bool = False):
        """Log current progress."""
        if not force and not self.should_log():
            return

        elapsed = time.time() - self.start_time
        rate = self.processed / elapsed if elapsed > 0 else 0
        remaining = self.total - self.processed
        eta_seconds = remaining / rate if rate > 0 else 0

        # Format ETA
        if eta_seconds < 60:
            eta_str = f"{eta_seconds:.0f}s"
        elif eta_seconds < 3600:
            eta_str = f"{eta_seconds/60:.1f}m"
        else:
            eta_str = f"{eta_seconds/3600:.1f}h"

        pct = self.processed * 100 // self.total if self.total > 0 else 0

        status = f"[{self.processed}/{self.total}] ({pct}%)"
        stats = f"✓{self.success} ✗{self.failed}"
        timing = f"{rate:.2f}/s | ETA: {eta_str}"

        logger.info(f"{self.name}: {status} | {stats} | {timing}")

    def finish(self):
        """Log final summary."""
        elapsed = time.time() - self.start_time
        rate = self.processed / elapsed if elapsed > 0 else 0

        if elapsed < 60:
            time_str = f"{elapsed:.1f}s"
        else:
            time_str = f"{elapsed/60:.1f}m"

        logger.info(
            f"{self.name} complete: {self.success} success, {self.failed} failed "
            f"in {time_str} ({rate:.2f}/s)"
        )


# =============================================================================
# Prompt Templates and Perturbation
# =============================================================================

PROMPT_TEMPLATES = [
    "Rewrite in {author}'s style (~{word_count} words):",
    "Transform this text into {author}'s voice (~{word_count} words):",
    "Convert to {author}'s style (~{word_count} words):",
    "Apply {author}'s style (~{word_count} words):",
    "Rewrite the following in {author}'s voice (~{word_count} words):",
    "Style transfer to {author} (~{word_count} words):",
]

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

MUNDANE_TOPICS = [
    "making toast for breakfast", "doing the weekly laundry", "organizing a closet",
    "washing dishes after dinner", "vacuuming the living room", "folding clean towels",
    "watering houseplants", "making the bed", "cleaning the bathroom mirror",
    "filing tax returns", "attending a staff meeting", "writing a work email",
    "waiting in line at the DMV", "filling out insurance forms", "updating a spreadsheet",
    "grocery shopping on Saturday", "pumping gas at the station", "returning library books",
    "picking up dry cleaning", "waiting for a bus", "walking to the mailbox",
    "brewing morning coffee", "tying shoelaces", "checking the weather forecast",
    "setting an alarm clock", "microwaving leftovers", "packing a lunch",
]


def perturb_text(text: str, rate: float = 0.08) -> str:
    """Apply light perturbations (8% rate)."""
    words = text.split()
    result = []
    droppable = {'the', 'a', 'an', 'very', 'really', 'just', 'quite'}

    for word in words:
        if random.random() > rate:
            result.append(word)
            continue

        choice = random.random()
        if choice < 0.4:
            word_lower = word.lower().rstrip('.,!?;:')
            if word_lower in SYNONYMS:
                synonym = random.choice(SYNONYMS[word_lower])
                if word[0].isupper():
                    synonym = synonym.capitalize()
                result.append(synonym + word[len(word_lower):])
            else:
                result.append(word)
        elif choice < 0.7:
            if word.lower() not in droppable:
                result.append(word)
        else:
            if len(word) > 3:
                i = random.randint(1, len(word) - 2)
                word = word[:i] + word[i+1] + word[i] + word[i+2:]
            result.append(word)

    return ' '.join(result)


def heavy_perturb_text(text: str, rate: float = 0.15) -> str:
    """Apply heavy perturbations for Noise variants (15% rate)."""
    words = text.split()
    result = []
    droppable = {'the', 'a', 'an', 'very', 'really', 'just', 'quite', 'some', 'this', 'that'}

    for word in words:
        if random.random() > rate:
            result.append(word)
            continue

        choice = random.random()
        if choice < 0.30:
            word_lower = word.lower().rstrip('.,!?;:')
            if word_lower in SYNONYMS:
                synonym = random.choice(SYNONYMS[word_lower])
                if word[0].isupper():
                    synonym = synonym.capitalize()
                result.append(synonym + word[len(word_lower):])
            else:
                result.append(word)
        elif choice < 0.50:
            if word.lower() not in droppable:
                result.append(word)
        elif choice < 0.75:
            if len(word) > 3:
                i = random.randint(1, len(word) - 2)
                word = word[:i] + word[i+1] + word[i] + word[i+2:]
            result.append(word)
        elif choice < 0.90:
            if len(word) > 2:
                i = random.randint(0, len(word) - 1)
                word = word[:i] + word[i] + word[i:]
            result.append(word)
        else:
            if random.random() < 0.5:
                result.append(word.lower())
            else:
                result.append(word.upper() if len(word) <= 4 else word)

    return ' '.join(result)


def generate_style_tag(text: str) -> str:
    """Generate structural style tag."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if not sentences:
        return "[STYLE: Simple]"

    lengths = [len(s.split()) for s in sentences]
    avg = sum(lengths) / len(lengths)
    std = (sum((x - avg) ** 2 for x in lengths) / len(lengths)) ** 0.5 if len(lengths) > 1 else 0

    complex_markers = [';', '—', '--', ':', '(', ')']
    has_complex = any(m in text for m in complex_markers)

    if avg < 12:
        length_tag = "Short & Punchy"
    elif avg > 25:
        length_tag = "Long & Flowing"
    elif std > 8:
        length_tag = "Varied Lengths"
    else:
        length_tag = "Medium Length"

    complexity_tag = "Complex Syntax" if has_complex else "Simple Syntax"
    return f"[STYLE: {length_tag} | {complexity_tag}]"


def format_example(neutral: str, styled: str, author: str, variation_type: str) -> dict:
    """Format a training example for base model."""
    word_count = len(styled.split())
    template = random.choice(PROMPT_TEMPLATES)
    instruction = template.format(author=author, word_count=word_count)
    style_tag = generate_style_tag(styled)

    # Apply perturbation based on type
    if variation_type == "noise":
        perturbed = heavy_perturb_text(neutral)
    else:
        perturbed = perturb_text(neutral)

    prompt = f"{instruction}\n{style_tag}\n{perturbed}\n###\n"

    return {
        "text": prompt + styled,
        "prompt": prompt,
        "word_count": word_count,
        "variation_type": variation_type,
    }


# =============================================================================
# Phase 1: Anchors (RTT Neutralization)
# =============================================================================

def print_next_step(phase: int, chunks_path: str, author: str, output_dir: Path, workers: int):
    """Print the command for the next phase."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("NEXT STEP")
    logger.info("=" * 60)

    if phase == 1:
        logger.info("Run Phase 2 (Noise variants - instant, no LLM cost):")
        logger.info("")
        logger.info(f"  ./venv/bin/python3 scripts/generate_phased_training.py \\")
        logger.info(f"      --chunks {chunks_path} \\")
        logger.info(f"      --author \"{author}\" \\")
        logger.info(f"      --output {output_dir} \\")
        logger.info(f"      --phase 2")
    elif phase == 2:
        logger.info("Option A: Merge now (skip Snowflakes, ~6,500 examples):")
        logger.info("")
        logger.info(f"  ./venv/bin/python3 scripts/generate_phased_training.py \\")
        logger.info(f"      --chunks {chunks_path} \\")
        logger.info(f"      --author \"{author}\" \\")
        logger.info(f"      --output {output_dir} \\")
        logger.info(f"      --phase 2 --merge")
        logger.info("")
        logger.info("Option B: Run Phase 3 first (Snowflakes - adds ~2,150 examples):")
        logger.info("")
        logger.info(f"  ./venv/bin/python3 scripts/generate_phased_training.py \\")
        logger.info(f"      --chunks {chunks_path} \\")
        logger.info(f"      --author \"{author}\" \\")
        logger.info(f"      --output {output_dir} \\")
        logger.info(f"      --phase 3 --workers {workers}")
    elif phase == 3:
        logger.info("Run merge to create train/valid/test splits:")
        logger.info("")
        logger.info(f"  ./venv/bin/python3 scripts/generate_phased_training.py \\")
        logger.info(f"      --chunks {chunks_path} \\")
        logger.info(f"      --author \"{author}\" \\")
        logger.info(f"      --output {output_dir} \\")
        logger.info(f"      --phase 3 --merge")


def run_phase1(chunks: list, author: str, output_dir: Path, monotone: bool = True, workers: int = 4,
               chunks_path: str = None):
    """Phase 1: Generate Anchors by RTT neutralizing all chunks.

    Args:
        chunks: List of chunk dicts with 'text' field
        author: Author name for formatting
        output_dir: Output directory
        monotone: Apply monotone flattening
        workers: Number of concurrent workers for batch processing
        chunks_path: Path to chunks file (for printing next step)
    """
    from src.llm.mlx_provider import create_rtt_neutralizer, DeepSeekRTTNeutralizer

    output_file = output_dir / "anchors.jsonl"

    logger.info("=" * 60)
    logger.info("PHASE 1: Generating Anchors via RTT Neutralization")
    logger.info("=" * 60)
    logger.info(f"Total chunks: {len(chunks)}")
    logger.info(f"Output: {output_file}")
    logger.info(f"Monotone flattening: {monotone}")
    logger.info(f"Workers: {workers}")

    # Check for resume
    processed = set()
    if output_file.exists():
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if 'source_idx' in entry:
                        processed.add(entry['source_idx'])
                except:
                    pass
        if processed:
            logger.info(f"Resuming from checkpoint: {len(processed)} already processed")

    pending = [(i, c) for i, c in enumerate(chunks) if i not in processed]
    if not pending:
        logger.info("All chunks already processed!")
        return

    remaining = len(pending)
    logger.info(f"Remaining to process: {remaining}")

    # Initialize neutralizer
    neutralizer = create_rtt_neutralizer()
    batch_size = getattr(neutralizer, 'batch_size', 1)
    is_deepseek = isinstance(neutralizer, DeepSeekRTTNeutralizer)

    # Configure concurrent batches for DeepSeek
    if is_deepseek and hasattr(neutralizer, 'concurrent_batches'):
        neutralizer.concurrent_batches = workers
        logger.info(f"DeepSeek concurrent batches: {workers}")

    logger.info(f"Neutralizer: {type(neutralizer).__name__}")
    logger.info(f"Batch size: {batch_size}")
    logger.info("-" * 60)

    progress = ProgressTracker(len(chunks), "Phase 1")
    progress.processed = len(processed)
    progress.success = len(processed)

    with open(output_file, 'a') as f:
        if is_deepseek and batch_size > 1:
            # Batch processing for DeepSeek
            # Send batch_size * workers texts at once for true parallelism
            mega_batch_size = batch_size * workers
            logger.info(f"Mega-batch size: {mega_batch_size} ({batch_size} x {workers} workers)")

            for mega_start in range(0, len(pending), mega_batch_size):
                mega_batch = pending[mega_start:mega_start + mega_batch_size]
                texts = [chunks[i]['text'] for i, _ in mega_batch]

                progress.log()
                logger.info(f"Processing {len(texts)} texts in {(len(texts) + batch_size - 1) // batch_size} batches ({workers} concurrent)")

                try:
                    neutrals = neutralizer.neutralize_batch(texts, monotone=monotone)
                except Exception as e:
                    logger.warning(f"Mega-batch error: {e}")
                    neutrals = [None] * len(mega_batch)

                for (idx, chunk), neutral in zip(mega_batch, neutrals):
                    if neutral:
                        example = format_example(
                            neutral=neutral,
                            styled=chunk['text'],
                            author=author,
                            variation_type="anchor"
                        )
                        example['source_idx'] = idx
                        example['neutral'] = neutral  # Save for Phase 2/3
                        f.write(json.dumps(example) + '\n')
                        progress.update(success=True)
                    else:
                        progress.update(success=False)

                f.flush()
        else:
            # Sequential processing (MLX or single-threaded)
            for idx, chunk in pending:
                progress.log()

                try:
                    neutral = neutralizer.neutralize(chunk['text'], monotone=monotone)
                except Exception as e:
                    logger.warning(f"RTT error for chunk {idx}: {e}")
                    neutral = None

                if neutral:
                    example = format_example(
                        neutral=neutral,
                        styled=chunk['text'],
                        author=author,
                        variation_type="anchor"
                    )
                    example['source_idx'] = idx
                    example['neutral'] = neutral
                    f.write(json.dumps(example) + '\n')
                    f.flush()
                    progress.update(success=True)
                else:
                    progress.update(success=False)

    progress.finish()

    # Print next step
    if chunks_path:
        print_next_step(1, chunks_path, author, output_dir, workers)


# =============================================================================
# Phase 2: Noise Variants (Script Only - Zero LLM Cost)
# =============================================================================

def run_phase2(chunks: list, author: str, output_dir: Path, chunks_path: str = None, workers: int = 4):
    """Phase 2: Generate Noise variants for even-indexed chunks (script only)."""
    anchors_file = output_dir / "anchors.jsonl"
    output_file = output_dir / "noise.jsonl"

    logger.info("=" * 60)
    logger.info("PHASE 2: Generating Noise Variants (Zero LLM Cost)")
    logger.info("=" * 60)

    if not anchors_file.exists():
        logger.error("Phase 1 must complete first! anchors.jsonl not found.")
        return

    # Load anchors to get neutral texts
    logger.info(f"Loading anchors from {anchors_file}")
    anchors = {}
    with open(anchors_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            if 'source_idx' in entry and 'neutral' in entry:
                anchors[entry['source_idx']] = entry['neutral']

    logger.info(f"Loaded {len(anchors)} anchors with neutral text")

    # Select even indices (50% of chunks)
    even_indices = [i for i in range(0, len(chunks), 2) if i in anchors]
    logger.info(f"Even indices to process: {len(even_indices)}")
    logger.info(f"Output: {output_file}")
    logger.info("-" * 60)

    progress = ProgressTracker(len(even_indices), "Phase 2")

    with open(output_file, 'w') as f:
        for idx in even_indices:
            chunk = chunks[idx]
            neutral = anchors[idx]

            # Create Noise variant: same output, heavily perturbed input
            example = format_example(
                neutral=neutral,
                styled=chunk['text'],
                author=author,
                variation_type="noise"
            )
            example['source_idx'] = idx
            f.write(json.dumps(example) + '\n')

            progress.update(success=True)
            progress.log()

    progress.finish()

    # Print next step
    if chunks_path:
        print_next_step(2, chunks_path, author, output_dir, workers)


# =============================================================================
# Phase 3: Snowflake Variants (LLM Topic Swap)
# =============================================================================

def run_phase3(chunks: list, author: str, output_dir: Path, workers: int = 4, chunks_path: str = None):
    """Phase 3: Generate Snowflake variants for odd-indexed chunks (LLM calls).

    Args:
        chunks: List of chunk dicts
        author: Author name
        output_dir: Output directory
        workers: Number of concurrent API workers
        chunks_path: Path to chunks file (for printing next step)
    """
    import requests
    from concurrent.futures import ThreadPoolExecutor, as_completed

    anchors_file = output_dir / "anchors.jsonl"
    output_file = output_dir / "snowflakes.jsonl"

    logger.info("=" * 60)
    logger.info("PHASE 3: Generating Snowflake Variants (LLM Topic Swap)")
    logger.info("=" * 60)

    if not anchors_file.exists():
        logger.error("Phase 1 must complete first! anchors.jsonl not found.")
        return

    # Load anchors
    logger.info(f"Loading anchors from {anchors_file}")
    anchors = {}
    with open(anchors_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            if 'source_idx' in entry:
                anchors[entry['source_idx']] = entry

    logger.info(f"Loaded {len(anchors)} anchors")

    # Select odd indices (50% of chunks)
    odd_indices = [i for i in range(1, len(chunks), 2) if i in anchors]

    # Check for resume
    processed = set()
    if output_file.exists():
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if 'source_idx' in entry:
                        processed.add(entry['source_idx'])
                except:
                    pass
        if processed:
            logger.info(f"Resuming from checkpoint: {len(processed)} already processed")

    pending = [i for i in odd_indices if i not in processed]

    logger.info(f"Odd indices to process: {len(pending)}")
    logger.info(f"Workers: {workers}")
    logger.info(f"Output: {output_file}")

    if not pending:
        logger.info("All Snowflakes already generated!")
        return

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        logger.error("DEEPSEEK_API_KEY environment variable not set!")
        return

    logger.info("-" * 60)

    def create_snowflake(idx: int) -> tuple:
        """Generate a Snowflake (topic swap) for one chunk."""
        chunk = chunks[idx]
        topic = random.choice(MUNDANE_TOPICS)

        system = f"""You are a literary style transfer assistant specializing in {author}'s writing style.
Rewrite the given passage to be about a mundane everyday topic while preserving:
- The EXACT sentence structure (same number of sentences, same clause patterns)
- The author's characteristic rhythm and cadence
- Similar vocabulary complexity and word choices"""

        prompt = f"""Rewrite this passage to be about "{topic}".

Original:
{chunk['text']}

Requirements:
1. The new passage must be ENTIRELY about "{topic}"
2. Match the word count closely (~{len(chunk['text'].split())} words)
3. Keep the same sentence structure and rhythm

Output only the rewritten passage."""

        try:
            response = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 512
                },
                timeout=60
            )
            response.raise_for_status()
            snowflake_text = response.json()["choices"][0]["message"]["content"].strip()
            snowflake_text = snowflake_text.strip('`"\' \n')
            return idx, snowflake_text, topic, None
        except Exception as e:
            return idx, None, topic, str(e)

    progress = ProgressTracker(len(odd_indices), "Phase 3")
    progress.processed = len(processed)
    progress.success = len(processed)

    with open(output_file, 'a') as f:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(create_snowflake, idx): idx for idx in pending}

            for future in as_completed(futures):
                idx, snowflake_text, topic, error = future.result()

                if snowflake_text:
                    # Get the neutral from anchor
                    anchor = anchors[idx]
                    neutral = anchor.get('neutral', '')

                    # Create training example with snowflake as output
                    example = format_example(
                        neutral=neutral,  # Same neutral input
                        styled=snowflake_text,  # But snowflake output
                        author=author,
                        variation_type="snowflake"
                    )
                    example['source_idx'] = idx
                    example['topic'] = topic
                    f.write(json.dumps(example) + '\n')
                    f.flush()
                    progress.update(success=True)
                else:
                    if error:
                        logger.debug(f"Snowflake {idx} failed: {error}")
                    progress.update(success=False)

                progress.log()

    progress.finish()

    # Print next step
    if chunks_path:
        print_next_step(3, chunks_path, author, output_dir, workers)


# =============================================================================
# Merge and Split
# =============================================================================

def merge_and_split(output_dir: Path):
    """Merge all phases and create train/valid/test splits."""
    logger.info("=" * 60)
    logger.info("MERGING PHASES AND CREATING SPLITS")
    logger.info("=" * 60)

    all_examples = []
    files_loaded = []

    for fname in ["anchors.jsonl", "noise.jsonl", "snowflakes.jsonl"]:
        fpath = output_dir / fname
        if fpath.exists():
            count = 0
            with open(fpath, 'r') as f:
                for line in f:
                    if line.strip():
                        ex = json.loads(line)
                        all_examples.append(ex)
                        count += 1
            files_loaded.append(f"{fname}: {count}")
            logger.info(f"Loaded {fname}: {count} examples")

    if not all_examples:
        logger.error("No examples found! Run phases 1-3 first.")
        return

    logger.info("-" * 60)
    logger.info(f"Total examples: {len(all_examples)}")

    # Count by type
    by_type = {}
    for ex in all_examples:
        vtype = ex.get('variation_type', 'unknown')
        by_type[vtype] = by_type.get(vtype, 0) + 1

    for vtype, count in sorted(by_type.items()):
        pct = count * 100 // len(all_examples)
        logger.info(f"  {vtype}: {count} ({pct}%)")

    # Shuffle and split
    random.seed(42)
    random.shuffle(all_examples)

    n = len(all_examples)
    n_train = int(n * 0.8)
    n_valid = int(n * 0.1)

    train = all_examples[:n_train]
    valid = all_examples[n_train:n_train + n_valid]
    test = all_examples[n_train + n_valid:]

    logger.info("-" * 60)
    logger.info("Creating splits (80/10/10):")

    # Write splits (text only for training)
    for name, examples in [("train", train), ("valid", valid), ("test", test)]:
        path = output_dir / f"{name}.jsonl"
        with open(path, 'w') as f:
            for ex in examples:
                f.write(json.dumps({"text": ex["text"]}) + '\n')
        logger.info(f"  {name}.jsonl: {len(examples)} examples")

    # Also save all.jsonl with metadata
    all_path = output_dir / "all.jsonl"
    with open(all_path, 'w') as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + '\n')
    logger.info(f"  all.jsonl: {len(all_examples)} examples (with metadata)")

    # Summary
    logger.info("=" * 60)
    logger.info("MERGE COMPLETE")
    logger.info("=" * 60)

    # Print training command
    logger.info("")
    logger.info("NEXT STEP: Run LoRA training")
    logger.info("")
    logger.info("  mlx_lm.lora --config lora_adapters/<author>/config.yaml")
    logger.info("")
    logger.info("Make sure config.yaml has:")
    logger.info(f"  data: \"{output_dir}\"")
    logger.info("  keys:")
    logger.info("    - \"self_attn.q_proj\"")
    logger.info("    - \"self_attn.k_proj\"")
    logger.info("    - \"self_attn.v_proj\"")
    logger.info("    - \"self_attn.o_proj\"")
    logger.info("    - \"mlp.gate_proj\"")
    logger.info("    - \"mlp.up_proj\"")
    logger.info("    - \"mlp.down_proj\"")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate training data using 50/50 Split Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Phase 1: RTT neutralization (slowest, run first)
  python %(prog)s --chunks chunks.json --author "H.P. Lovecraft" --output out/ --phase 1 --workers 4

  # Phase 2: Noise variants (instant, no LLM)
  python %(prog)s --chunks chunks.json --author "H.P. Lovecraft" --output out/ --phase 2

  # Phase 3: Snowflake topic swaps (parallel LLM)
  python %(prog)s --chunks chunks.json --author "H.P. Lovecraft" --output out/ --phase 3 --workers 8

  # Merge all into train/valid/test
  python %(prog)s --chunks chunks.json --author "H.P. Lovecraft" --output out/ --phase 1 --merge
        """
    )
    parser.add_argument("--chunks", required=True, help="Path to chunks JSON file")
    parser.add_argument("--author", required=True, help="Author name")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], required=True,
                        help="Phase to run: 1=Anchors (RTT), 2=Noise (script), 3=Snowflakes (LLM)")
    parser.add_argument("--merge", action="store_true",
                        help="After running phase, merge all and create train/valid/test splits")
    parser.add_argument("--no-monotone", action="store_true",
                        help="Disable monotone flattening in Phase 1")
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent workers for LLM calls (Phase 1 & 3)")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load chunks
    logger.info(f"Loading chunks from {args.chunks}")
    with open(args.chunks, 'r') as f:
        chunks = json.load(f)
    logger.info(f"Loaded {len(chunks)} chunks")
    logger.info("")

    # Run requested phase
    if args.phase == 1:
        run_phase1(
            chunks, args.author, output_dir,
            monotone=not args.no_monotone,
            workers=args.workers,
            chunks_path=args.chunks
        )
    elif args.phase == 2:
        run_phase2(chunks, args.author, output_dir, chunks_path=args.chunks, workers=args.workers)
    elif args.phase == 3:
        run_phase3(chunks, args.author, output_dir, workers=args.workers, chunks_path=args.chunks)

    # Optionally merge
    if args.merge:
        logger.info("")
        merge_and_split(output_dir)


if __name__ == "__main__":
    main()
