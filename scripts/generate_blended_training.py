#!/usr/bin/env python3
"""Generate training data for a blended author style (e.g., "Howard Russell").

Takes blended paragraphs produced by experiment_blend_sentences.py and feeds them
through the same training pipeline as generate_flat_training.py: overlapping chunks,
RTT neutralization, snowflake topic variations, many-to-one input variants, persona
frames, perturbation.

Pipeline:
1. Load blended paragraphs (from JSON)
2. Create overlapping chunks (style lives in transitions)
3. Optionally generate snowflake topic variations via LLM
4. Feed all chunks through generate_flat_training.py's generate_training_data()
   which handles: RTT, many-to-one variants, persona frames, perturbation, output

Usage:
    # Full pipeline (overlapping chunks + RTT + snowflakes) — run on server
    python scripts/generate_blended_training.py \
        --blended data/blended/howard_russell_corpus_full.json \
        --author "Howard Russell" \
        --output data/training/howard_russell/LlamaFactory

    # Skip snowflakes (faster, ~2x fewer examples)
    python scripts/generate_blended_training.py \
        --blended data/blended/howard_russell_corpus_full.json \
        --author "Howard Russell" \
        --output data/training/howard_russell/LlamaFactory \
        --skip-snowflakes
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import List, Tuple

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Import the real training pipeline
from scripts.generate_flat_training import (
    MUNDANE_TOPICS,
    OverlapConfig,
    PERSONA_FRAMES,
    create_overlapping_chunks,
    create_topic_variation,
    generate_training_data,
    load_intermediate,
    save_intermediate,
)


# =============================================================================
# Custom Snowflake Topics (matched to the book's actual content domains)
# =============================================================================

# These topics match what the LoRA will encounter at inference (chapters 5-11).
# Mixed with MUNDANE_TOPICS to maintain both topic-independence and domain-relevance.
BOOK_TOPICS = [
    # Neuroscience & consciousness (Chapter 5)
    "how neurons form memories through synaptic strengthening",
    "the brain's prediction engine and its role in perception",
    "why consciousness might be an emergent property of information processing",
    "the recursive nature of self-awareness in biological systems",
    "how homeostasis drives all voluntary action",
    "the simulation engine inside every animal brain",
    "why the hard problem of consciousness resists material explanation",
    "how the brain constructs a model of itself",
    # Communication & theory of mind (Chapter 6)
    "how organisms evolved to communicate at a distance",
    "the energy economics of cooperation versus competition",
    "how theory of mind enables both empathy and manipulation",
    "why language activates mental models rather than transmitting meaning directly",
    "the paradox of shared understanding through ambiguous symbols",
    "how deception became an evolutionary catalyst for intelligence",
    # Memes & cultural evolution (Chapter 7)
    "how ideas replicate and mutate like genes in a population",
    "substrate independence and why patterns matter more than material",
    "the computational universality of neural systems",
    "how cultural selection operates on memes through social networks",
    "the cost of integrating new ideas into an existing worldview",
    "why information patterns are the fundamental unit of reality",
    # Dialectics & social systems (Chapter 8)
    "how contradictions in a system drive its transformation",
    "why material conditions determine the shape of ideas and institutions",
    "the dialectical relationship between technology and social organization",
    "how societies function as metaorganisms with emergent properties",
    "why unrestrained markets produce metabolic dysfunction in the social body",
    "the tension between individual freedom and collective coordination",
    "how the mode of production shapes consciousness",
    # AI & computation (Chapter 9)
    "why artificial minds need embodied experience to achieve understanding",
    "the problem of creating meaningful scarcity in a virtual environment",
    "how shared context between humans and machines enables safe AI",
    "why genetic algorithms mirror biological evolution in digital substrates",
    "the challenge of grounding language in sensory reality",
    "whether artificial consciousness requires subjective experience",
    # Cosmic perspective (Chapter 10)
    "why post-biological intelligence may dominate the cosmos",
    "the narrow window during which biological and digital minds can communicate",
    "how exponential technological change compresses centuries into decades",
    "why the Fermi paradox may be explained by the nature of intelligence itself",
    "the transition from biological to synthetic minds as an evolutionary inevitability",
    # General analytical/philosophical (cross-cutting)
    "how fractal patterns repeat across scales from cells to civilizations",
    "the distinction between what can be observed and what can be inferred",
    "why reductionism fails to capture emergent properties",
    "how selection pressure operates on any self-replicating pattern",
    "the relationship between entropy and the emergence of complexity",
]


def load_persona_frames(worldview_path: Path) -> dict:
    """Load persona frames from a worldview file into PERSONA_FRAMES dict.

    For blended authors, we merge narrative and conceptual frames into BOTH
    content type keys. This bypasses the content classifier's narrative bias
    (which tags Russell's analytical text as narrative due to past-tense verbs
    and person entities like "Kant", "Plato"). Every Howard Russell frame was
    designed to blend analytical rigor with Lovecraftian dread — they all work
    on both narrative and conceptual content.

    Observed bug in previous runs: classifier put 88.5% of examples in narrative
    frames and only 11.5% in conceptual frames, despite most content being
    analytical philosophy. Merging ensures uniform access to all 15 frames.
    """
    text = worldview_path.read_text(encoding="utf-8")

    narrative_frames = []
    conceptual_frames = []
    current_section = None

    for line in text.split("\n"):
        line = line.strip()
        if line == "[PERSONA_FRAMES_NARRATIVE]":
            current_section = "narrative"
        elif line == "[PERSONA_FRAMES_CONCEPTUAL]":
            current_section = "conceptual"
        elif line == "---":
            continue
        elif line and current_section:
            if current_section == "narrative":
                narrative_frames.append(line)
            else:
                conceptual_frames.append(line)

    # Merge: both content type keys get all frames
    all_frames = narrative_frames + conceptual_frames
    frames = {"narrative": all_frames, "conceptual": all_frames}

    logger.info(f"Loaded {len(narrative_frames)} narrative + "
                f"{len(conceptual_frames)} conceptual frames "
                f"(merged into unified pool of {len(all_frames)} for both content types)")
    return frames


def generate_snowflakes(
    paragraphs: List[Tuple[str, Tuple[int, ...]]],
    author: str,
    topics: List[str],
    max_snowflakes: int = 0,
    workers: int = 4,
) -> List[Tuple[str, str, Tuple[int, ...]]]:
    """Generate snowflake topic variations for blended paragraphs.

    Uses a mix of book-specific topics and mundane topics. Takes
    (text, source_paragraphs) pairs and returns
    (varied_paragraph, "snowflake", source_paragraphs) tuples.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Interleave book topics and mundane topics (60/40 split)
    all_topics = list(topics)
    random.shuffle(all_topics)
    mundane = list(MUNDANE_TOPICS)
    random.shuffle(mundane)

    topic_pool = []
    bi, mi = 0, 0
    while bi < len(all_topics) or mi < len(mundane):
        if bi < len(all_topics) and (random.random() < 0.6 or mi >= len(mundane)):
            topic_pool.append(all_topics[bi])
            bi += 1
        elif mi < len(mundane):
            topic_pool.append(mundane[mi])
            mi += 1

    # Cycle through topics
    import itertools
    topic_iter = itertools.cycle(topic_pool)

    tasks = []
    for para, src in paragraphs:
        if max_snowflakes and len(tasks) >= max_snowflakes:
            break
        tasks.append((para, src, next(topic_iter)))

    logger.info(f"Generating {len(tasks)} snowflake variations ({workers} workers)...")
    logger.info(f"  Topic pool: {len(topics)} book topics + {len(MUNDANE_TOPICS)} mundane topics")
    results = []

    def _make_variation(args):
        para, src, topic = args
        return create_topic_variation(para, author, topic), src

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_make_variation, t): i for i, t in enumerate(tasks)}
        done = 0
        for future in as_completed(futures):
            done += 1
            if done % 50 == 0:
                logger.info(f"  Snowflakes: {done}/{len(tasks)} ({len(results)} ok)")
            result, src = future.result()
            if result:
                results.append((result, "snowflake", src))

    logger.info(f"  Generated {len(results)}/{len(tasks)} snowflake variations")
    return results


def build_chunks(args, styled_paragraphs: List[str]) -> list:
    """Chunk, add snowflakes and robustness rows, shuffle."""
    # =========================================================================
    # Step 1: Word-based overlapping chunks (matches proven Lovecraft pipeline)
    #
    # Uses create_overlapping_chunks() from generate_flat_training.py — the SAME
    # function that produced the working Lovecraft adapter. Key properties:
    #   - Chunks of 150-400 words (200-530 tokens), capturing multi-sentence arcs
    #   - 2-sentence overlap at chunk boundaries (moderate, ~1.2× exposure)
    #   - Cross-paragraph spans: treats the corpus as a flat sentence stream
    #
    # Why cross-paragraph spans are OK for a blended corpus:
    #   - Each blended paragraph is a self-contained Russell+Lovecraft unit
    #   - Chunks spanning boundaries teach style invariance across topics
    #     (same lesson snowflakes teach, extended to within-sequence variation)
    #   - The Lovecraft adapter was trained this way and works in production
    #   - The base model's attention handles topic boundaries; the LoRA only
    #     modulates style
    # =========================================================================
    overlap_config = OverlapConfig(
        min_words=args.min_chunk_words,
        max_words=args.max_chunk_words,
        overlap_sentences=args.overlap_sentences,
    )

    raw_entries = [(p, "original", (i,)) for i, p in enumerate(styled_paragraphs)]
    chunks = create_overlapping_chunks(raw_entries, overlap_config)
    logger.info(f"Chunking: {len(styled_paragraphs)} blended paragraphs → {len(chunks)} chunks "
                f"(min {args.min_chunk_words}w, max {args.max_chunk_words}w, "
                f"{args.overlap_sentences}-sentence overlap)")

    # =========================================================================
    # Step 2: Generate snowflake topic variations
    # Uses book-specific topics (60%) + mundane topics (40%)
    # =========================================================================
    if not args.skip_snowflakes:
        # Extract just the text from original-type chunks for snowflake generation
        originals = [(text, src) for text, vtype, src in chunks if vtype == "original"]
        snowflakes = generate_snowflakes(
            originals, args.author,
            topics=BOOK_TOPICS,
            workers=args.snowflake_workers,
        )
        chunks.extend(snowflakes)
        logger.info(f"After snowflakes: {len(chunks)} total chunks")
    else:
        logger.info(f"Snowflakes skipped. Total chunks: {len(chunks)}")

    # =========================================================================
    # Step 3: Add robustness entries (heavy perturbation variants)
    # =========================================================================
    originals = [(text, src) for text, vtype, src in chunks if vtype == "original"]
    n_robustness = len(originals) // 3
    robustness = [(p, "robustness", src) for p, src in random.sample(originals, n_robustness)]
    chunks.extend(robustness)
    logger.info(f"  + {len(robustness)} robustness entries = {len(chunks)} total")

    # Shuffle
    random.shuffle(chunks)

    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Generate blended-style training data using full RTT pipeline"
    )
    parser.add_argument("--blended", type=Path, required=True,
                        help="Path to blended paragraphs JSON")
    parser.add_argument("--author", type=str, default="Howard Russell",
                        help="Blended author name (default: Howard Russell)")
    parser.add_argument("--worldview", type=Path,
                        default=PROJECT_ROOT / "prompts" / "howard_russell_worldview.txt",
                        help="Path to worldview file with persona frames")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory (train.jsonl written here)")
    parser.add_argument("--no-nli", action="store_true",
                        help="Skip the two-way entailment filter when writing LlamaFactory splits")
    parser.add_argument("--format", choices=["llama_factory", "mlx"],
                        default="llama_factory",
                        help="Output format (default: llama_factory)")
    parser.add_argument("--skip-snowflakes", action="store_true",
                        help="Skip snowflake topic variations (faster)")
    parser.add_argument("--snowflake-workers", type=int, default=4,
                        help="Parallel workers for snowflake generation (default: 4)")
    parser.add_argument("--min-chunk-words", type=int, default=40,
                        help="Min words per chunk (default: 40 — chapters p25 = 61w, "
                             "40 captures the short-paragraph tail)")
    parser.add_argument("--max-chunk-words", type=int, default=100,
                        help="Max words per chunk (default: 100 — chapters median 75w, "
                             "p90 111w. Produces chunks averaging ~94w matching inference)")
    parser.add_argument("--overlap-sentences", type=int, default=2,
                        help="Sentences overlapping between adjacent chunks (default: 2)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")

    args = parser.parse_args()

    # Load blended paragraphs
    logger.info(f"Loading blended paragraphs from {args.blended}")
    with open(args.blended, "r", encoding="utf-8") as f:
        blended = json.load(f)
    logger.info(f"  {len(blended)} paragraphs loaded")

    # Load and register persona frames
    logger.info(f"Loading persona frames from {args.worldview}")
    frames = load_persona_frames(args.worldview)
    PERSONA_FRAMES[args.author] = frames

    # Extract styled text from blended paragraphs
    # Use the most processed version: transplanted > aligned > raw
    styled_paragraphs = []
    for para in blended:
        text = para.get("transplanted_text") or para.get("aligned_text") or para["text"]
        styled_paragraphs.append(text)

    total_words = sum(len(p.split()) for p in styled_paragraphs)
    logger.info(f"  {total_words:,} total words across {len(styled_paragraphs)} paragraphs")

    # --resume matches rows to chunks by index, and snowflakes and the shuffle
    # differ every run, so resumed runs reuse the saved chunk list.
    args.output.mkdir(parents=True, exist_ok=True)
    chunks_path = args.output / "chunks.json"
    if args.resume and chunks_path.exists():
        chunks = load_intermediate(chunks_path, stage="chunks")
    else:
        chunks = build_chunks(args, styled_paragraphs)
        save_intermediate(chunks, chunks_path, stage="chunks")

    # =========================================================================
    # Step 4: Generate training data using the real pipeline
    # Handles: RTT neutralization, many-to-one variants (standard + info_dropout
    # + abstract), persona frames, perturbation, lexical bleed filtering
    # =========================================================================
    output_path = args.output / "train.jsonl"
    logger.info(f"\nStarting training data generation → {output_path}")

    n_originals = sum(1 for _, vtype, _ in chunks if vtype == "original")
    n_other = len(chunks) - n_originals
    expected = int(n_originals * 3 * 0.9 + n_other * 0.9)
    logger.info(f"Expected output: ~{expected} training examples")
    logger.info(f"  ({n_originals} originals × 3 variants + {n_other} snowflake/robustness × 1)")

    n_written = generate_training_data(
        chunks=chunks,
        author=args.author,
        output_path=output_path,
        output_format=args.format,
        resume=args.resume,
    )

    logger.info(f"\nDone! {n_written} training examples written to {output_path}")

    if args.format == "llama_factory":
        from scripts.filter_training_data import finalize
        finalize(output_path, args.output / "LlamaFactory", dataset_name=args.output.name,
                 nli=not args.no_nli, log=logger.info)


if __name__ == "__main__":
    main()
