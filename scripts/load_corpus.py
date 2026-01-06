#!/usr/bin/env python3
"""Unified corpus loading script: clean, analyze, and index into ChromaDB.

This script performs a complete corpus ingestion pipeline:
1. Load and split raw corpus into paragraphs
2. Filter for quality (length, encoding, completeness)
3. Optionally deduplicate using semantic similarity
4. Compute style metrics (sentence length, complexity, etc.)
5. Generate embeddings for semantic search
6. Extract rhetorical skeletons via LLM
7. Store everything in ChromaDB

Usage:
    # Basic usage
    python scripts/load_corpus.py --input data/corpus/author.txt --author "Author Name"

    # With all options
    python scripts/load_corpus.py \
        --input data/corpus/author.txt \
        --author "Author Name" \
        --min-words 30 \
        --clear \
        --skip-skeletons \
        -v

    # List indexed authors
    python scripts/load_corpus.py --list
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


@dataclass
class ChunkStats:
    """Statistics from corpus loading."""
    total_paragraphs: int = 0
    filtered_paragraphs: int = 0
    indexed_chunks: int = 0
    skeletons_extracted: int = 0
    rejection_reasons: dict = None

    def __post_init__(self):
        if self.rejection_reasons is None:
            self.rejection_reasons = {}


def is_quality_paragraph(para: str, min_words: int = 30) -> Tuple[bool, str]:
    """Check if paragraph meets quality standards.

    Args:
        para: Paragraph text.
        min_words: Minimum word count.

    Returns:
        Tuple of (is_quality, reason_if_rejected)
    """
    words = para.split()
    word_count = len(words)

    # Too short
    if word_count < min_words:
        return False, f"too_short"

    # OCR/encoding artifacts (allow common Unicode punctuation)
    cleaned = re.sub(r'[\u2014\u2013\u2018\u2019\u201c\u201d\u2026]', '', para)
    if re.search(r'[^\x00-\x7F]{3,}', cleaned):
        return False, "encoding_artifacts"

    # Excessive special characters
    special_ratio = len(re.findall(r'[^a-zA-Z0-9\s.,!?;:\'"()\-\u2014\u2013]', para)) / max(len(para), 1)
    if special_ratio > 0.1:
        return False, "special_chars"

    # Check for complete sentences
    sentences = re.split(r'[.!?]+', para.strip())
    complete_sentences = [s for s in sentences if s.strip() and len(s.split()) >= 3]
    if len(complete_sentences) < 1:
        return False, "incomplete"

    # Excessive repetition
    word_freq = {}
    for w in words:
        w_lower = w.lower().strip('.,!?;:\'"')
        if len(w_lower) > 3:
            word_freq[w_lower] = word_freq.get(w_lower, 0) + 1

    max_freq = max(word_freq.values()) if word_freq else 0
    if max_freq > word_count * 0.15 and max_freq > 5:
        return False, "repetitive"

    return True, "ok"


def split_corpus(text: str) -> List[str]:
    """Split corpus into paragraphs.

    Handles various paragraph separators and normalizes whitespace.
    """
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Split on double newlines (standard paragraph break)
    paragraphs = re.split(r'\n\s*\n', text)

    # Clean and filter
    cleaned = []
    for para in paragraphs:
        # Normalize internal whitespace
        para = ' '.join(para.split())
        if para.strip():
            cleaned.append(para.strip())

    return cleaned


def deduplicate_paragraphs(
    paragraphs: List[str],
    threshold: float = 0.9
) -> Tuple[List[str], int]:
    """Remove near-duplicate paragraphs using semantic similarity.

    Args:
        paragraphs: List of paragraphs.
        threshold: Similarity threshold (0-1). Higher = stricter dedup.

    Returns:
        Tuple of (deduplicated list, number removed).
    """
    if len(paragraphs) < 2:
        return paragraphs, 0

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        logger.info("Computing embeddings for deduplication...")
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Truncate for efficiency
        truncated = [p[:512] for p in paragraphs]
        embeddings = model.encode(truncated, show_progress_bar=False)

        # Find duplicates using cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity

        keep_indices = set(range(len(paragraphs)))

        for i in range(len(paragraphs)):
            if i not in keep_indices:
                continue

            for j in range(i + 1, len(paragraphs)):
                if j not in keep_indices:
                    continue

                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                if sim > threshold:
                    # Keep the longer one
                    if len(paragraphs[j]) > len(paragraphs[i]):
                        keep_indices.discard(i)
                        break
                    else:
                        keep_indices.discard(j)

        deduped = [paragraphs[i] for i in sorted(keep_indices)]
        removed = len(paragraphs) - len(deduped)

        return deduped, removed

    except ImportError:
        logger.warning("sentence-transformers not available, skipping deduplication")
        return paragraphs, 0


def extract_skeletons_batch(
    chunks: List[str],
    llm_provider,
    batch_save_fn=None,
    batch_size: int = 10
) -> List[Optional[str]]:
    """Extract rhetorical skeletons for chunks.

    Args:
        chunks: List of text chunks.
        llm_provider: LLM provider with call() method.
        batch_save_fn: Optional callback to save after each batch.
        batch_size: How often to call batch_save_fn.

    Returns:
        List of skeleton strings (or None if extraction failed).
    """
    from src.rag.skeleton_extractor import extract_skeleton

    skeletons = []

    try:
        from tqdm import tqdm
        iterator = tqdm(enumerate(chunks), total=len(chunks), desc="Extracting skeletons")
    except ImportError:
        iterator = enumerate(chunks)

    for i, chunk in iterator:
        try:
            skeleton = extract_skeleton(chunk, llm_provider)
            if skeleton.moves:
                skeletons.append(skeleton.to_metadata())
            else:
                skeletons.append(None)
        except Exception as e:
            logger.warning(f"Skeleton extraction failed for chunk {i}: {e}")
            skeletons.append(None)

        # Batch save callback
        if batch_save_fn and (i + 1) % batch_size == 0:
            batch_save_fn(i + 1)

    return skeletons


def load_corpus(
    corpus_path: str,
    author: str,
    min_words: int = 30,
    deduplicate: bool = True,
    dedup_threshold: float = 0.9,
    extract_skeletons: bool = True,
    clear_existing: bool = False,
    verbose: bool = False,
) -> ChunkStats:
    """Load corpus into ChromaDB with full metadata.

    Args:
        corpus_path: Path to corpus text file.
        author: Author name.
        min_words: Minimum words per paragraph.
        deduplicate: Whether to remove near-duplicates.
        dedup_threshold: Similarity threshold for deduplication.
        extract_skeletons: Whether to extract rhetorical skeletons.
        clear_existing: Whether to clear existing chunks for this author.
        verbose: Whether to print detailed progress.

    Returns:
        ChunkStats with loading statistics.
    """
    from src.rag.corpus_indexer import get_indexer
    from src.rag.style_analyzer import StyleAnalyzer

    stats = ChunkStats()

    # Load corpus
    path = Path(corpus_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    logger.info(f"Loading corpus from {corpus_path}")
    text = path.read_text(encoding="utf-8")
    source_file = path.name

    # Split into paragraphs
    paragraphs = split_corpus(text)
    stats.total_paragraphs = len(paragraphs)
    logger.info(f"Split into {len(paragraphs)} paragraphs")

    # Quality filtering
    logger.info("Filtering for quality...")
    quality_paragraphs = []

    for para in paragraphs:
        is_quality, reason = is_quality_paragraph(para, min_words)
        if is_quality:
            quality_paragraphs.append(para)
        else:
            stats.rejection_reasons[reason] = stats.rejection_reasons.get(reason, 0) + 1

    stats.filtered_paragraphs = len(quality_paragraphs)
    logger.info(f"After quality filter: {len(quality_paragraphs)} paragraphs")

    if verbose and stats.rejection_reasons:
        print("Rejection reasons:")
        for reason, count in sorted(stats.rejection_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    if not quality_paragraphs:
        logger.error("No quality paragraphs found!")
        return stats

    # Deduplication
    if deduplicate:
        quality_paragraphs, removed = deduplicate_paragraphs(quality_paragraphs, dedup_threshold)
        if removed > 0:
            logger.info(f"Removed {removed} near-duplicates, {len(quality_paragraphs)} remaining")

    # Get indexer and analyzer
    indexer = get_indexer()
    analyzer = StyleAnalyzer()

    # Clear existing if requested
    if clear_existing:
        deleted = indexer._delete_author_chunks(author)
        if deleted > 0:
            logger.info(f"Cleared {deleted} existing chunks for {author}")

    # Generate embeddings
    logger.info("Generating embeddings...")
    embeddings = indexer.embedding_model.encode(quality_paragraphs, show_progress_bar=True)

    # Analyze style metrics
    logger.info("Analyzing style metrics...")
    metrics_list = analyzer.analyze_batch(quality_paragraphs)

    # Extract skeletons if requested
    skeleton_list = [None] * len(quality_paragraphs)
    if extract_skeletons:
        logger.info("Extracting rhetorical skeletons...")
        try:
            from src.config import load_config
            from src.llm.provider import create_critic_provider

            config = load_config()
            llm_provider = create_critic_provider(config.llm)

            skeleton_list = extract_skeletons_batch(quality_paragraphs, llm_provider)
            stats.skeletons_extracted = sum(1 for s in skeleton_list if s)
            logger.info(f"Extracted {stats.skeletons_extracted} skeletons")

        except Exception as e:
            logger.error(f"Skeleton extraction failed: {e}")
            logger.info("Continuing without skeletons...")

    # Prepare data for ChromaDB
    ids = []
    documents = []
    metadatas = []
    embedding_list = []

    for i, (chunk, metrics, embedding, skeleton) in enumerate(
        zip(quality_paragraphs, metrics_list, embeddings, skeleton_list)
    ):
        chunk_id = f"{author}_{source_file}_{i}"
        ids.append(chunk_id)
        documents.append(chunk)

        # Build metadata
        meta = metrics.to_dict()
        meta["author"] = author
        meta["source_file"] = source_file
        meta["chunk_index"] = i
        if skeleton:
            meta["skeleton"] = skeleton

        metadatas.append(meta)
        embedding_list.append(embedding.tolist())

    # Upsert to collection
    logger.info(f"Indexing {len(ids)} chunks into ChromaDB...")
    indexer.collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embedding_list,
    )

    stats.indexed_chunks = len(ids)
    logger.info(f"Successfully indexed {len(ids)} chunks for {author}")

    return stats


def list_indexed_authors():
    """List all authors indexed in ChromaDB."""
    from src.rag.corpus_indexer import get_indexer

    indexer = get_indexer()
    authors = indexer.get_authors()

    if not authors:
        print("No authors indexed yet.")
        return

    print(f"\nIndexed authors ({len(authors)}):")
    print("-" * 50)

    for author in sorted(authors):
        count = indexer.get_chunk_count(author)

        # Check skeleton coverage
        results = indexer.collection.get(
            where={"author": author},
            include=["metadatas"]
        )
        metas = results.get("metadatas", [])
        with_skeleton = sum(1 for m in metas if m.get("skeleton"))

        skeleton_pct = (with_skeleton / count * 100) if count > 0 else 0
        print(f"  {author}: {count} chunks ({with_skeleton} with skeletons, {skeleton_pct:.0f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Load corpus into ChromaDB with cleaning and metadata extraction"
    )

    # Input/output
    parser.add_argument(
        "--input", "-i",
        help="Path to corpus text file"
    )
    parser.add_argument(
        "--author", "-a",
        help="Author name for this corpus"
    )

    # Quality filtering
    parser.add_argument(
        "--min-words",
        type=int,
        default=30,
        help="Minimum words per paragraph (default: 30)"
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Skip deduplication"
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.9,
        help="Similarity threshold for deduplication (default: 0.9)"
    )

    # Skeleton extraction
    parser.add_argument(
        "--skip-skeletons",
        action="store_true",
        help="Skip rhetorical skeleton extraction (faster)"
    )

    # Index management
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing chunks for this author before loading"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all indexed authors and exit"
    )

    # Output
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed progress"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level="DEBUG" if args.verbose else "INFO")

    # List mode
    if args.list:
        list_indexed_authors()
        return 0

    # Validate required args
    if not args.input:
        parser.error("--input is required (or use --list)")
    if not args.author:
        parser.error("--author is required")

    # Load corpus
    try:
        stats = load_corpus(
            corpus_path=args.input,
            author=args.author,
            min_words=args.min_words,
            deduplicate=not args.no_dedup,
            dedup_threshold=args.dedup_threshold,
            extract_skeletons=not args.skip_skeletons,
            clear_existing=args.clear,
            verbose=args.verbose,
        )

        # Print summary
        print("\n" + "=" * 50)
        print("CORPUS LOADING COMPLETE")
        print("=" * 50)
        print(f"Author: {args.author}")
        print(f"Source: {args.input}")
        print(f"Total paragraphs: {stats.total_paragraphs}")
        print(f"After filtering: {stats.filtered_paragraphs}")
        print(f"Indexed chunks: {stats.indexed_chunks}")
        if not args.skip_skeletons:
            print(f"Skeletons extracted: {stats.skeletons_extracted}")

        if stats.rejection_reasons:
            print("\nRejection breakdown:")
            for reason, count in sorted(stats.rejection_reasons.items(), key=lambda x: -x[1]):
                print(f"  {reason}: {count}")

        return 0

    except Exception as e:
        logger.exception(f"Failed to load corpus: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
