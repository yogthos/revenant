#!/usr/bin/env python3
"""Select diverse, high-quality paragraphs from corpus for training.

Strategy:
1. Score each paragraph for stylistic quality (vocabulary richness, syntax complexity)
2. Cluster paragraphs by topic/content to ensure diversity
3. Select top paragraphs within token budget, ensuring cluster coverage

Usage:
    python scripts/select_diverse_paragraphs.py \
        --corpus data/corpus/lovecraft.txt \
        --output data/training/lovecraft_selected.json \
        --target-tokens 650000
"""

import argparse
import json
import logging
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ScoredParagraph:
    """A paragraph with quality scores."""
    text: str
    tokens: int
    quality_score: float
    vocabulary_richness: float
    syntax_complexity: float
    style_markers: float
    cluster_id: int = -1


def count_tokens(text: str) -> int:
    """Estimate token count (roughly 1.3 tokens per word for English)."""
    words = len(text.split())
    return int(words * 1.3)


def calculate_vocabulary_richness(text: str) -> float:
    """Calculate type-token ratio and rare word density.

    Returns score 0-1 where higher = richer vocabulary.
    """
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    if len(words) < 10:
        return 0.0

    # Type-token ratio (unique words / total words)
    unique_words = set(words)
    ttr = len(unique_words) / len(words)

    # Rare word bonus (words > 8 chars that aren't common)
    common_words = {
        'the', 'and', 'that', 'have', 'for', 'not', 'with', 'you', 'this',
        'but', 'his', 'from', 'they', 'were', 'been', 'have', 'their',
        'would', 'there', 'which', 'could', 'other', 'into', 'more',
        'about', 'through', 'between', 'because', 'something', 'everything',
        'anything', 'nothing', 'however', 'although', 'therefore', 'whatever'
    }
    long_rare = [w for w in unique_words if len(w) > 8 and w not in common_words]
    rare_ratio = len(long_rare) / max(len(unique_words), 1)

    # Combined score
    return min(1.0, (ttr * 0.6) + (rare_ratio * 0.4) + 0.2)


def calculate_syntax_complexity(text: str) -> float:
    """Measure syntactic complexity via sentence structure.

    Returns score 0-1 where higher = more complex syntax.
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return 0.0

    # Sentence length variance (Lovecraft has both short punchy and long flowing)
    lengths = [len(s.split()) for s in sentences]
    if len(lengths) < 2:
        variance_score = 0.3
    else:
        mean_len = sum(lengths) / len(lengths)
        variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        # Normalize: variance of 100 = good, 400+ = excellent
        variance_score = min(1.0, math.sqrt(variance) / 20)

    # Punctuation complexity (semicolons, em-dashes, parentheses)
    complex_punct = len(re.findall(r'[;:—–\(\)]', text))
    punct_score = min(1.0, complex_punct / 10)

    # Clause markers (subordinate clauses indicate complex structure)
    clause_markers = len(re.findall(
        r'\b(which|whom|whose|where|when|while|although|though|because|since|'
        r'unless|until|whereas|whereby|wherein|whereupon)\b',
        text.lower()
    ))
    clause_score = min(1.0, clause_markers / 5)

    return (variance_score * 0.3) + (punct_score * 0.3) + (clause_score * 0.4)


def calculate_style_markers(text: str) -> float:
    """Detect Lovecraftian style markers.

    Returns score 0-1 where higher = more stylistic.
    """
    text_lower = text.lower()

    # Lovecraftian vocabulary (cosmic horror, archaisms)
    lovecraft_words = {
        'eldritch', 'cyclopean', 'non-euclidean', 'blasphemous', 'nameless',
        'unnameable', 'gibbous', 'squamous', 'rugose', 'foetid', 'antediluvian',
        'charnel', 'daemon', 'daemonic', 'accursed', 'hellish', 'loathsome',
        'hideous', 'abhorrent', 'grotesque', 'monstrous', 'abysmal', 'stygian',
        'phantasmal', 'spectral', 'preternatural', 'abnormal', 'unnatural',
        'indescribable', 'unutterable', 'ineffable', 'cosmic', 'immemorial',
        'aeons', 'eons', 'primordial', 'prehistoric', 'archaic', 'ancient',
        'forgotten', 'forbidden', 'accursed', 'dreaded', 'feared', 'terrible',
        'horror', 'terror', 'dread', 'fear', 'madness', 'insanity', 'nightmare',
        'betwixt', 'whilst', 'amidst', 'amongst', 'wherefore', 'hitherto',
        'thence', 'hence', 'thereof', 'wherein', 'whereby'
    }
    found_words = sum(1 for w in lovecraft_words if w in text_lower)
    vocab_score = min(1.0, found_words / 5)

    # Atmospheric phrases
    atmosphere_patterns = [
        r'beyond the .{1,20} of',
        r'in the .{1,20} depths',
        r'from .{1,20} aeons',
        r'the .{1,20} horror',
        r'nameless .{1,20}',
        r'unutterable .{1,20}',
        r'cosmic .{1,20}',
    ]
    atmosphere_matches = sum(1 for p in atmosphere_patterns if re.search(p, text_lower))
    atmosphere_score = min(1.0, atmosphere_matches / 3)

    # First-person narrative markers (common in Lovecraft)
    first_person = len(re.findall(r'\b(I|my|me|myself)\b', text))
    first_person_score = min(1.0, first_person / 10) if first_person > 0 else 0.3

    # Exclamatory/emphatic constructions
    emphatic = len(re.findall(r'!|—|\.{3}', text))
    emphatic_score = min(1.0, emphatic / 5)

    return (vocab_score * 0.4) + (atmosphere_score * 0.3) + (first_person_score * 0.15) + (emphatic_score * 0.15)


def score_paragraph(text: str) -> ScoredParagraph:
    """Score a paragraph for quality."""
    tokens = count_tokens(text)
    vocab = calculate_vocabulary_richness(text)
    syntax = calculate_syntax_complexity(text)
    style = calculate_style_markers(text)

    # Combined quality score (weighted average)
    quality = (vocab * 0.3) + (syntax * 0.35) + (style * 0.35)

    return ScoredParagraph(
        text=text,
        tokens=tokens,
        quality_score=quality,
        vocabulary_richness=vocab,
        syntax_complexity=syntax,
        style_markers=style,
    )


def extract_paragraphs(corpus_path: Path, min_words: int = 80, max_words: int = 600) -> List[str]:
    """Extract paragraphs from corpus file."""
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split on double newlines (paragraph boundaries)
    paragraphs = re.split(r'\n\s*\n', text)

    # Clean and filter
    result = []
    for para in paragraphs:
        para = para.strip()
        para = re.sub(r'\s+', ' ', para)  # Normalize whitespace

        words = len(para.split())
        if words < min_words or words > max_words:
            continue

        # Skip chapter headings, titles, etc.
        if re.match(r'^(Chapter|Part|Section|[IVXLC]+\.?\s*$|[A-Z][A-Z\s]+$)', para):
            continue

        # Skip if too many special characters
        special_ratio = len(re.findall(r'[^a-zA-Z0-9\s.,;:!?\'"()\-—]', para)) / len(para)
        if special_ratio > 0.05:
            continue

        result.append(para)

    return result


def cluster_by_content(paragraphs: List[ScoredParagraph], n_clusters: int = 50) -> List[ScoredParagraph]:
    """Assign paragraphs to content clusters using signature words.

    Uses a hash-based approach: extract top distinctive words and hash them
    to create stable cluster assignments.
    """
    import hashlib

    # Common words to ignore
    stopwords = {
        'that', 'this', 'with', 'from', 'have', 'were', 'been', 'would',
        'could', 'should', 'their', 'there', 'which', 'about', 'into',
        'more', 'some', 'only', 'other', 'such', 'than', 'then', 'when',
        'what', 'where', 'very', 'just', 'even', 'most', 'also', 'over',
        'they', 'them', 'these', 'those', 'will', 'would', 'could', 'might',
        'must', 'shall', 'being', 'having', 'doing', 'made', 'said', 'came',
        'went', 'told', 'knew', 'thought', 'seemed', 'found', 'gave', 'took',
        'like', 'upon', 'before', 'after', 'through', 'between', 'under',
        'above', 'below', 'without', 'within', 'around', 'against', 'along',
        'across', 'behind', 'beyond', 'during', 'until', 'while', 'never',
        'always', 'often', 'still', 'already', 'ever', 'again', 'back',
        'down', 'away', 'here', 'now', 'then', 'well', 'much', 'many',
        'each', 'every', 'both', 'itself', 'himself', 'myself', 'themselves',
    }

    def get_signature(text: str) -> str:
        """Get a content signature from distinctive nouns/entities."""
        words = re.findall(r'\b[a-zA-Z]{5,}\b', text.lower())
        # Filter stopwords and get word frequencies
        word_counts = Counter(w for w in words if w not in stopwords)
        # Take top 5 most frequent distinctive words
        top_words = sorted(word_counts.keys(), key=lambda w: (-word_counts[w], w))[:5]
        return ' '.join(sorted(top_words))

    # Assign clusters based on signature hash
    for para in paragraphs:
        sig = get_signature(para.text)
        # Hash to cluster ID
        hash_val = int(hashlib.md5(sig.encode()).hexdigest()[:8], 16)
        para.cluster_id = hash_val % n_clusters

    cluster_counts = Counter(p.cluster_id for p in paragraphs)
    logger.info(f"Created {len(cluster_counts)} content clusters "
                f"(sizes: {min(cluster_counts.values())}-{max(cluster_counts.values())})")

    return paragraphs


def select_diverse_paragraphs(
    paragraphs: List[ScoredParagraph],
    target_tokens: int,
    min_per_cluster: int = 2,
) -> List[ScoredParagraph]:
    """Select diverse, high-quality paragraphs within token budget.

    Strategy:
    1. Ensure minimum representation from each cluster
    2. Fill remaining budget with highest quality paragraphs
    3. Maintain diversity by not over-sampling any cluster
    """
    # Sort paragraphs by quality within each cluster
    clusters = {}
    for para in paragraphs:
        if para.cluster_id not in clusters:
            clusters[para.cluster_id] = []
        clusters[para.cluster_id].append(para)

    for c_id in clusters:
        clusters[c_id].sort(key=lambda p: p.quality_score, reverse=True)

    selected = []
    total_tokens = 0
    cluster_counts = Counter()

    # Phase 1: Select minimum from each cluster (best quality first)
    for c_id, paras in clusters.items():
        for para in paras[:min_per_cluster]:
            if total_tokens + para.tokens <= target_tokens:
                selected.append(para)
                total_tokens += para.tokens
                cluster_counts[c_id] += 1

    logger.info(f"Phase 1: Selected {len(selected)} paragraphs ({total_tokens} tokens) for minimum cluster coverage")

    # Phase 2: Fill remaining budget with highest quality, maintaining diversity
    remaining = []
    for c_id, paras in clusters.items():
        remaining.extend(paras[min_per_cluster:])

    remaining.sort(key=lambda p: p.quality_score, reverse=True)

    # Calculate max per cluster to maintain diversity
    max_per_cluster = max(10, len(paragraphs) // len(clusters) * 2)

    for para in remaining:
        if total_tokens >= target_tokens:
            break
        if cluster_counts[para.cluster_id] >= max_per_cluster:
            continue  # Skip over-represented clusters
        if total_tokens + para.tokens <= target_tokens * 1.05:  # 5% tolerance
            selected.append(para)
            total_tokens += para.tokens
            cluster_counts[para.cluster_id] += 1

    logger.info(f"Phase 2: Selected {len(selected)} total paragraphs ({total_tokens} tokens)")
    logger.info(f"Cluster distribution: min={min(cluster_counts.values())}, max={max(cluster_counts.values())}, "
                f"mean={sum(cluster_counts.values())/len(cluster_counts):.1f}")

    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Select diverse, high-quality paragraphs for training"
    )
    parser.add_argument("--corpus", required=True, help="Path to corpus file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--target-tokens", type=int, default=650000,
                        help="Target token count (default: 650000)")
    parser.add_argument("--min-words", type=int, default=80,
                        help="Minimum words per paragraph")
    parser.add_argument("--max-words", type=int, default=600,
                        help="Maximum words per paragraph")
    parser.add_argument("--n-clusters", type=int, default=50,
                        help="Number of content clusters for diversity")
    parser.add_argument("--min-quality", type=float, default=0.0,
                        help="Minimum quality score (0-1) to include paragraph")

    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    output_path = Path(args.output)

    # Step 1: Extract paragraphs
    logger.info(f"Extracting paragraphs from {corpus_path}...")
    paragraphs = extract_paragraphs(corpus_path, args.min_words, args.max_words)
    logger.info(f"Found {len(paragraphs)} candidate paragraphs")

    # Step 2: Score each paragraph
    logger.info("Scoring paragraphs for quality...")
    scored = [score_paragraph(p) for p in paragraphs]

    total_available_tokens = sum(p.tokens for p in scored)
    logger.info(f"Total available: {total_available_tokens:,} tokens")

    # Filter by minimum quality if specified
    if args.min_quality > 0:
        before = len(scored)
        scored = [p for p in scored if p.quality_score >= args.min_quality]
        logger.info(f"Quality filter (>={args.min_quality}): {before} -> {len(scored)} paragraphs")

    # Quality distribution
    scores = [p.quality_score for p in scored]
    logger.info(f"Quality scores: min={min(scores):.2f}, max={max(scores):.2f}, "
                f"mean={sum(scores)/len(scores):.2f}")

    # Step 3: Cluster for diversity
    logger.info("Clustering paragraphs by content...")
    scored = cluster_by_content(scored, args.n_clusters)

    # Step 4: Select diverse high-quality paragraphs
    logger.info(f"Selecting paragraphs for {args.target_tokens:,} token target...")
    selected = select_diverse_paragraphs(scored, args.target_tokens)

    # Step 5: Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "metadata": {
            "source": str(corpus_path),
            "total_paragraphs": len(paragraphs),
            "selected_paragraphs": len(selected),
            "target_tokens": args.target_tokens,
            "actual_tokens": sum(p.tokens for p in selected),
            "mean_quality": sum(p.quality_score for p in selected) / len(selected),
        },
        "paragraphs": [
            {
                "text": p.text,
                "tokens": p.tokens,
                "quality_score": round(p.quality_score, 3),
                "vocabulary_richness": round(p.vocabulary_richness, 3),
                "syntax_complexity": round(p.syntax_complexity, 3),
                "style_markers": round(p.style_markers, 3),
                "cluster_id": p.cluster_id,
            }
            for p in selected
        ]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(selected)} paragraphs to {output_path}")

    # Summary stats
    print("\n" + "="*60)
    print("SELECTION SUMMARY")
    print("="*60)
    print(f"Selected: {len(selected)} paragraphs")
    print(f"Tokens: {sum(p.tokens for p in selected):,}")
    print(f"Mean quality: {sum(p.quality_score for p in selected)/len(selected):.3f}")
    print(f"Clusters represented: {len(set(p.cluster_id for p in selected))}")
    print()
    print("With 2 variations each:")
    print(f"  Total examples: {len(selected) * 3:,}")
    print(f"  Total tokens: ~{sum(p.tokens for p in selected) * 3:,}")
    print("="*60)


if __name__ == "__main__":
    main()
