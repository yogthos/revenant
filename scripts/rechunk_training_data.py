#!/usr/bin/env python3
"""Re-chunk training data into smaller ~150 word segments.

Takes existing chunks (300-450 words) and splits them into ~150 word chunks
with sentence-boundary awareness and optional overlap.

Usage:
    python scripts/rechunk_training_data.py \
        --input data/training/lovecraft/chunks.json \
        --output data/training/lovecraft/chunks_small.json \
        --target-words 150
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def split_into_sentences(text: str) -> list:
    """Split text into sentences."""
    # Split on sentence endings, keeping the delimiter
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def rechunk_text(text: str, target_words: int = 150, overlap_sentences: int = 1) -> list:
    """Split a long text into smaller chunks with overlap.

    Args:
        text: The text to split
        target_words: Target words per chunk (~150)
        overlap_sentences: Number of sentences to overlap between chunks

    Returns:
        List of smaller chunks
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks = []
    i = 0

    while i < len(sentences):
        # Build a chunk starting at sentence i
        chunk_sentences = []
        word_count = 0
        j = i

        # Add sentences until we reach target words
        while j < len(sentences):
            sent = sentences[j]
            sent_words = len(sent.split())

            # If adding this sentence would exceed target by too much, stop
            # But always include at least 2 sentences
            if word_count + sent_words > target_words * 1.3 and len(chunk_sentences) >= 2:
                break

            chunk_sentences.append(sent)
            word_count += sent_words
            j += 1

            # If we've reached target and hit a good boundary, stop
            if word_count >= target_words:
                break

        if chunk_sentences:
            chunk_text = ' '.join(chunk_sentences)
            # Only include if it has reasonable length
            if word_count >= target_words * 0.5:  # At least 50% of target
                chunks.append(chunk_text)

        # Move forward, accounting for overlap
        sentences_used = len(chunk_sentences)
        step = max(1, sentences_used - overlap_sentences)
        i += step

        # If we're near the end and would create a tiny chunk, stop
        remaining_sentences = sentences[i:]
        if remaining_sentences:
            remaining_words = sum(len(s.split()) for s in remaining_sentences)
            if remaining_words < target_words * 0.4:
                break

    return chunks


def process_chunks(input_path: Path, output_path: Path, target_words: int = 150, overlap: int = 1):
    """Process chunks file and create smaller chunks.

    Args:
        input_path: Path to input chunks.json
        output_path: Path to output chunks_small.json
        target_words: Target words per chunk
        overlap: Sentence overlap between chunks
    """
    print(f"Loading chunks from {input_path}")
    with open(input_path, 'r') as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks")

    # Analyze current chunk sizes
    word_counts = [len(c['text'].split()) for c in chunks]
    avg_words = sum(word_counts) / len(word_counts)
    print(f"Current avg words per chunk: {avg_words:.0f}")
    print(f"Target words per chunk: {target_words}")

    # Process by variation type
    new_chunks = []
    stats = {'original': 0, 'snowflake': 0, 'robustness': 0, 'other': 0}

    for chunk in chunks:
        text = chunk['text']
        vtype = chunk.get('variation_type', 'original')
        word_count = len(text.split())

        # If chunk is already small enough, keep as-is
        if word_count <= target_words * 1.2:
            new_chunks.append(chunk)
            stats[vtype if vtype in stats else 'other'] += 1
            continue

        # For originals: use overlap (continuous narrative)
        # For variations: no overlap (each is independent topic)
        if vtype == 'original':
            sub_chunks = rechunk_text(text, target_words=target_words, overlap_sentences=overlap)
        else:
            sub_chunks = rechunk_text(text, target_words=target_words, overlap_sentences=0)

        for sub_text in sub_chunks:
            new_chunks.append({
                'text': sub_text,
                'variation_type': vtype
            })
            stats[vtype if vtype in stats else 'other'] += 1

    # Calculate new stats
    new_word_counts = [len(c['text'].split()) for c in new_chunks]
    new_avg = sum(new_word_counts) / len(new_word_counts) if new_word_counts else 0

    print(f"\nResults:")
    print(f"  Original chunks: {len(chunks)}")
    print(f"  New chunks: {len(new_chunks)} ({len(new_chunks)/len(chunks):.1f}x)")
    print(f"  New avg words: {new_avg:.0f}")
    print(f"  By type: {stats}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(new_chunks, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {output_path}")

    return new_chunks


def main():
    parser = argparse.ArgumentParser(description="Re-chunk training data into smaller segments")
    parser.add_argument("--input", required=True, help="Input chunks.json file")
    parser.add_argument("--output", required=True, help="Output chunks file")
    parser.add_argument("--target-words", type=int, default=150, help="Target words per chunk")
    parser.add_argument("--overlap", type=int, default=1, help="Sentence overlap for originals")

    args = parser.parse_args()

    process_chunks(
        Path(args.input),
        Path(args.output),
        target_words=args.target_words,
        overlap=args.overlap
    )


if __name__ == "__main__":
    main()
