#!/usr/bin/env python3
"""Chunk a corpus into ~150 word segments for style transfer training.

Creates overlapping chunks with sentence-boundary awareness.
Style lives in transitions, so we overlap sentences between chunks.

Usage:
    python scripts/chunk_corpus.py \
        --corpus data/corpus/curated/lovecraft.txt \
        --output data/training/lovecraft/chunks_150.json \
        --target-words 150
"""

import argparse
import json
import re
from pathlib import Path


def clean_text(text: str) -> str:
    """Clean corpus text."""
    # Normalize whitespace
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    # Normalize quotes and dashes
    text = text.replace(''', "'").replace(''', "'")
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace('—', '-').replace('–', '-')
    return text.strip()


def split_into_sentences(text: str) -> list:
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_corpus(
    text: str,
    target_words: int = 150,
    overlap_sentences: int = 2,
    min_words: int = 100
) -> list:
    """Create overlapping chunks from corpus.

    Args:
        text: Full corpus text
        target_words: Target words per chunk (~150)
        overlap_sentences: Sentences to overlap between chunks
        min_words: Minimum words per chunk

    Returns:
        List of chunk dicts with text and metadata
    """
    text = clean_text(text)

    # Split into paragraphs first (story boundaries)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    # Collect all sentences with paragraph markers
    all_sentences = []
    for para_idx, para in enumerate(paragraphs):
        sentences = split_into_sentences(para)
        for sent_idx, sent in enumerate(sentences):
            all_sentences.append({
                'text': sent,
                'words': len(sent.split()),
                'para_idx': para_idx,
                'is_para_start': sent_idx == 0,
                'is_para_end': sent_idx == len(sentences) - 1,
            })

    print(f"Total sentences: {len(all_sentences)}")
    print(f"Total paragraphs: {len(paragraphs)}")

    # Create chunks with sliding window
    chunks = []
    i = 0

    while i < len(all_sentences):
        chunk_sentences = []
        word_count = 0
        j = i
        start_para = all_sentences[i]['para_idx']

        # Add sentences until we reach target
        # Allow crossing paragraph boundaries - "style lives in transitions"
        while j < len(all_sentences):
            sent = all_sentences[j]

            # Check if adding this would exceed target by too much
            if word_count + sent['words'] > target_words * 1.4 and word_count >= min_words:
                break

            chunk_sentences.append(sent)
            word_count += sent['words']
            j += 1

            # Stop if we've reached target and hit a sentence boundary
            if word_count >= target_words:
                break

        # Only keep chunks that meet minimum size
        if word_count >= min_words:
            chunk_text = ' '.join(s['text'] for s in chunk_sentences)
            chunks.append({
                'text': chunk_text,
                'variation_type': 'original',
                'word_count': word_count,
                'sentence_count': len(chunk_sentences),
            })

        # Move forward with overlap
        sentences_used = len(chunk_sentences)
        step = max(1, sentences_used - overlap_sentences)
        i += step

        # If remaining text is too small, stop
        remaining = all_sentences[i:] if i < len(all_sentences) else []
        if remaining:
            remaining_words = sum(s['words'] for s in remaining)
            if remaining_words < min_words * 0.7:
                break

    return chunks


def main():
    parser = argparse.ArgumentParser(description="Chunk corpus for style training")
    parser.add_argument("--corpus", required=True, help="Input corpus file")
    parser.add_argument("--output", required=True, help="Output chunks JSON file")
    parser.add_argument("--target-words", type=int, default=150, help="Target words per chunk")
    parser.add_argument("--overlap", type=int, default=2, help="Sentence overlap")
    parser.add_argument("--min-words", type=int, default=100, help="Minimum words per chunk")

    args = parser.parse_args()

    print(f"Reading corpus: {args.corpus}")
    with open(args.corpus, 'r') as f:
        text = f.read()

    total_words = len(text.split())
    print(f"Total words: {total_words:,}")
    print(f"Target chunk size: {args.target_words} words")
    print(f"Overlap: {args.overlap} sentences")

    chunks = chunk_corpus(
        text,
        target_words=args.target_words,
        overlap_sentences=args.overlap,
        min_words=args.min_words
    )

    # Stats
    word_counts = [c['word_count'] for c in chunks]
    avg_words = sum(word_counts) / len(word_counts)

    print(f"\nResults:")
    print(f"  Chunks created: {len(chunks)}")
    print(f"  Avg words/chunk: {avg_words:.0f}")
    print(f"  Min words: {min(word_counts)}")
    print(f"  Max words: {max(word_counts)}")

    # Expected training size with variations
    print(f"\n  With 3 variations: ~{len(chunks) * 3:,} training examples")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
