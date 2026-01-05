#!/usr/bin/env python3
"""Generate training data for structural style transfer.

This script creates training pairs that capture STRUCTURAL style elements
(sentence rhythm, clause nesting, punctuation patterns) independent of content.

The key insight: style lives in syntax, not vocabulary. We want:
- Simple → Complex nested sentences
- Direct → Inverted word order
- Short sentences → Long flowing periods with em-dashes
- Modern punctuation → Semicolons, em-dashes, parentheticals

Usage:
    python scripts/generate_structural_training.py \
        --corpus data/corpus/diverse_samples.txt \
        --author "H.P. Lovecraft" \
        --output data/training/lovecraft_structural
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
import random

# DeepSeek API
import requests

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


@dataclass
class StyleFeatures:
    """Structural style features to apply."""
    em_dashes: bool = True  # Use em-dashes for parentheticals
    semicolons: bool = True  # Join clauses with semicolons
    inversions: bool = True  # Inverted sentence structures
    nested_clauses: bool = True  # Complex nested modifiers
    long_sentences: bool = True  # Flowing multi-clause sentences
    archaic_connectives: bool = True  # "for", "yet", "whilst"


def call_deepseek(prompt: str, system: str = "", max_retries: int = 3) -> Optional[str]:
    """Call DeepSeek API."""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    data = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 2000,
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{DEEPSEEK_BASE_URL}/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"API call failed (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)

    return None


# Global RTT neutralizer (shared across calls)
_rtt_neutralizer = None


def get_rtt_neutralizer():
    """Get or create shared RTT neutralizer (singleton pattern)."""
    global _rtt_neutralizer
    if _rtt_neutralizer is None:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.llm.mlx_provider import RTTNeutralizer
        print("Initializing RTT neutralizer (local MLX model)...")
        _rtt_neutralizer = RTTNeutralizer()
        print("RTT neutralizer ready")
    return _rtt_neutralizer


def rtt_neutralize(text: str) -> Optional[str]:
    """Round-trip translate through Mandarin to strip style.

    Uses local MLX model (Qwen2.5-3B-Instruct) for fast inference.
    Configuration in config.json under llm.providers.mlx_rtt.
    """
    try:
        neutralizer = get_rtt_neutralizer()
        return neutralizer.neutralize(text, max_retries=2)
    except Exception as e:
        print(f"RTT neutralization failed: {e}")
        return None


def apply_structural_style(neutral_text: str, author: str) -> Optional[str]:
    """Apply structural style transformation (syntax, rhythm, punctuation).

    This focuses on STRUCTURE not vocabulary:
    - Sentence architecture
    - Clause nesting
    - Punctuation patterns
    - Rhythmic flow
    """

    style_prompt = f"""Transform this text using {author}'s STRUCTURAL style.

FOCUS ON STRUCTURE, NOT VOCABULARY:
1. SENTENCE ARCHITECTURE: Convert simple sentences to complex nested structures
2. PUNCTUATION: Use em-dashes (—) for dramatic parentheticals, semicolons to join related thoughts
3. INVERSIONS: "It was X that..." instead of "X was..."
4. FLOWING RHYTHM: Connect short sentences into longer, flowing periods
5. CONNECTIVES: Use "for", "yet", "whilst", "and yet" to link clauses

DO NOT:
- Change the topic or subject matter
- Add horror/supernatural elements unless present in original
- Substitute vocabulary unnecessarily (keep technical terms)

EXAMPLE TRANSFORMATION:
Before: "The bank reported losses. Investors grew concerned. The market showed instability."
After: "It was losses—grave and mounting—that the bank reported; and at this disclosure, investors grew concerned, for the market had begun to show those first tremors of instability that presage greater upheaval."

Text to transform:
{neutral_text}

Transformed text (preserve all facts, change only structure):"""

    styled = call_deepseek(
        style_prompt,
        system=f"You are a style transformer. Apply {author}'s sentence structure and rhythm to any content while preserving meaning exactly."
    )

    if styled:
        styled = styled.strip('`"\' \n')
        # Clean any meta-commentary
        styled = re.sub(r'^(Here|This|I\'ve).*?:\s*', '', styled, flags=re.IGNORECASE)
        styled = re.sub(r'\n\n.*?(Note|This transformation).*$', '', styled, flags=re.IGNORECASE | re.DOTALL)

    return styled


def create_diverse_samples() -> List[str]:
    """Create diverse content samples for training."""

    samples = [
        # Financial/Economic
        """The quarterly report showed declining revenues. Operating costs increased by fifteen percent. Management attributed this to supply chain disruptions. Shareholders expressed concern at the annual meeting.""",

        """Interest rates affect housing prices directly. When rates rise, fewer people can afford mortgages. This reduces demand and prices fall. The cycle typically takes eighteen months to complete.""",

        # Scientific/Technical
        """The experiment yielded unexpected results. Temperature fluctuations caused measurement errors. The team recalibrated their instruments. Subsequent trials confirmed the initial hypothesis.""",

        """Neural networks process information through layers. Each layer transforms the input data. The final layer produces the output prediction. Training adjusts the weights between neurons.""",

        # Historical/Political
        """The treaty was signed in Vienna. Both parties made significant concessions. Economic sanctions were lifted immediately. Political prisoners were released within thirty days.""",

        """The revolution began in the capital city. Workers organized protests throughout the industrial district. The government deployed military forces. Within weeks, the regime had collapsed.""",

        # Philosophical/Abstract
        """Knowledge requires justification and truth. Mere belief does not constitute knowledge. The evidence must support the conclusion. This forms the basis of epistemology.""",

        """Time appears to flow in one direction. Events occur in sequence. The past cannot be changed. These observations suggest time has an inherent asymmetry.""",

        # Everyday/Mundane
        """The coffee machine broke this morning. Water leaked across the counter. Someone had forgotten to empty the filter. The repair technician will arrive tomorrow.""",

        """Traffic was heavy on the highway. An accident blocked two lanes. Commuters waited for hours. The delay affected thousands of workers.""",
    ]

    return samples


def process_chunk(text: str, author: str) -> Optional[Tuple[str, str]]:
    """Process a single chunk: neutralize then apply structural style."""

    # Step 1: RTT neutralization
    neutral = rtt_neutralize(text)
    if not neutral:
        print(f"  RTT failed for: {text[:50]}...")
        return None

    # Step 2: Apply structural style
    styled = apply_structural_style(neutral, author)
    if not styled:
        print(f"  Style application failed for: {neutral[:50]}...")
        return None

    # Validate lengths are reasonable
    neutral_words = len(neutral.split())
    styled_words = len(styled.split())

    if styled_words < neutral_words * 0.5 or styled_words > neutral_words * 3:
        print(f"  Length mismatch: {neutral_words} → {styled_words}")
        return None

    return (neutral, styled)


def generate_training_data(
    samples: List[str],
    author: str,
    output_dir: Path,
) -> int:
    """Generate training data from diverse samples."""

    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = output_dir / "train.jsonl"
    valid_file = output_dir / "valid.jsonl"

    all_pairs = []

    for i, sample in enumerate(samples):
        print(f"Processing sample {i+1}/{len(samples)}...")

        result = process_chunk(sample, author)
        if result:
            neutral, styled = result
            all_pairs.append((neutral, styled))
            print(f"  ✓ Generated pair: {len(neutral.split())} → {len(styled.split())} words")

        # Rate limiting
        time.sleep(1)

    if not all_pairs:
        print("No training pairs generated!")
        return 0

    # Shuffle and split
    random.shuffle(all_pairs)
    split_idx = max(1, int(len(all_pairs) * 0.9))
    train_pairs = all_pairs[:split_idx]
    valid_pairs = all_pairs[split_idx:]

    # Format for training
    author_tag = author.upper().replace(' ', '_').replace('.', '')

    def format_example(neutral: str, styled: str) -> dict:
        text = f"Rewrite the following neutral text into the style of {author}.\n\n[NEUTRAL INPUT]: {neutral}\n\n[{author_tag} OUTPUT]: {styled}"
        return {"text": text}

    # Write files
    with open(train_file, 'w') as f:
        for neutral, styled in train_pairs:
            f.write(json.dumps(format_example(neutral, styled)) + '\n')

    with open(valid_file, 'w') as f:
        for neutral, styled in valid_pairs:
            f.write(json.dumps(format_example(neutral, styled)) + '\n')

    print(f"\nGenerated {len(train_pairs)} train, {len(valid_pairs)} valid examples")
    print(f"Output: {output_dir}")

    return len(all_pairs)


def main():
    parser = argparse.ArgumentParser(
        description="Generate structural style training data"
    )
    parser.add_argument("--author", default="H.P. Lovecraft", help="Author name")
    parser.add_argument("--output", default="data/training/structural", help="Output directory")
    parser.add_argument("--corpus", help="Optional corpus file (one paragraph per line)")

    args = parser.parse_args()

    if not DEEPSEEK_API_KEY:
        print("Error: DEEPSEEK_API_KEY environment variable not set")
        return

    # Get samples
    if args.corpus and Path(args.corpus).exists():
        print(f"Loading corpus from {args.corpus}")
        with open(args.corpus, 'r') as f:
            samples = [line.strip() for line in f if line.strip() and len(line.split()) > 20]
    else:
        print("Using built-in diverse samples")
        samples = create_diverse_samples()

    print(f"Processing {len(samples)} samples for author: {args.author}")

    count = generate_training_data(
        samples=samples,
        author=args.author,
        output_dir=Path(args.output),
    )

    print(f"\nDone! Generated {count} training examples")
    print("\nTo train:")
    print(f"  mlx_lm.lora --train -c lora_adapters/{args.author.lower().replace(' ', '_')}/config.yaml")


if __name__ == "__main__":
    main()
