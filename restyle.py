#!/usr/bin/env python3
"""CLI entry point for text style transfer pipeline.

DEPRECATED: This script uses the legacy RAG-based pipeline.
For the new LoRA-based approach (faster, better quality), use:

    python scripts/fast_restyle.py input.md -o output.md \\
        --adapter lora_adapters/author \\
        --author "Author Name"

To train a LoRA adapter for a new author:

    # 1. Curate corpus (optional, for large corpuses)
    python scripts/curate_corpus.py --input corpus.txt --output curated.txt

    # 2. Generate training data
    python scripts/neutralize_corpus.py --input curated.txt \\
        --output data/neutralized/author.jsonl --author "Author"

    # 3. Train the adapter
    python scripts/train_mlx_lora.py --from-neutralized data/neutralized/author.jsonl \\
        --author "Author" --train --output lora_adapters/author
"""

import sys

def main():
    print("=" * 60)
    print("DEPRECATED: restyle.py uses the legacy RAG-based pipeline")
    print("=" * 60)
    print()
    print("Please use the new LoRA-based approach instead:")
    print()
    print("  python scripts/fast_restyle.py input.md -o output.md \\")
    print("      --adapter lora_adapters/author \\")
    print("      --author 'Author Name'")
    print()
    print("For more info, see: python scripts/fast_restyle.py --help")
    print()
    sys.exit(1)


if __name__ == '__main__':
    main()
