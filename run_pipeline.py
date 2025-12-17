#!/usr/bin/env python3
"""Simple script to run the text style transfer pipeline."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.pipeline import run_pipeline

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_pipeline.py <input_file> [output_file]")
        print("Example: python run_pipeline.py input/small.md output/small.md")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    if output_file is None:
        # Generate output filename from input
        input_path = Path(input_file)
        output_file = f"output/{input_path.name}"

    print(f"Processing: {input_file}")
    print(f"Output: {output_file}")
    print()

    run_pipeline(input_file=input_file, output_file=output_file)

