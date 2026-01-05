#!/usr/bin/env python3
"""Convert training data from instruct (messages) format to base model format.

Instruct format (before):
    {"messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "neutral text"},
        {"role": "assistant", "content": "styled text"}
    ], "word_count": N}

Base format (after - prompt/completion for pattern triggering):
    {
        "prompt": "Rewrite the following neutral text into the style of H.P. Lovecraft.\n\n[NEUTRAL INPUT]: ...\n\n[LOVECRAFT OUTPUT]:",
        "completion": " The ancient figure shambled..."
    }

Note: Leading space in completion ensures it connects to prompt.

Usage:
    python scripts/convert_to_base_format.py \
        --input data/training/lovecraft \
        --author "H.P. Lovecraft"
"""

import argparse
import json
from pathlib import Path


def convert_example(example: dict, author: str) -> dict:
    """Convert a single example to base model text format."""
    neutral_text = ""
    styled_text = ""

    # Handle messages format
    if "messages" in example:
        messages = example["messages"]
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                neutral_text = content
            elif role == "assistant":
                styled_text = content

    # Handle prompt/completion format
    elif "prompt" in example and "completion" in example:
        prompt = example["prompt"]
        styled_text = example["completion"].strip()

        # Extract neutral text from prompt (after [NEUTRAL INPUT]:)
        if "[NEUTRAL INPUT]:" in prompt:
            parts = prompt.split("[NEUTRAL INPUT]:", 1)
            if len(parts) > 1:
                neutral_part = parts[1]
                # Get text before the output tag
                if "[" in neutral_part:
                    neutral_text = neutral_part.split("\n\n[")[0].strip()
                else:
                    neutral_text = neutral_part.strip()

    # Handle text format with ### separator
    elif "text" in example:
        text = example["text"]
        if "###\n" in text:
            parts = text.split("###\n", 1)
            first_part = parts[0]
            if "\n" in first_part:
                neutral_text = first_part.split("\n", 1)[1].strip()
            else:
                neutral_text = first_part.strip()
            styled_text = parts[1] if len(parts) > 1 else ""
        elif "[NEUTRAL INPUT]:" in text and " OUTPUT]:" in text:
            # New format with pattern triggers
            parts = text.split("[NEUTRAL INPUT]:", 1)
            if len(parts) > 1:
                rest = parts[1]
                if " OUTPUT]:" in rest:
                    neutral_styled = rest.split(" OUTPUT]:", 1)
                    neutral_text = neutral_styled[0].strip().rstrip("\n\n[HP_LOVECRAFT").rstrip("\n\n[")
                    styled_text = neutral_styled[1].strip() if len(neutral_styled) > 1 else ""
        else:
            return None
    else:
        return None

    if not neutral_text or not styled_text:
        return None

    # Format for base model: single "text" field
    # Base models don't have chat templates, so we use TextDataset format
    # The rigid pattern triggers help the model learn the transformation
    author_tag = author.upper().replace(' ', '_').replace('.', '')

    text = f"Rewrite the following neutral text into the style of {author}.\n\n[NEUTRAL INPUT]: {neutral_text}\n\n[{author_tag} OUTPUT]: {styled_text}"

    return {
        "text": text
    }


def convert_file(input_path: Path, output_path: Path, author: str) -> int:
    """Convert a JSONL file from messages to text format."""
    converted = 0

    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            if not line.strip():
                continue

            example = json.loads(line)
            converted_example = convert_example(example, author)

            if converted_example:
                f_out.write(json.dumps(converted_example) + '\n')
                converted += 1

    return converted


def main():
    parser = argparse.ArgumentParser(
        description="Convert training data from instruct to base model format"
    )
    parser.add_argument("--input", required=True, help="Input directory with train/valid/test.jsonl")
    parser.add_argument("--author", required=True, help="Author name for prompt")
    parser.add_argument("--backup", action="store_true", help="Create .bak backup of originals")

    args = parser.parse_args()

    input_dir = Path(args.input)

    for split in ["train", "valid", "test"]:
        input_file = input_dir / f"{split}.jsonl"

        if not input_file.exists():
            print(f"Skipping {split}.jsonl (not found)")
            continue

        # Backup if requested
        if args.backup:
            backup_file = input_dir / f"{split}.jsonl.bak"
            import shutil
            shutil.copy(input_file, backup_file)
            print(f"Backed up {split}.jsonl -> {split}.jsonl.bak")

        # Convert in place (write to temp, then rename)
        temp_file = input_dir / f"{split}.jsonl.tmp"
        count = convert_file(input_file, temp_file, args.author)

        # Replace original
        temp_file.rename(input_file)
        print(f"Converted {split}.jsonl: {count} examples")

    print("\nDone! Training data is now in base model format.")
    print("Format: {\"text\": \"Rewrite in {author}'s style (~N words):\\n{neutral}\\n###\\n{styled}\"}")


if __name__ == "__main__":
    main()
