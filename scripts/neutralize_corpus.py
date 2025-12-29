#!/usr/bin/env python3
"""Convert author corpus to content descriptions for LoRA training.

This script takes an author's distinctive text and generates content descriptions
(what happens, atmosphere, relationships) rather than paraphrases. This follows the
"instruction back-translation" approach from the Gertrude Stein paper.

The output is used for training: description → author style pairs.
Proper nouns are automatically replaced with generic descriptions to capture
STYLE rather than specific content.

Usage:
    # Default: Uses DeepSeek (best instruction following, requires API key)
    python scripts/neutralize_corpus.py \
        --input styles/sample_author.txt \
        --output data/neutralized/author.jsonl \
        --author "Author Name"

    # With parallelization:
    python scripts/neutralize_corpus.py \
        --input styles/sample_author.txt \
        --output data/neutralized/author.jsonl \
        --author "Author Name" \
        --workers 4

    # Use local MLX (no API key needed, slower):
    python scripts/neutralize_corpus.py \
        --input styles/sample_author.txt \
        --output data/neutralized/author.jsonl \
        --author "Author Name" \
        --llm mlx
"""

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def create_deepseek_generator(model: str = "deepseek-chat", timeout: int = 120):
    """Create a DeepSeek generator function (best instruction following)."""
    import os
    import requests

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable required")

    def generate(prompt: str) -> str:
        try:
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 512,
                },
                timeout=timeout,
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                print(f"DeepSeek error: {response.status_code} {response.text}")
                return ""
        except Exception as e:
            print(f"DeepSeek error: {e}")
            return ""

    return generate


def create_ollama_generator(model: str = "qwen3:8b", timeout: int = 300):
    """Create an Ollama generator function (fallback)."""
    import requests

    def generate(prompt: str) -> str:
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3},
                },
                timeout=timeout,
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            return ""
        except Exception as e:
            print(f"Ollama error: {e}")
            return ""

    return generate


def create_mlx_generator(model: str = None):
    """Create an MLX generator function (default, self-contained)."""
    from src.llm.mlx_provider import MLXGenerator

    generator = MLXGenerator(model_name=model, temperature=0.3)

    def generate(prompt: str) -> str:
        try:
            return generator.generate(prompt, max_tokens=512)
        except Exception as e:
            print(f"MLX error: {e}")
            return ""

    return generate


# Content description prompt - following the paper's approach
# This generates semantic descriptions rather than paraphrases
# We abstract away specific names to capture STYLE not CONTENT
DESCRIBE_PROMPT = """Summarize what happens in 50-100 words. You MUST replace ALL proper nouns with generic descriptions.

ABSOLUTE RULES - VIOLATION IS FAILURE:
1. NO CHARACTER NAMES (John, Mary, Jervas) → "a young man", "the woman", "the protagonist"
2. NO PLACE NAMES (London, Arkham, the Misty Valley) → "a city", "a dark forest", "a valley"
3. NO FAMILY/CLAN NAMES (the Smiths, the Hydes, House Stark) → "a noble family", "an old lineage", "the ancient house"
4. Start directly with action, never with "The passage..." or "The narrator..."

If you see ANY capitalized name, replace it with a generic description.

PASSAGE:
{chunk}

SUMMARY:"""


# Re-segmentation prompt for cleaning up chunk boundaries
RESEGMENT_PROMPT = """The following text excerpt may have been cut off mid-sentence at the beginning or end. Clean it up by:
1. Removing any incomplete sentence fragments at the START
2. Removing any incomplete sentence fragments at the END
3. Maintaining the original word count as closely as possible
4. Breaking only at grammatically natural places (sentence boundaries)

Return ONLY the cleaned text, nothing else.

TEXT:
{chunk}

CLEANED TEXT:"""


def describe_chunk(chunk: str, llm_generate) -> str:
    """Generate content description of a text chunk.

    Unlike paraphrasing, this creates a semantic description of WHAT happens,
    not HOW it's written. This forces the model to truly generate style
    during training rather than just doing minor rewording.
    """
    import re

    prompt = DESCRIBE_PROMPT.format(chunk=chunk[:3000])

    response = llm_generate(prompt)
    if not response:
        return chunk  # Fallback to original

    description = response.strip()

    # For base models: detect thinking/reasoning output and skip past it
    thinking_patterns = [
        r'^(Okay|Let me|First|I need|I\'ll|Sure|Alright|The user|This is|Here)[^:]*:?\s*',
    ]
    for pattern in thinking_patterns:
        match = re.match(pattern, description, re.IGNORECASE)
        if match:
            description = description[match.end():]
            break

    # Remove any leading/trailing quotes
    description = description.strip('"\'')

    # Stop at patterns indicating model is rambling or starting new content
    stop_patterns = ['\n\nEXCERPT:', '\n\nDESCRIPTION:', '\n\n---', '\nNote:', '\nI hope']
    for pattern in stop_patterns:
        if pattern in description:
            description = description.split(pattern)[0]

    # Clean up any trailing incomplete sentences
    description = description.strip()
    if description and description[-1] not in '.!?':
        last_period = description.rfind('.')
        if last_period > len(description) * 0.5:
            description = description[:last_period + 1]

    # Post-process to remove any proper nouns that slipped through
    description = remove_proper_nouns(description.strip())

    return description


# Lazy-loaded spaCy model for NER
_nlp = None


def get_nlp():
    """Get spaCy model, loading if needed."""
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def remove_proper_nouns(text: str) -> str:
    """Use spaCy NER to replace proper nouns with generic descriptions."""
    import re
    nlp = get_nlp()
    doc = nlp(text)

    # Map entity types to generic replacements
    replacements = {
        "PERSON": "someone",
        "GPE": "a place",
        "LOC": "a location",
        "FAC": "a building",
        "ORG": "an organization",
        "NORP": "a group",
    }

    # Build replacement list (reverse order to preserve offsets)
    to_replace = []
    for ent in doc.ents:
        if ent.label_ in replacements:
            to_replace.append((ent.start_char, ent.end_char, ent.text, ent.label_))

    # Apply replacements in reverse order
    result = text
    for start, end, original, label in reversed(to_replace):
        replacement = replacements[label]
        before = result[:start]
        after = result[end:]

        # Handle common patterns
        # "named X" -> remove the name entirely
        if re.search(r'\bnamed\s*$', before, re.IGNORECASE):
            before = re.sub(r'\bnamed\s*$', '', before, flags=re.IGNORECASE)
            replacement = ""
        # "the X family" -> "an old family"
        elif re.search(r'\bthe\s*$', before, re.IGNORECASE) and re.match(r'\s*family\b', after, re.IGNORECASE):
            before = re.sub(r'\bthe\s*$', '', before, flags=re.IGNORECASE)
            replacement = "an old"
        # "X family" -> "an old family"
        elif re.match(r'\s*family\b', after, re.IGNORECASE):
            replacement = "an old"
        # "the X" -> "the person/place"
        elif re.search(r'\b(the|The)\s*$', before):
            replacement = replacement.replace("a ", "").replace("an ", "")
        # "a X" or "an X" -> keep the article, just use noun
        elif re.search(r'\b(a|an|A|An)\s*$', before):
            replacement = replacement.replace("a ", "").replace("an ", "")

        result = before + replacement + after

    # Clean up any double spaces
    result = re.sub(r'  +', ' ', result)
    # Clean up ", ," patterns
    result = re.sub(r',\s*,', ',', result)

    return result.strip()


def needs_resegmentation(chunk: str) -> bool:
    """Check if chunk likely has incomplete sentences at boundaries.

    Returns True if the chunk appears to start or end mid-sentence.
    """
    chunk = chunk.strip()
    if not chunk:
        return False

    # Check if starts mid-sentence (lowercase letter, no capital after newline)
    first_char = chunk[0]
    if first_char.islower():
        return True

    # Check if ends mid-sentence (no terminal punctuation)
    last_char = chunk.rstrip()[-1] if chunk.rstrip() else ''
    if last_char not in '.!?"\'':
        return True

    return False


def clean_chunk_boundaries(chunk: str, llm_generate) -> str:
    """Use LLM to clean up chunk boundaries at sentence level.

    Per the paper: use LLM to re-segment with instructions to
    "maintain original word count" and "break at grammatically natural places".
    """
    if not needs_resegmentation(chunk):
        return chunk

    original_words = len(chunk.split())
    prompt = RESEGMENT_PROMPT.format(chunk=chunk)

    response = llm_generate(prompt)
    if not response:
        return chunk  # Fallback to original

    cleaned = response.strip()

    # Remove any preamble the model might add
    if cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1]

    # Validate the result isn't drastically different in length
    cleaned_words = len(cleaned.split())
    if cleaned_words < original_words * 0.7 or cleaned_words > original_words * 1.1:
        # LLM changed too much, use original
        return chunk

    return cleaned


def split_long_paragraph(para: str, max_words: int) -> list:
    """Split a long paragraph at sentence boundaries.

    Args:
        para: Paragraph text.
        max_words: Maximum words per segment.

    Returns:
        List of paragraph segments.
    """
    import re

    words = para.split()
    if len(words) <= max_words:
        return [para]

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', para)

    segments = []
    current_segment = []
    current_words = 0

    for sent in sentences:
        sent_words = len(sent.split())

        if current_words + sent_words > max_words and current_segment:
            segments.append(' '.join(current_segment))
            current_segment = []
            current_words = 0

        current_segment.append(sent)
        current_words += sent_words

    if current_segment:
        segments.append(' '.join(current_segment))

    return segments


def is_quality_paragraph(para: str) -> tuple:
    """Check if a paragraph is suitable for style training.

    Returns:
        Tuple of (is_quality, reason_if_rejected)
    """
    import re

    para = para.strip()
    if not para:
        return False, "empty"

    words = para.split()
    word_count = len(words)

    # Too short - likely headers, captions, or fragments
    if word_count < 20:
        return False, "too short"

    # Single line without sentence structure (likely a header or title)
    if '\n' not in para and '.' not in para and word_count < 50:
        return False, "single line without periods"

    # Mostly a quote (starts and ends with quotes, or > 50% quoted)
    quote_chars = para.count('"') + para.count('"') + para.count('"')
    if quote_chars > len(para) * 0.1:  # More than 10% quote marks
        return False, "mostly quotes"

    # Starts with common non-prose patterns
    skip_starters = [
        'chapter', 'part', 'section', 'book', 'volume',
        'table of contents', 'contents', 'index',
        'copyright', 'published', 'printed', 'isbn',
        'acknowledgment', 'dedication', 'preface', 'foreword',
        'introduction by', 'edited by', 'translated by',
        'about the author', 'bibliography', 'notes', 'appendix',
    ]
    para_lower = para.lower()
    for starter in skip_starters:
        if para_lower.startswith(starter):
            return False, f"starts with '{starter}'"

    # Too many special characters (likely corrupted or non-prose)
    special_count = len(re.findall(r'[^a-zA-Z0-9\s.,!?;:\'"()\-—–]', para))
    if special_count > len(para) * 0.05:  # More than 5% special chars
        return False, "too many special characters"

    # All caps (likely a header)
    if para.isupper() and word_count < 20:
        return False, "all caps header"

    # Looks like a list (multiple lines starting with numbers or bullets)
    lines = para.split('\n')
    list_lines = sum(1 for line in lines if re.match(r'^\s*(\d+[.)]|\*|\-|•)', line.strip()))
    if list_lines > len(lines) * 0.5:
        return False, "looks like a list"

    # Check for at least one complete sentence
    sentences = re.split(r'[.!?]+', para)
    complete_sentences = [s for s in sentences if s.strip() and len(s.split()) >= 5]
    if len(complete_sentences) < 1:
        return False, "no complete sentences"

    return True, "ok"


def segment_corpus(text: str, min_words: int = 250, max_words: int = 650, overlap: bool = True) -> list:
    """Segment corpus into overlapping chunks.

    Args:
        text: Full corpus text.
        min_words: Minimum words per chunk (default 250 per paper).
        max_words: Maximum words per chunk (default 650 per paper).
        overlap: If True, keep last paragraph for overlap (style lives in transitions).

    Returns:
        List of text chunks.
    """
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Filter out low-quality paragraphs and split long ones
    paragraphs = []
    skipped_reasons = {}
    for para in raw_paragraphs:
        is_quality, reason = is_quality_paragraph(para)
        if not is_quality:
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
            continue

        if len(para.split()) > max_words:
            paragraphs.extend(split_long_paragraph(para, max_words))
        else:
            paragraphs.append(para)

    # Report skipped paragraphs
    if skipped_reasons:
        total_skipped = sum(skipped_reasons.values())
        print(f"\nFiltered out {total_skipped} low-quality paragraphs:")
        for reason, count in sorted(skipped_reasons.items(), key=lambda x: -x[1]):
            print(f"  - {reason}: {count}")

    chunks = []
    current_chunk = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())

        # Finalize current chunk if adding this paragraph would exceed max_words
        # and we have enough content (or would go significantly over)
        would_exceed = current_words + para_words > max_words
        has_enough = current_words >= min_words
        significantly_over = current_words + para_words > max_words * 1.1

        if would_exceed and (has_enough or (significantly_over and current_words > 0)):
            # Finalize current chunk
            chunks.append("\n\n".join(current_chunk))

            # Start new chunk with overlap (last paragraph carries over)
            # BUT skip overlap if it would cause the new chunk to exceed max_words
            if overlap and current_chunk:
                last_paragraph = current_chunk[-1]
                last_para_words = len(last_paragraph.split())

                # Only use overlap if it won't cause immediate overflow
                if last_para_words + para_words <= max_words * 1.1:
                    current_chunk = [last_paragraph]
                    current_words = last_para_words
                else:
                    # Skip overlap to avoid creating oversized chunk
                    current_chunk = []
                    current_words = 0
            else:
                current_chunk = []
                current_words = 0

        current_chunk.append(para)
        current_words += para_words

    # Don't forget the last chunk
    if current_chunk and current_words >= min_words:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def validate_chunks(chunks: list, original_words: int, min_words: int, max_words: int) -> dict:
    """Validate chunk sizes and coverage.

    Returns:
        Dict with validation results and statistics.
    """
    if not chunks:
        return {"valid": False, "error": "No chunks produced"}

    lengths = [len(c.split()) for c in chunks]
    total_chunk_words = sum(lengths)

    stats = {
        "valid": True,
        "chunk_count": len(chunks),
        "total_words": total_chunk_words,
        "min_length": min(lengths),
        "max_length": max(lengths),
        "avg_length": sum(lengths) // len(lengths),
        "under_min": sum(1 for l in lengths if l < min_words),
        "over_max": sum(1 for l in lengths if l > max_words),
        "coverage_ratio": total_chunk_words / original_words if original_words > 0 else 0,
        "warnings": []
    }

    # Check for chunks outside bounds
    if stats["under_min"] > 0:
        stats["warnings"].append(f"{stats['under_min']} chunks under {min_words} words")

    if stats["over_max"] > 0:
        stats["warnings"].append(f"{stats['over_max']} chunks over {max_words} words")

    # Coverage should be >= 1.0 with overlap (>1 due to overlapping content)
    if stats["coverage_ratio"] < 0.9:
        stats["valid"] = False
        stats["warnings"].append(f"Low coverage: {stats['coverage_ratio']:.1%} of original content")

    return stats


def load_checkpoint(checkpoint_path: Path) -> tuple:
    """Load checkpoint if it exists.

    Returns:
        Tuple of (results_list, completed_indices_set)
    """
    if not checkpoint_path.exists():
        return [], set()

    results = []
    completed = set()
    try:
        with open(checkpoint_path, 'r') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    results.append(entry)
                    completed.add(entry.get("chunk_index", len(results) - 1))
        print(f"Loaded checkpoint: {len(results)} chunks already processed")
    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")
        return [], set()

    return results, completed


def save_checkpoint(checkpoint_path: Path, results: list):
    """Save current progress to checkpoint file."""
    with open(checkpoint_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')


class ThreadSafeCheckpointer:
    """Thread-safe checkpoint manager for parallel processing."""

    def __init__(self, checkpoint_path: Path, results: list, completed: set):
        self.checkpoint_path = checkpoint_path
        self.results = results
        self.completed = completed
        self.lock = threading.Lock()
        self.processed_count = len(completed)

    def add_result(self, result: dict):
        """Thread-safe addition of a result."""
        with self.lock:
            self.results.append(result)
            self.completed.add(result.get("chunk_index", len(self.results) - 1))
            self.processed_count += 1
            # Save checkpoint
            save_checkpoint(self.checkpoint_path, self.results)

    def is_completed(self, index: int) -> bool:
        """Check if a chunk index has been processed."""
        with self.lock:
            return index in self.completed

    def get_progress(self) -> int:
        """Get current progress count."""
        with self.lock:
            return self.processed_count


def process_single_chunk(
    chunk_info: tuple,
    llm_generate,
    author: str,
    do_cleanup: bool,
) -> dict:
    """Process a single chunk - used by both sequential and parallel modes.

    Args:
        chunk_info: Tuple of (index, chunk_text, total_chunks)
        llm_generate: LLM generation function
        author: Author name
        do_cleanup: Whether to clean chunk boundaries

    Returns:
        Result dict or None if processing failed
    """
    i, chunk, total = chunk_info

    # Clean up chunk boundaries if needed
    if do_cleanup and needs_resegmentation(chunk):
        chunk = clean_chunk_boundaries(chunk, llm_generate)

    original_words = len(chunk.split())
    description = describe_chunk(chunk, llm_generate)
    description_words = len(description.split())

    # Retry if too short
    if description_words < 20:
        description = describe_chunk(chunk, llm_generate)
        description_words = len(description.split())

    # Final fallback
    if description_words < 10:
        description = f"The text discusses various topics in {original_words} words."

    return {
        "chunk_index": i,
        "author": author,
        "original": chunk,
        "description": description,
        "word_count": original_words,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert author corpus to content descriptions for training"
    )
    parser.add_argument("--input", "-i", required=True, help="Input corpus file")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file")
    parser.add_argument("--author", "-a", required=True, help="Author name")
    parser.add_argument(
        "--llm",
        default="deepseek",
        help="LLM provider: 'deepseek' (default, best quality), 'mlx', or 'ollama:model'"
    )
    parser.add_argument("--min-words", type=int, default=250, help="Min chunk words (default: 250)")
    parser.add_argument("--max-words", type=int, default=650, help="Max chunk words (default: 650)")
    parser.add_argument("--no-overlap", action="store_true", help="Disable overlap between chunks")
    parser.add_argument("--no-cleanup", action="store_true", help="Skip LLM cleanup of chunk boundaries")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, ignore checkpoint")
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of parallel workers (DeepSeek/Ollama, MLX is single-GPU). Default: 1"
    )

    args = parser.parse_args()

    # Load corpus
    print(f"Loading corpus from {args.input}...")
    with open(args.input, 'r') as f:
        corpus = f.read()

    total_words = len(corpus.split())
    print(f"Corpus: {total_words} words")

    # Segment into chunks with overlap for transitions
    use_overlap = not args.no_overlap
    chunks = segment_corpus(corpus, args.min_words, args.max_words, overlap=use_overlap)

    # Validate chunks
    stats = validate_chunks(chunks, total_words, args.min_words, args.max_words)

    print(f"\nChunk Statistics:")
    print(f"  Count: {stats['chunk_count']}")
    print(f"  Total words: {stats['total_words']:,} (coverage: {stats['coverage_ratio']:.1%})")
    print(f"  Length range: {stats['min_length']}-{stats['max_length']} words (avg: {stats['avg_length']})")

    if stats['warnings']:
        print(f"\nWarnings:")
        for w in stats['warnings']:
            print(f"  - {w}")

    if not stats['valid']:
        print(f"\nERROR: Chunk validation failed!")
        sys.exit(1)

    print(f"\nSegmented into {len(chunks)} {'overlapping ' if use_overlap else ''}chunks")

    # Set up LLM
    if args.llm == "deepseek" or args.llm.startswith("deepseek:"):
        # Use DeepSeek (best instruction following)
        parts = args.llm.split(":", 1)
        model = parts[1] if len(parts) > 1 else "deepseek-chat"
        print(f"Using DeepSeek: {model}")
        llm_generate = create_deepseek_generator(model)
    elif args.llm == "mlx" or args.llm.startswith("mlx:"):
        # Use MLX (self-contained, no external services)
        parts = args.llm.split(":", 1)
        model = parts[1] if len(parts) > 1 else None
        print(f"Using MLX generator (self-contained)")
        llm_generate = create_mlx_generator(model)
    elif args.llm.startswith("ollama"):
        # Use Ollama (requires server)
        parts = args.llm.split(":", 1)
        model = parts[1] if len(parts) > 1 else "qwen3:8b"
        print(f"Using Ollama: {model}")
        llm_generate = create_ollama_generator(model)
    else:
        print(f"Unknown LLM provider: {args.llm}")
        print("Use 'deepseek' (default), 'mlx', or 'ollama:model_name'")
        sys.exit(1)

    # Process chunks with checkpointing
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.with_suffix(output_path.suffix + ".checkpoint")

    # Load existing progress unless --no-resume
    if args.no_resume:
        results = []
        completed = set()
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print("Starting fresh (checkpoint cleared)")
    else:
        results, completed = load_checkpoint(checkpoint_path)

    # Determine effective worker count
    is_mlx = args.llm == "mlx" or args.llm.startswith("mlx:")
    workers = args.workers

    if is_mlx and workers > 1:
        print(f"Note: MLX is single-GPU, ignoring --workers {workers} (using 1)")
        workers = 1

    # Prepare chunks to process (skip completed)
    chunks_to_process = [
        (i, chunk, len(chunks))
        for i, chunk in enumerate(chunks)
        if i not in completed
    ]

    if not chunks_to_process:
        print("All chunks already processed!")
    else:
        print(f"\nProcessing {len(chunks_to_process)} chunks with {workers} worker(s)...")
        do_cleanup = not args.no_cleanup

        try:
            if workers == 1:
                # Sequential processing
                for chunk_info in chunks_to_process:
                    i, chunk, total = chunk_info
                    print(f"[{i+1}/{total}] Processing chunk ({len(chunk.split())} words)...")

                    result = process_single_chunk(
                        chunk_info, llm_generate, args.author, do_cleanup
                    )
                    results.append(result)
                    completed.add(i)
                    save_checkpoint(checkpoint_path, results)
            else:
                # Parallel processing (Ollama)
                checkpointer = ThreadSafeCheckpointer(checkpoint_path, results, completed)
                print_lock = threading.Lock()

                def worker_fn(chunk_info):
                    i, chunk, total = chunk_info
                    with print_lock:
                        print(f"[{i+1}/{total}] Starting chunk ({len(chunk.split())} words)...")

                    result = process_single_chunk(
                        chunk_info, llm_generate, args.author, do_cleanup
                    )
                    checkpointer.add_result(result)

                    with print_lock:
                        progress = checkpointer.get_progress()
                        print(f"[{i+1}/{total}] Done (progress: {progress}/{total})")

                    return result

                with ThreadPoolExecutor(max_workers=workers) as executor:
                    # Submit all tasks
                    futures = {
                        executor.submit(worker_fn, chunk_info): chunk_info[0]
                        for chunk_info in chunks_to_process
                    }

                    # Wait for completion
                    for future in as_completed(futures):
                        idx = futures[future]
                        try:
                            future.result()
                        except Exception as e:
                            print(f"Error processing chunk {idx}: {e}")

                # Update results from checkpointer
                results = checkpointer.results
                completed = checkpointer.completed

        except KeyboardInterrupt:
            print(f"\n\nInterrupted! Progress saved to {checkpoint_path}")
            print(f"Resume with: python {sys.argv[0]} --input {args.input} --output {args.output} --author '{args.author}'")
            sys.exit(1)

    # Sort results by chunk index for consistent output
    results.sort(key=lambda x: x.get("chunk_index", 0))

    # Save final results (without chunk_index field)
    with open(output_path, 'w') as f:
        for r in results:
            # Remove chunk_index from final output
            output_record = {k: v for k, v in r.items() if k != "chunk_index"}
            f.write(json.dumps(output_record) + '\n')

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Checkpoint cleaned up")

    # Final statistics
    final_word_counts = [r.get("word_count", 0) for r in results]
    total_output_words = sum(final_word_counts)
    est_tokens = int(total_output_words * 1.3)

    print(f"\n{'='*60}")
    print(f"Neutralization Complete")
    print(f"{'='*60}")
    print(f"Output: {output_path}")
    print(f"Examples: {len(results)}")
    print(f"Total words: {total_output_words:,} (~{est_tokens:,} tokens)")
    print(f"Word count range: {min(final_word_counts)}-{max(final_word_counts)} (avg: {sum(final_word_counts)//len(final_word_counts)})")

    # Warn if below recommended token count
    if est_tokens < 500000:
        print(f"\nNote: Dataset has ~{est_tokens:,} tokens.")
        print(f"  Paper recommends ~900K tokens for optimal style capture.")
        print(f"  Consider using --epochs 3 during training to compensate.")

    print(f"\nTrain LoRA adapter with:")
    print(f"  python scripts/train_mlx_lora.py --from-neutralized {output_path} --author '{args.author}' --train --output lora_adapters/{args.author.lower()}")


if __name__ == "__main__":
    main()
