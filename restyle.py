#!/usr/bin/env python3
"""Style transfer using LoRA-adapted models.

Uses pre-trained LoRA adapters for fast, consistent style transfer with
a critic/repair loop to ensure content preservation and grammatical correctness.

Usage:
    # Basic usage
    python restyle.py input.md -o output.md \\
        --adapter lora_adapters/sagan \\
        --author "Carl Sagan"

    # With verbose output
    python restyle.py input.md -o output.md \\
        --adapter lora_adapters/sagan \\
        --author "Carl Sagan" \\
        --verbose

    # List available adapters
    python restyle.py --list-adapters

To train a LoRA adapter for a new author:
    # 1. Curate corpus
    python scripts/curate_corpus.py --input corpus.txt --output curated.txt

    # 2. Index in ChromaDB
    python scripts/load_corpus.py --input curated.txt --author "Author"

    # 3. Generate training data
    python scripts/generate_flat_training.py --corpus curated.txt \\
        --author "Author" --output data/training/author

    # 4. Create config.yaml (see docs/architecture.md for template)
    # 5. Train with mlx_lm.lora --config data/training/author/config.yaml
"""

import os
# Disable tokenizers parallelism warning (must be set before importing transformers)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import sys
import time
from pathlib import Path

from src.utils.logging import setup_logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def index_corpus(corpus_path: str, author: str, clear: bool = False) -> None:
    """Index an author's corpus for RAG retrieval.

    Args:
        corpus_path: Path to corpus text file.
        author: Author name.
        clear: Whether to clear existing chunks for this author.
    """
    try:
        from src.rag import CorpusIndexer, get_indexer
    except ImportError:
        print("Error: RAG dependencies not installed.")
        print("Install with: pip install chromadb sentence-transformers")
        sys.exit(1)

    indexer = get_indexer()

    print(f"Indexing corpus: {corpus_path}")
    print(f"Author: {author}")

    if clear:
        print("Clearing existing chunks...")

    try:
        count = indexer.index_corpus(corpus_path, author, clear_existing=clear)
        print(f"\nIndexed {count} chunks for {author}")
        print(f"RAG index location: data/rag_index/")
    except FileNotFoundError:
        print(f"Error: Corpus file not found: {corpus_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error indexing corpus: {e}")
        sys.exit(1)


def run_repl_mode(
    adapter_path: str,
    author: str,
    config_path: str = "config.json",
    temperature: float = 0.4,
    perspective: str = None,
    verify: bool = True,
) -> None:
    """Run interactive REPL mode.

    Args:
        adapter_path: Path to LoRA adapter.
        author: Author name.
        config_path: Path to config file.
        temperature: Generation temperature.
        perspective: Output perspective.
        verify: Whether to verify entailment.
    """
    from src.repl import run_repl
    from src.config import load_config
    from src.llm.deepseek import DeepSeekProvider

    # Load config for critic provider
    try:
        app_config = load_config(config_path)
    except FileNotFoundError:
        app_config = None

    # Create critic provider
    critic_provider = None
    if app_config and app_config.llm.providers.get("deepseek"):
        deepseek_config = app_config.llm.get_provider_config("deepseek")
        critic_provider = DeepSeekProvider(config=deepseek_config)
    else:
        import os
        from src.config import LLMProviderConfig
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if api_key:
            deepseek_config = LLMProviderConfig(
                api_key=api_key,
                model="deepseek-chat",
                base_url="https://api.deepseek.com",
            )
            critic_provider = DeepSeekProvider(config=deepseek_config)

    # Run REPL
    run_repl(
        adapter_path=adapter_path,
        author=author,
        config_path=config_path,
        temperature=temperature,
        perspective=perspective or "preserve",
        verify=verify,
        critic_provider=critic_provider,
    )


def list_rag_authors() -> None:
    """List authors indexed in RAG."""
    try:
        from src.rag import get_indexer
    except ImportError:
        print("RAG dependencies not installed.")
        print("Install with: pip install chromadb sentence-transformers")
        return

    indexer = get_indexer()
    authors = indexer.get_authors()

    if not authors:
        print("\nNo authors indexed yet.")
        print("\nTo index an author's corpus:")
        print("  python restyle.py index-corpus corpus.txt --author 'Author Name'")
        return

    print("\nIndexed authors in RAG:")
    print("-" * 40)
    for author in authors:
        count = indexer.get_chunk_count(author)
        print(f"  {author}: {count} chunks")
    print()


def list_adapters(adapters_dir: str = "lora_adapters") -> None:
    """List available LoRA adapters."""
    adapters_path = Path(adapters_dir)

    if not adapters_path.exists():
        print(f"No adapters directory found at: {adapters_path}")
        print("\nTo train an adapter, see the training workflow in README.md or run:")
        print("  1. python scripts/curate_corpus.py --input corpus.txt --output curated.txt")
        print("  2. python scripts/load_corpus.py --input curated.txt --author 'Author Name'")
        print("  3. python scripts/generate_flat_training.py --corpus curated.txt \\")
        print("         --author 'Author Name' --output data/training/author")
        print("  4. mlx_lm.lora --config data/training/author/config.yaml")
        return

    adapters = []
    for item in adapters_path.iterdir():
        if item.is_dir():
            metadata_path = item / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                adapters.append({
                    "path": str(item),
                    "author": metadata.get("author", "Unknown"),
                    "base_model": metadata.get("base_model", "Unknown"),
                    "rank": metadata.get("lora_rank", 16),
                    "examples": metadata.get("training_examples", 0),
                })

    if not adapters:
        print(f"No adapters found in: {adapters_path}")
        return

    print(f"\nAvailable LoRA adapters in {adapters_path}:\n")
    print(f"{'Author':<25} {'Path':<30} {'Rank':<6} {'Examples'}")
    print("-" * 75)

    for adapter in adapters:
        print(
            f"{adapter['author']:<25} "
            f"{Path(adapter['path']).name:<30} "
            f"{adapter['rank']:<6} "
            f"{adapter['examples']}"
        )

    print()


def transfer_file(
    input_path: str,
    output_path: str,
    adapters: list,
    author: str,
    config_path: str = "config.json",
    temperature: float = 0.2,
    perspective: str = None,
    verify: bool = True,
    verbose: bool = False,
) -> None:
    """Transfer a file using LoRA adapter(s).

    Args:
        input_path: Path to input file.
        output_path: Path to output file.
        adapters: List of AdapterSpec objects specifying adapters and their scales.
        author: Author name.
        config_path: Path to config file.
        temperature: Generation temperature.
        perspective: Output perspective (None uses config default).
        verify: Whether to verify entailment.
        verbose: Whether to print verbose output.
    """
    from src.generation.transfer import StyleTransfer, TransferConfig
    from src.generation.lora_generator import AdapterSpec
    from src.config import load_config
    from src.llm.deepseek import DeepSeekProvider

    # Load config
    try:
        app_config = load_config(config_path)
    except FileNotFoundError:
        print(f"Warning: Config file not found at {config_path}, using defaults")
        app_config = None

    # Load input
    print(f"Loading: {input_path}")
    with open(input_path, 'r') as f:
        input_text = f.read()

    word_count = len(input_text.split())
    print(f"Input: {word_count} words")

    # Configure transfer from app config or defaults
    # Determine perspective: CLI overrides config
    effective_perspective = perspective
    if effective_perspective is None and app_config:
        effective_perspective = app_config.style.perspective
    if effective_perspective is None:
        effective_perspective = "preserve"

    if app_config:
        gen = app_config.generation
        config = TransferConfig(
            # Use CLI temperature if specified, otherwise config value
            temperature=temperature,
            verify_entailment=verify,
            # Perspective
            perspective=effective_perspective,
            # From config file
            max_repair_attempts=gen.max_repair_attempts,
            repair_temperature=gen.repair_temperature,
            entailment_threshold=gen.entailment_threshold,
            max_hallucinations_before_reject=app_config.validation.max_hallucinations_before_reject,
            max_expansion_ratio=gen.max_expansion_ratio,
            target_expansion_ratio=gen.target_expansion_ratio,
            # Neutralization
            skip_neutralization=gen.skip_neutralization,
            # Post-processing
            reduce_repetition=gen.reduce_repetition,
            repetition_threshold=gen.repetition_threshold,
            use_document_context=gen.use_document_context,
            pass_headings_unchanged=gen.pass_headings_unchanged,
            min_paragraph_words=gen.min_paragraph_words,
            # RAG settings
            use_structural_rag=gen.use_structural_rag,
            use_structural_grafting=gen.use_structural_grafting,
            # Persona settings
            use_persona=gen.use_persona,
            # Sentence post-processing
            restructure_sentences=gen.restructure_sentences,
            split_sentences=gen.split_sentences,
            max_sentence_length=gen.max_sentence_length,
            sentence_length_variance=gen.sentence_length_variance,
            # Grammar correction
            correct_grammar=gen.correct_grammar,
            grammar_language=gen.grammar_language,
        )
    else:
        config = TransferConfig(
            temperature=temperature,
            verify_entailment=verify,
            perspective=effective_perspective,
            use_structural_rag=True,  # Default to enabled
        )

    # Create critic provider for repairs
    if app_config and app_config.llm.providers.get("deepseek"):
        deepseek_config = app_config.llm.get_provider_config("deepseek")
        critic_provider = DeepSeekProvider(config=deepseek_config)
        print(f"Using DeepSeek for critic/repair")
    else:
        # Try to get API key from environment
        import os
        from src.config import LLMProviderConfig
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            print("Warning: No DeepSeek API key found. Repairs will be disabled.")
            print("Set DEEPSEEK_API_KEY or configure in config.json")
            critic_provider = None
        else:
            deepseek_config = LLMProviderConfig(
                api_key=api_key,
                model="deepseek-chat",
                base_url="https://api.deepseek.com",
            )
            critic_provider = DeepSeekProvider(config=deepseek_config)
            print(f"Using DeepSeek for critic/repair (from env)")

    # Create transfer pipeline
    if len(adapters) == 1:
        print(f"\nInitializing LoRA adapter: {adapters[0].path} (scale={adapters[0].scale})")
        if adapters[0].checkpoint:
            print(f"Checkpoint: {adapters[0].checkpoint}")
    else:
        print(f"\nInitializing {len(adapters)} LoRA adapters:")
        for adapter in adapters:
            ckpt = f" checkpoint={adapter.checkpoint}" if adapter.checkpoint else ""
            print(f"  - {adapter.path} (scale={adapter.scale}){ckpt}")
    print(f"Author: {author}")

    transfer = StyleTransfer(
        adapter_path=None,
        author_name=author,
        critic_provider=critic_provider,
        config=config,
        adapters=adapters,
    )

    # Set up output file for streaming
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Track paragraphs for streaming output
    output_paragraphs = []

    # Progress callback
    def on_progress(current: int, total: int, status: str):
        if verbose:
            print(f"  [{current}/{total}] {status}")
        else:
            # Simple progress bar
            pct = int(current / total * 50)
            bar = "=" * pct + "-" * (50 - pct)
            print(f"\r  [{bar}] {current}/{total}", end="", flush=True)

    # Paragraph callback - write to file as each paragraph completes
    def on_paragraph(index: int, paragraph: str):
        output_paragraphs.append(paragraph)
        # Write all paragraphs so far to file (overwrite for clean state)
        with open(output_file, 'w') as f:
            f.write("\n\n".join(output_paragraphs))
        if verbose:
            print(f"\n--- Paragraph {index + 1} written to {output_path} ---")

    # Run transfer
    print(f"\nTransferring... (streaming to {output_path})")
    start_time = time.time()

    try:
        output_text, stats = transfer.transfer_document(
            input_text,
            on_progress=on_progress,
            on_paragraph=on_paragraph,
        )

        if not verbose:
            print()  # New line after progress bar

        # Final save (ensures proper formatting)
        with open(output_file, 'w') as f:
            f.write(output_text)

        # Print stats
        elapsed = time.time() - start_time
        output_words = len(output_text.split())

        print(f"\nComplete!")
        print(f"  Output: {output_path}")
        print(f"  Words: {word_count} -> {output_words}")
        print(f"  Time: {elapsed:.1f}s ({stats.avg_time_per_paragraph:.1f}s/paragraph)")

        if stats.entailment_scores:
            avg_score = sum(stats.entailment_scores) / len(stats.entailment_scores)
            print(f"  Content preservation: {avg_score:.1%}")

        if stats.paragraphs_repaired > 0:
            print(f"  Paragraphs repaired: {stats.paragraphs_repaired}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        # Get partial results
        partial_text, partial_stats = transfer.get_partial_results()
        elapsed = time.time() - start_time

        if output_paragraphs:
            # Save what we have
            with open(output_file, 'w') as f:
                f.write("\n\n".join(output_paragraphs))
            print(f"  Partial output saved: {output_path}")
            print(f"  Paragraphs completed: {len(output_paragraphs)}")
            print(f"  Time: {elapsed:.1f}s")
        else:
            print("  No paragraphs completed yet.")

        sys.exit(130)  # Standard exit code for Ctrl+C


def main():
    parser = argparse.ArgumentParser(
        description="Fast style transfer using LoRA adapters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Positional arguments
    parser.add_argument(
        "input",
        nargs="?",
        help="Input file path",
    )

    # Output
    parser.add_argument(
        "-o", "--output",
        help="Output file path",
    )

    # Adapter settings
    parser.add_argument(
        "--adapter",
        action="append",
        dest="adapters",
        metavar="PATH[:SCALE]",
        help="Path to LoRA adapter directory with optional scale (e.g., 'lora_adapters/sagan:0.5'). "
             "Can be specified multiple times to blend styles. Scale defaults to --lora-scale or 1.0.",
    )
    parser.add_argument(
        "--author",
        help="Author name (optional if adapter has metadata)",
    )

    # Generation settings
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Generation temperature (default: 0.4, helps complete sentences)",
    )
    parser.add_argument(
        "--perspective",
        choices=["preserve", "first_person_singular", "first_person_plural",
                 "third_person", "author_voice_third_person"],
        default=None,
        help="Output perspective: preserve (default), first_person_singular, "
             "first_person_plural, third_person, or author_voice_third_person "
             "(writes AS author using third person)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Disable entailment verification",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=None,
        help="LoRA influence scale (0.0=base only, 0.5=balanced, 1.0=full). "
             "Overrides config setting.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific checkpoint file to use (e.g., '0000600_adapters.safetensors'). "
             "Uses final adapter if not specified.",
    )

    # Utility options
    parser.add_argument(
        "--list-adapters",
        action="store_true",
        help="List available LoRA adapters",
    )
    parser.add_argument(
        "--list-rag",
        action="store_true",
        help="List authors indexed in RAG",
    )
    parser.add_argument(
        "--adapters-dir",
        default="lora_adapters",
        help="Directory containing adapters (default: lora_adapters)",
    )

    # Index corpus subcommand (handled as special input)
    parser.add_argument(
        "--index-corpus",
        metavar="CORPUS_FILE",
        help="Index a corpus file for RAG (requires --author)",
    )
    parser.add_argument(
        "--clear-rag",
        action="store_true",
        help="Clear existing RAG chunks for author when indexing",
    )

    # Config
    parser.add_argument(
        "-c", "--config",
        default="config.json",
        help="Path to config file (default: config.json)",
    )

    # Output options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    # REPL mode
    parser.add_argument(
        "--repl",
        action="store_true",
        help="Start interactive REPL mode for live style transfer",
    )

    args = parser.parse_args()

    # Setup logging - use config.log_level as default, -v overrides to INFO
    try:
        from src.config import load_config
        app_config = load_config()
        default_level = app_config.log_level
    except Exception:
        default_level = "WARNING"

    log_level = "INFO" if args.verbose else default_level
    setup_logging(level=log_level)

    # List adapters mode
    if args.list_adapters:
        list_adapters(args.adapters_dir)
        return

    # List RAG authors mode
    if args.list_rag:
        list_rag_authors()
        return

    # Index corpus mode
    if args.index_corpus:
        if not args.author:
            parser.error("--author is required for --index-corpus")
        index_corpus(args.index_corpus, args.author, args.clear_rag)
        return

    # Import AdapterSpec for parsing
    from src.generation.lora_generator import AdapterSpec

    # REPL mode
    if args.repl:
        if not args.adapters:
            parser.error("--adapter is required for REPL mode")

        # For REPL, use first adapter only
        adapter_path = AdapterSpec.parse(args.adapters[0]).path

        # Load author from metadata if not provided
        author = args.author
        if not author:
            metadata_path = Path(adapter_path) / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                author = metadata.get("author")

        if not author:
            parser.error("--author is required (not found in adapter metadata)")

        run_repl_mode(
            adapter_path=adapter_path,
            author=author,
            config_path=args.config,
            temperature=args.temperature,
            perspective=args.perspective,
            verify=not args.no_verify,
        )
        return

    # Validate required arguments for transfer
    if not args.input:
        parser.error("Input file is required (or use --list-adapters, --list-rag)")

    if not args.output:
        # Default output name
        input_path = Path(args.input)
        args.output = str(input_path.with_suffix(".styled" + input_path.suffix))

    # Parse adapter specs from CLI or config
    adapters = []

    if args.adapters:
        # CLI adapters specified - parse them
        default_scale = args.lora_scale if args.lora_scale is not None else 1.0

        for spec_str in args.adapters:
            adapter = AdapterSpec.parse(spec_str)
            # If no scale was specified in the spec string, use the default
            if ':' not in spec_str:
                adapter.scale = default_scale
            # Apply checkpoint to first adapter if specified via --checkpoint
            if len(adapters) == 0 and args.checkpoint:
                adapter.checkpoint = args.checkpoint
            adapters.append(adapter)
    else:
        # Try to load adapters from config file
        from src.config import load_config
        try:
            app_config = load_config(args.config)
            lora_adapters = app_config.generation.lora_adapters
            if lora_adapters:
                for path, value in lora_adapters.items():
                    # Value can be a number (scale) or dict {scale, checkpoint}
                    if isinstance(value, dict):
                        scale = value.get("scale", 1.0)
                        checkpoint = value.get("checkpoint")
                        adapters.append(AdapterSpec(path=path, scale=scale, checkpoint=checkpoint))
                    else:
                        # Simple number format (scale only)
                        adapters.append(AdapterSpec(path=path, scale=float(value)))
                print(f"Using adapters from config: {args.config}")
        except (FileNotFoundError, AttributeError):
            pass

    if not adapters:
        parser.error("--adapter is required (or configure lora_adapters in config.json)")

    # Load author from metadata if not provided
    author = args.author
    if not author:
        metadata_path = Path(adapters[0].path) / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            author = metadata.get("author")

    if not author:
        parser.error("--author is required (not found in adapter metadata)")

    # Run transfer
    transfer_file(
        input_path=args.input,
        output_path=args.output,
        adapters=adapters,
        author=author,
        config_path=args.config,
        temperature=args.temperature,
        perspective=args.perspective,
        verify=not args.no_verify,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        sys.exit(130)
