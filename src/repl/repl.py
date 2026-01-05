"""Interactive REPL for style transfer.

Provides a terminal UI similar to Claude Code for interactive style transfer.
"""

import sys
import os
import textwrap
from typing import Optional, Callable
from dataclasses import dataclass

from ..generation.transfer import StyleTransfer, TransferConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)

# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"

    # Foreground
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background
    BG_BLACK = "\033[40m"
    BG_BLUE = "\033[44m"
    BG_CYAN = "\033[46m"


def supports_color() -> bool:
    """Check if terminal supports colors."""
    if os.environ.get("NO_COLOR"):
        return False
    if not hasattr(sys.stdout, "isatty"):
        return False
    if not sys.stdout.isatty():
        return False
    return True


def get_terminal_width() -> int:
    """Get terminal width, default to 80."""
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 80


@dataclass
class REPLConfig:
    """Configuration for the REPL."""
    author: str
    adapter_path: str
    temperature: float = 0.4
    verify: bool = True
    perspective: str = "preserve"
    use_color: bool = True


class StyleREPL:
    """Interactive REPL for style transfer."""

    def __init__(
        self,
        transfer: StyleTransfer,
        config: REPLConfig,
    ):
        """Initialize the REPL.

        Args:
            transfer: Initialized StyleTransfer instance.
            config: REPL configuration.
        """
        self.transfer = transfer
        self.config = config
        self.use_color = config.use_color and supports_color()
        self.history: list[tuple[str, str]] = []  # (input, output) pairs
        self.running = False

    def _color(self, text: str, *codes: str) -> str:
        """Apply color codes to text if colors are enabled."""
        if not self.use_color:
            return text
        return "".join(codes) + text + Colors.RESET

    def _print_header(self) -> None:
        """Print the REPL header."""
        width = get_terminal_width()

        # Title
        title = f" Style Transfer: {self.config.author} "
        padding = (width - len(title)) // 2

        print()
        if self.use_color:
            print(self._color("─" * width, Colors.DIM))
            print(
                self._color("─" * padding, Colors.DIM) +
                self._color(title, Colors.BOLD, Colors.CYAN) +
                self._color("─" * (width - padding - len(title)), Colors.DIM)
            )
            print(self._color("─" * width, Colors.DIM))
        else:
            print("─" * width)
            print("─" * padding + title + "─" * (width - padding - len(title)))
            print("─" * width)

        # Instructions
        print()
        print(self._color("  Enter text to transform (press Enter twice to submit)", Colors.DIM))
        print(self._color("  Commands: /help, /clear, /history, /quit", Colors.DIM))
        print()

    def _print_help(self) -> None:
        """Print help information."""
        print()
        print(self._color("Commands:", Colors.BOLD))
        print(f"  {self._color('/help', Colors.CYAN)}     Show this help message")
        print(f"  {self._color('/clear', Colors.CYAN)}    Clear the screen")
        print(f"  {self._color('/history', Colors.CYAN)}  Show transformation history")
        print(f"  {self._color('/last', Colors.CYAN)}     Show last transformation")
        print(f"  {self._color('/quit', Colors.CYAN)}     Exit the REPL")
        print()
        print(self._color("Input:", Colors.BOLD))
        print("  Type or paste your text, then press Enter twice to transform.")
        print("  Multi-line input is supported - just keep typing until done.")
        print()

    def _print_history(self) -> None:
        """Print transformation history."""
        if not self.history:
            print(self._color("  No transformations yet.", Colors.DIM))
            return

        print()
        print(self._color(f"History ({len(self.history)} transformations):", Colors.BOLD))
        print()

        for i, (inp, out) in enumerate(self.history, 1):
            # Truncate for display
            inp_preview = inp[:50] + "..." if len(inp) > 50 else inp
            out_preview = out[:50] + "..." if len(out) > 50 else out

            print(f"  {self._color(f'[{i}]', Colors.DIM)}")
            print(f"    {self._color('Input:', Colors.YELLOW)} {inp_preview}")
            print(f"    {self._color('Output:', Colors.GREEN)} {out_preview}")
            print()

    def _print_last(self) -> None:
        """Print the last transformation."""
        if not self.history:
            print(self._color("  No transformations yet.", Colors.DIM))
            return

        inp, out = self.history[-1]
        print()
        print(self._color("Last transformation:", Colors.BOLD))
        print()
        print(self._color("Input:", Colors.YELLOW))
        self._print_wrapped(inp)
        print()
        print(self._color("Output:", Colors.GREEN))
        self._print_wrapped(out)
        print()

    def _print_wrapped(self, text: str, indent: int = 2) -> None:
        """Print text with word wrapping."""
        width = get_terminal_width() - indent - 2
        wrapper = textwrap.TextWrapper(
            width=width,
            initial_indent=" " * indent,
            subsequent_indent=" " * indent,
        )

        for paragraph in text.split("\n\n"):
            if paragraph.strip():
                print(wrapper.fill(paragraph))
                print()

    def _read_input(self) -> Optional[str]:
        """Read multi-line input until double Enter."""
        lines = []
        empty_count = 0

        print(self._color("│ ", Colors.BLUE), end="", flush=True)

        try:
            while True:
                try:
                    line = input()
                except EOFError:
                    return None

                if not line:
                    empty_count += 1
                    if empty_count >= 1 and lines:
                        # Double enter - submit
                        break
                    elif empty_count >= 2:
                        # Triple enter with no content - skip
                        return ""
                else:
                    empty_count = 0
                    lines.append(line)

                # Show continuation prompt
                if empty_count == 0:
                    print(self._color("│ ", Colors.BLUE), end="", flush=True)
        except KeyboardInterrupt:
            print()
            return ""

        return "\n".join(lines)

    def _transform_text(self, text: str) -> Optional[str]:
        """Transform text using the style transfer pipeline."""
        if not text.strip():
            return None

        # Show processing indicator
        print()
        print(self._color("  Transforming...", Colors.DIM), end="", flush=True)

        try:
            # Use transfer_paragraph for single chunks, transfer_document for longer
            word_count = len(text.split())

            if word_count < 100:
                # Short text - single paragraph mode
                output, score = self.transfer.transfer_paragraph(text)
            else:
                # Longer text - document mode
                output, stats = self.transfer.transfer_document(text)

            # Clear the "Transforming..." line
            print("\r" + " " * 50 + "\r", end="")

            return output

        except Exception as e:
            print("\r" + " " * 50 + "\r", end="")
            print(self._color(f"  Error: {e}", Colors.RED))
            logger.exception("Transform error")
            return None

    def _print_output(self, output: str) -> None:
        """Print the transformed output."""
        width = get_terminal_width()

        print()
        print(self._color("─" * width, Colors.DIM))
        print(self._color(f"  Output ({len(output.split())} words):", Colors.GREEN, Colors.BOLD))
        print(self._color("─" * width, Colors.DIM))
        print()

        self._print_wrapped(output)

        print(self._color("─" * width, Colors.DIM))
        print()

    def _handle_command(self, cmd: str) -> bool:
        """Handle a command. Returns True to continue, False to quit."""
        cmd = cmd.lower().strip()

        if cmd in ("/quit", "/exit", "/q"):
            return False
        elif cmd in ("/help", "/h", "/?"):
            self._print_help()
        elif cmd in ("/clear", "/cls"):
            os.system("clear" if os.name != "nt" else "cls")
            self._print_header()
        elif cmd in ("/history", "/hist"):
            self._print_history()
        elif cmd == "/last":
            self._print_last()
        else:
            print(self._color(f"  Unknown command: {cmd}", Colors.RED))
            print(self._color("  Type /help for available commands", Colors.DIM))

        return True

    def run(self) -> None:
        """Run the REPL main loop."""
        self.running = True
        self._print_header()

        while self.running:
            try:
                # Read input
                text = self._read_input()

                if text is None:
                    # EOF
                    break

                if not text:
                    # Empty input
                    continue

                # Check for commands
                if text.startswith("/"):
                    if not self._handle_command(text):
                        break
                    continue

                # Transform text
                output = self._transform_text(text)

                if output:
                    self._print_output(output)
                    self.history.append((text, output))

            except KeyboardInterrupt:
                print()
                print(self._color("\n  Use /quit to exit", Colors.DIM))
                continue

        # Goodbye
        print()
        print(self._color("  Goodbye!", Colors.CYAN))
        print()


def run_repl(
    adapter_path: str,
    author: str,
    config_path: str = "config.json",
    temperature: float = 0.4,
    perspective: str = "preserve",
    verify: bool = True,
    critic_provider = None,
) -> None:
    """Run the interactive REPL.

    Args:
        adapter_path: Path to LoRA adapter.
        author: Author name.
        config_path: Path to config file.
        temperature: Generation temperature.
        perspective: Output perspective.
        verify: Whether to verify entailment.
        critic_provider: Optional critic provider for repairs.
    """
    from ..config import load_config

    # Load config
    try:
        app_config = load_config(config_path)
    except FileNotFoundError:
        app_config = None

    # Build transfer config
    if app_config:
        gen = app_config.generation
        transfer_config = TransferConfig(
            temperature=temperature,
            verify_entailment=verify,
            perspective=perspective,
            max_repair_attempts=gen.max_repair_attempts,
            entailment_threshold=gen.entailment_threshold,
            max_expansion_ratio=gen.max_expansion_ratio,
            target_expansion_ratio=gen.target_expansion_ratio,
            lora_scale=gen.lora_scale,
            skip_neutralization=gen.skip_neutralization,
            reduce_repetition=gen.reduce_repetition,
            repetition_threshold=gen.repetition_threshold,
            use_structural_rag=gen.use_structural_rag,
            # Disable document context for REPL (interactive mode)
            use_document_context=False,
            pass_headings_unchanged=False,
            min_paragraph_words=5,  # Lower threshold for REPL
        )
    else:
        transfer_config = TransferConfig(
            temperature=temperature,
            verify_entailment=verify,
            perspective=perspective,
            use_document_context=False,
            min_paragraph_words=5,
        )

    # Print loading message
    print()
    print(f"Loading LoRA adapter: {adapter_path}")
    print(f"Author: {author}")
    print()

    # Initialize transfer
    transfer = StyleTransfer(
        adapter_path=adapter_path,
        author_name=author,
        critic_provider=critic_provider,
        config=transfer_config,
    )

    # Create REPL config
    repl_config = REPLConfig(
        author=author,
        adapter_path=adapter_path,
        temperature=temperature,
        verify=verify,
        perspective=perspective,
    )

    # Run REPL
    repl = StyleREPL(transfer, repl_config)
    repl.run()
