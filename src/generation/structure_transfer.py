"""Structure-preserving style transfer.

This module implements style transfer that preserves the narrative structure
of the source paragraph - not just the content, but the rhythm, sentence count,
and rhetorical flow.

Key insight: A paragraph is a STORY with structure. Changing the structure
changes the story. We must preserve:
1. Sentence count (roughly)
2. Sentence lengths (rhythm pattern)
3. Rhetorical moves (claim → support → example → conclusion)
4. Inter-sentence flow (each sentence connects to the next)

The LoRA can change vocabulary and phrasing, but not the fundamental structure.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import re

from .lora_generator import LoRAStyleGenerator
from ..utils.nlp import get_nlp
from ..utils.logging import get_logger

logger = get_logger(__name__)


class RhetoricalMove(Enum):
    """Types of rhetorical moves in a paragraph."""
    CLAIM = "claim"           # Main assertion or topic statement
    SUPPORT = "support"       # Supporting detail or elaboration
    EXAMPLE = "example"       # Concrete example or illustration
    CONSEQUENCE = "consequence"  # Result or implication
    CONTRAST = "contrast"     # But/however/yet
    REFLECTION = "reflection"  # Closing thought or thematic return


@dataclass
class SentenceSpec:
    """Specification for a single sentence in the output."""
    index: int
    original: str
    target_words: int
    move: RhetoricalMove
    content_hint: str  # What this sentence should convey
    is_short: bool  # True if this should be punchy (≤6 words)
    connects_to_previous: bool  # True if this builds on previous sentence


@dataclass
class ParagraphStructure:
    """Complete structural analysis of a paragraph."""
    sentence_count: int
    sentences: List[SentenceSpec]
    total_words: int

    # Rhythm analysis
    short_sentence_indices: List[int]  # Which sentences are punchy
    length_pattern: str  # e.g., "MMLMSSLMM" (Medium, Long, Short)

    def get_length_category(self, word_count: int) -> str:
        """Categorize a word count as Short/Medium/Long."""
        if word_count <= 6:
            return "S"
        elif word_count <= 12:
            return "M"
        else:
            return "L"


class StructuralAnalyzer:
    """Analyzes the narrative structure of a paragraph."""

    def __init__(self):
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def analyze(self, text: str) -> ParagraphStructure:
        """Analyze paragraph structure for preservation during transfer."""
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        if not sentences:
            return ParagraphStructure(
                sentence_count=0,
                sentences=[],
                total_words=0,
                short_sentence_indices=[],
                length_pattern="",
            )

        specs = []
        short_indices = []
        pattern_chars = []

        for i, sent in enumerate(sentences):
            word_count = len(sent.split())
            is_short = word_count <= 6

            if is_short:
                short_indices.append(i)

            # Classify rhetorical move
            move = self._classify_move(sent, i, len(sentences))

            # Extract content hint (key concepts)
            content_hint = self._extract_content_hint(sent)

            # Check connection to previous
            connects = i > 0 and self._connects_to_previous(sent, sentences[i-1])

            specs.append(SentenceSpec(
                index=i,
                original=sent,
                target_words=word_count,
                move=move,
                content_hint=content_hint,
                is_short=is_short,
                connects_to_previous=connects,
            ))

            # Build length pattern
            if word_count <= 6:
                pattern_chars.append("S")
            elif word_count <= 12:
                pattern_chars.append("M")
            else:
                pattern_chars.append("L")

        return ParagraphStructure(
            sentence_count=len(sentences),
            sentences=specs,
            total_words=len(text.split()),
            short_sentence_indices=short_indices,
            length_pattern="".join(pattern_chars),
        )

    def _classify_move(self, sent: str, index: int, total: int) -> RhetoricalMove:
        """Classify the rhetorical move of a sentence."""
        sent_lower = sent.lower()
        word_count = len(sent.split())

        # Short sentences are often examples
        if word_count <= 6:
            return RhetoricalMove.EXAMPLE

        # First sentence is usually a claim
        if index == 0:
            return RhetoricalMove.CLAIM

        # Last sentence is often reflection
        if index == total - 1:
            return RhetoricalMove.REFLECTION

        # Contrast markers
        if any(w in sent_lower.split()[:3] for w in ['yet', 'but', 'however', 'although']):
            return RhetoricalMove.CONTRAST

        # Consequence markers
        if any(w in sent_lower for w in ['therefore', 'thus', 'consequently', 'as a result']):
            return RhetoricalMove.CONSEQUENCE

        # Default to support
        return RhetoricalMove.SUPPORT

    def _extract_content_hint(self, sent: str) -> str:
        """Extract the key content hint from a sentence."""
        doc = self.nlp(sent)

        # Get main subject-verb-object if possible
        subjects = []
        verbs = []
        objects = []

        for token in doc:
            if token.dep_ in ['nsubj', 'nsubjpass']:
                subjects.append(token.text)
            elif token.pos_ == 'VERB' and token.dep_ == 'ROOT':
                verbs.append(token.lemma_)
            elif token.dep_ in ['dobj', 'pobj', 'attr']:
                objects.append(token.text)

        # Build hint from key content words
        content_words = []
        for token in doc:
            if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN'] and not token.is_stop:
                content_words.append(token.text.lower())

        return ' '.join(content_words[:6])  # Limit to key words

    def _connects_to_previous(self, sent: str, prev_sent: str) -> bool:
        """Check if sentence explicitly connects to previous."""
        sent_lower = sent.lower()

        # Explicit connectors
        connectors = ['this', 'that', 'these', 'those', 'such', 'it', 'they',
                      'yet', 'but', 'however', 'and', 'so', 'thus', 'therefore']

        first_word = sent_lower.split()[0] if sent_lower.split() else ""
        return first_word in connectors


class StructuredGenerator:
    """Generates styled text while preserving paragraph structure."""

    def __init__(self, lora_generator: LoRAStyleGenerator):
        self.lora_generator = lora_generator
        self.analyzer = StructuralAnalyzer()

    def transfer(
        self,
        source_text: str,
        author: str,
        temperature: float = 0.3,
    ) -> str:
        """Transfer style while preserving structure.

        Args:
            source_text: Original paragraph
            author: Target author style
            temperature: Generation temperature

        Returns:
            Styled paragraph with preserved structure
        """
        # Analyze source structure
        structure = self.analyzer.analyze(source_text)

        if structure.sentence_count == 0:
            return source_text

        logger.info(f"Source structure: {structure.sentence_count} sentences, "
                   f"pattern={structure.length_pattern}")

        # Build structured prompt
        prompt = self._build_structured_prompt(structure, author)

        # Generate
        output = self.lora_generator.generate(
            content=prompt,
            author=author,
            target_words=structure.total_words,
            temperature=temperature,
        )

        if not output:
            return source_text

        # Validate and fix structure
        output = self._validate_and_fix_structure(output, structure)

        return output

    def _build_structured_prompt(
        self,
        structure: ParagraphStructure,
        author: str,
    ) -> str:
        """Build a prompt that enforces structure preservation."""
        lines = [
            f"Rewrite this paragraph in {author}'s distinctive voice.",
            "",
            f"CRITICAL: Write exactly {structure.sentence_count} sentences.",
            f"Match this rhythm pattern: {structure.length_pattern}",
            "(S=short 3-6 words, M=medium 7-12 words, L=long 13+ words)",
            "",
            "Write each sentence following this structure:",
            ""
        ]

        for spec in structure.sentences:
            length_desc = "SHORT (3-6 words)" if spec.is_short else f"~{spec.target_words} words"
            move_desc = spec.move.value.upper()

            lines.append(f"  {spec.index + 1}. [{move_desc}] [{length_desc}]")
            lines.append(f"     Content: {spec.original}")
            lines.append("")

        lines.extend([
            "RULES:",
            "- Each numbered item = ONE sentence in output",
            "- SHORT sentences must be punchy and direct (3-6 words only)",
            "- Do NOT combine multiple items into one sentence",
            "- Do NOT add content not in the original",
            "- Each sentence should flow naturally to the next",
            "",
            "Write the paragraph:"
        ])

        return "\n".join(lines)

    def _validate_and_fix_structure(
        self,
        output: str,
        target: ParagraphStructure,
    ) -> str:
        """Validate output structure and fix issues."""
        doc = self.analyzer.nlp(output)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        # Check for run-ons (sentences way too long)
        fixed_sentences = []
        for sent in sentences:
            word_count = len(sent.split())

            # If sentence is much longer than any in target, it's probably a run-on
            max_target = max(s.target_words for s in target.sentences) + 5

            if word_count > max_target + 10:
                # Try to split at conjunctions or semicolons
                parts = self._split_runon(sent)
                fixed_sentences.extend(parts)
            else:
                fixed_sentences.append(sent)

        # If we have too many sentences, we might have over-split
        if len(fixed_sentences) > target.sentence_count + 2:
            # Merge some back together
            fixed_sentences = self._merge_to_target(fixed_sentences, target.sentence_count)

        # Ensure short sentences stay short
        for i, idx in enumerate(target.short_sentence_indices):
            if idx < len(fixed_sentences):
                sent = fixed_sentences[idx]
                if len(sent.split()) > 8:
                    # Truncate or simplify
                    fixed_sentences[idx] = self._shorten_sentence(sent)

        return ' '.join(fixed_sentences)

    def _split_runon(self, sent: str) -> List[str]:
        """Split a run-on sentence at natural break points."""
        # Split at semicolons
        if ';' in sent:
            parts = [p.strip() for p in sent.split(';') if p.strip()]
            if len(parts) > 1:
                # Add periods
                return [p if p.endswith('.') else p + '.' for p in parts]

        # Split at conjunctions followed by independent clauses
        # Pattern: ", and/but/yet + subject + verb"
        patterns = [
            r',\s*(and|but|yet|so)\s+([A-Z])',  # Comma + conjunction + capital
            r'—\s*([A-Z])',  # Em-dash + capital
            r'\s+-\s*([A-Z])',  # Spaced dash + capital
        ]

        for pattern in patterns:
            if re.search(pattern, sent):
                parts = re.split(pattern, sent)
                if len(parts) > 1:
                    # Reconstruct sentences
                    result = []
                    current = parts[0].strip()
                    for i in range(1, len(parts)):
                        p = parts[i].strip()
                        if p and p[0].isupper() and len(p) > 3:
                            if current:
                                result.append(current.rstrip(',') + '.')
                            current = p
                        elif p:
                            current += ' ' + p
                    if current:
                        result.append(current if current.endswith('.') else current + '.')
                    if result:
                        return result

        # Couldn't split - return as is
        return [sent]

    def _merge_to_target(self, sentences: List[str], target_count: int) -> List[str]:
        """Merge sentences to approach target count."""
        while len(sentences) > target_count + 1:
            # Find shortest adjacent pair to merge
            min_combined = float('inf')
            merge_idx = 0

            for i in range(len(sentences) - 1):
                combined_len = len(sentences[i].split()) + len(sentences[i+1].split())
                if combined_len < min_combined:
                    min_combined = combined_len
                    merge_idx = i

            # Merge
            merged = sentences[merge_idx].rstrip('.') + ', ' + sentences[merge_idx + 1].lower()
            sentences = sentences[:merge_idx] + [merged] + sentences[merge_idx + 2:]

        return sentences

    def _shorten_sentence(self, sent: str) -> str:
        """Shorten a sentence that should be punchy."""
        words = sent.split()

        # If it has a dash or comma, take first part
        if '—' in sent:
            return sent.split('—')[0].strip() + '.'
        if ', ' in sent:
            first_part = sent.split(', ')[0]
            if len(first_part.split()) <= 6:
                return first_part + '.'

        # Just take first 5-6 words
        if len(words) > 6:
            short = ' '.join(words[:5])
            return short.rstrip('.,') + '.'

        return sent


class StructurePreservingTransfer:
    """Main class for structure-preserving style transfer.

    This approach prioritizes maintaining the narrative rhythm and flow
    of the source paragraph while applying stylistic transformation.
    """

    def __init__(
        self,
        lora_generator: LoRAStyleGenerator,
        author: str,
    ):
        self.generator = StructuredGenerator(lora_generator)
        self.author = author

    def transfer(
        self,
        source_text: str,
        temperature: float = 0.3,
    ) -> Tuple[str, bool]:
        """Transfer style while preserving structure.

        Args:
            source_text: Original paragraph
            temperature: Generation temperature

        Returns:
            Tuple of (styled_text, success)
        """
        output = self.generator.transfer(
            source_text=source_text,
            author=self.author,
            temperature=temperature,
        )

        # Basic validation
        source_sentences = len(re.split(r'[.!?]+', source_text))
        output_sentences = len(re.split(r'[.!?]+', output))

        # Allow some variance in sentence count
        success = abs(source_sentences - output_sentences) <= 3

        return output, success


def transfer_with_structure(
    source_text: str,
    lora_generator: LoRAStyleGenerator,
    author: str,
    temperature: float = 0.3,
) -> Tuple[str, bool]:
    """Convenience function for structure-preserving transfer.

    Args:
        source_text: Original paragraph
        lora_generator: LoRA generator instance
        author: Target author style
        temperature: Generation temperature

    Returns:
        Tuple of (styled_text, success)
    """
    transferer = StructurePreservingTransfer(lora_generator, author)
    return transferer.transfer(source_text, temperature)
