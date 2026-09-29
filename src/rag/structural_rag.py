"""Structural RAG - retrieves rhythm and syntax patterns, not content.

Instead of returning text examples that get copied, this returns:
- Rhythm fingerprints (sentence length sequences)
- Punctuation patterns
- Syntactic templates (POS sequences)
- Opening/transition patterns
- Vocabulary clusters (author-specific word banks)
- Transition inventory (author's connectives)
- Emotional stance markers

This addresses the core issues:
- Mechanical Transitions → Natural transition rhythm from author
- Impersonal Tone → Vocabulary clusters + emotional stance markers
- Mechanical Precision → Concrete syntactic templates (not just length categories)
- Formulaic Flow → Actual author rhythm fingerprints
- Robotic Formality → Fragment usage, informal punctuation
- Sophisticated Clarity → Transition inventory + opening patterns
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
import random

from .structural_analyzer import (
    RhythmFingerprint,
    get_structural_analyzer,
)
from .enhanced_analyzer import (
    EnhancedStyleProfile,
    get_enhanced_analyzer,
)
from .corpus_indexer import get_indexer
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StructuralGuidance:
    """Structural guidance for prompt injection."""
    rhythm_pattern: str  # e.g., "LONG → SHORT → MEDIUM → FRAGMENT → LONG"
    punctuation_hints: List[str]  # e.g., ["use dashes", "semicolons for lists"]
    length_guidance: str  # e.g., "Vary between 5 and 35 words per sentence"
    fragment_hint: str  # e.g., "Include 1-2 sentence fragments"
    opening_hint: str  # e.g., "Start with: determiner, noun, or adverb"
    enhanced_profile: Optional[EnhancedStyleProfile] = None  # Enhanced patterns
    exemplar_sentences: Optional[List[str]] = None  # Actual sentences from author's corpus

    def format_for_prompt(self) -> str:
        """Format as concise prompt guidance matching training format.

        Training used simple, direct guidance without verbose headers.
        Keep this minimal - just the key structural hints.
        """
        lines = []

        # Concise structural patterns (matching training simplicity)
        if self.rhythm_pattern:
            lines.append(f"Rhythm: {self.rhythm_pattern}")

        if self.length_guidance:
            lines.append(f"Length: {self.length_guidance}")

        if self.fragment_hint:
            lines.append(f"Fragments: {self.fragment_hint}")

        if self.punctuation_hints:
            hints = "; ".join(self.punctuation_hints[:3])  # Limit to 3
            lines.append(f"Punctuation: {hints}")

        if self.opening_hint:
            lines.append(f"Openings: {self.opening_hint}")

        return "\n".join(lines)


class StructuralRAG:
    """Retrieves structural patterns from author corpus."""

    def __init__(self, author: str):
        self.author = author
        self.analyzer = get_structural_analyzer()
        self.enhanced_analyzer = get_enhanced_analyzer()
        self.indexer = get_indexer()
        self._cached_rhythms: List[RhythmFingerprint] = []
        self._enhanced_profile: Optional[EnhancedStyleProfile] = None
        self._loaded = False

    def load_patterns(self, sample_size: int = 50) -> int:
        """Load rhythm patterns from author's indexed corpus."""
        if self._loaded:
            return len(self._cached_rhythms)

        # Get chunks from indexer
        try:
            chunks = self.indexer.get_random_chunks(self.author, n=sample_size)
        except Exception as e:
            logger.warning(f"Could not load chunks for {self.author}: {e}")
            return 0

        # Analyze each chunk for rhythm
        for chunk_text in chunks:
            try:
                rhythm = self.analyzer.extract_rhythm(chunk_text)
                if rhythm.sentence_count >= 2:  # Need at least 2 sentences
                    self._cached_rhythms.append(rhythm)
            except Exception as e:
                logger.warning(f"Could not analyze chunk: {e}")
                continue

        # Run enhanced analysis on all chunks
        if chunks:
            try:
                self._enhanced_profile = self.enhanced_analyzer.analyze(chunks)
                logger.info(f"Enhanced analysis complete: {len(self._enhanced_profile.syntactic_templates)} templates")
            except Exception as e:
                logger.warning(f"Enhanced analysis failed: {e}")
                self._enhanced_profile = None

        self._loaded = True
        logger.info(f"Loaded {len(self._cached_rhythms)} rhythm patterns for {self.author}")
        return len(self._cached_rhythms)

    def get_rhythm_pattern(self, target_sentences: int = 4) -> str:
        """Get a rhythm pattern matching target sentence count."""
        if not self._cached_rhythms:
            self.load_patterns()

        if not self._cached_rhythms:
            # Fallback: generate default rhythm pattern
            logger.debug(f"No cached rhythms for {self.author}, using default fallback")
            return self._generate_default_rhythm(target_sentences)

        # Find patterns with similar sentence count
        matching = [r for r in self._cached_rhythms
                   if abs(r.sentence_count - target_sentences) <= 2]

        if not matching:
            matching = self._cached_rhythms

        # Pick random matching pattern
        rhythm = random.choice(matching)
        return rhythm.to_rhythm_string()

    def _generate_default_rhythm(self, n: int) -> str:
        """Generate default fallback rhythm pattern."""
        # Generic characteristic patterns:
        # - Long complex sentences followed by short punchy ones
        # - Occasional fragments for emphasis
        # - Building tension with increasing length
        patterns = [
            ["LONG", "SHORT", "MEDIUM", "FRAGMENT", "LONG"],
            ["MEDIUM", "LONG", "SHORT", "LONG", "FRAGMENT"],
            ["SHORT", "LONG", "MEDIUM", "SHORT", "VERY_LONG"],
            ["LONG", "FRAGMENT", "LONG", "SHORT", "MEDIUM"],
            ["MEDIUM", "SHORT", "LONG", "FRAGMENT", "LONG"],
        ]
        pattern = random.choice(patterns)[:n]
        return " → ".join(pattern)

    def get_punctuation_hints(self) -> List[str]:
        """Get punctuation usage hints from corpus analysis."""
        if not self._cached_rhythms:
            self.load_patterns()

        hints = []

        if not self._cached_rhythms:
            return ["vary punctuation for rhythm"]

        # Analyze cached rhythms for punctuation patterns
        dash_count = sum(1 for r in self._cached_rhythms if r.punctuation_density > 0.3)
        if dash_count > len(self._cached_rhythms) * 0.3:
            hints.append("use em-dashes (—) for interruptions and asides")

        # Fragment guidance is handled by get_fragment_hint() — not duplicated here

        return hints if hints else ["vary punctuation for rhythm"]

    def get_length_guidance(self) -> str:
        """Get sentence length guidance from corpus."""
        if not self._cached_rhythms:
            self.load_patterns()

        if not self._cached_rhythms:
            return "Vary between 5 and 40 words per sentence"

        # Calculate range from corpus
        all_avgs = [r.avg_sentence_length for r in self._cached_rhythms]
        all_vars = [r.length_variance for r in self._cached_rhythms]

        avg_len = sum(all_avgs) / len(all_avgs)
        avg_var = sum(all_vars) / len(all_vars)

        min_len = max(3, int(avg_len - avg_var**0.5))
        max_len = int(avg_len + avg_var**0.5)

        if avg_var > 100:
            return f"HIGH variation: {min_len}-{max_len} words, mix short punchy with long flowing"
        elif avg_var > 50:
            return f"MODERATE variation: {min_len}-{max_len} words per sentence"
        else:
            return f"Sentences around {int(avg_len)} words, some variation"

    def get_fragment_hint(self) -> str:
        """Get fragment usage hint."""
        if not self._cached_rhythms:
            self.load_patterns()

        if not self._cached_rhythms:
            return "Include 1-2 sentence fragments for emphasis"

        avg_ratio = sum(r.fragment_ratio for r in self._cached_rhythms) / len(self._cached_rhythms)

        if avg_ratio > 0.2:
            return f"Use fragments liberally (~{int(avg_ratio*100)}% of sentences)"
        elif avg_ratio > 0.1:
            return "Include occasional fragments for punch"
        else:
            return "Use complete sentences primarily"

    def get_guidance(self, input_text: str) -> StructuralGuidance:
        """Get complete structural guidance for input text."""
        # Ensure patterns are loaded before computing opening_hint
        self.load_patterns()

        # Analyze input to match sentence count
        input_rhythm = self.analyzer.extract_rhythm(input_text)
        target_sentences = max(3, input_rhythm.sentence_count)

        # Build opening hint from enhanced profile if available
        opening_hint = ""
        if self._enhanced_profile and self._enhanced_profile.openings.patterns:
            top_patterns = list(self._enhanced_profile.openings.patterns.keys())[:3]
            if top_patterns:
                opening_hint = f"Vary: {', '.join(top_patterns)}"

        return StructuralGuidance(
            rhythm_pattern=self.get_rhythm_pattern(target_sentences),
            punctuation_hints=self.get_punctuation_hints(),
            length_guidance=self.get_length_guidance(),
            fragment_hint=self.get_fragment_hint(),
            opening_hint=opening_hint,
            enhanced_profile=self._enhanced_profile,
        )

    def _get_exemplar_sentences(self, input_text: str, n: int = 5) -> List[str]:
        """Get exemplar sentences from corpus that demonstrate the author's voice.

        These are the MOST IMPORTANT part of style guidance. The model needs to
        SEE how the author actually writes, not just receive abstract rules.

        Returns sentences that:
        1. Have interesting rhythm (not too short, not too long)
        2. Show characteristic vocabulary and syntax
        3. Demonstrate emotional engagement
        """
        try:
            # Get relevant chunks from corpus based on input content
            results = self.indexer.retrieve_similar(self.author, input_text, n=10)
            chunks = [r["text"] for r in results]
        except Exception as e:
            logger.warning(f"Could not query similar chunks: {e}")
            chunks = []

        # Fall back to random chunks if semantic search returned nothing
        if not chunks:
            try:
                random_chunks = self.indexer.get_random_chunks(self.author, n=10)
                if random_chunks:
                    logger.info(f"Using random chunks as fallback for exemplar sentences")
                    chunks = random_chunks
            except Exception as e:
                logger.warning(f"Could not get random chunks: {e}")

        if not chunks:
            logger.warning(f"No exemplar sentences found for author '{self.author}' — corpus may be empty or unindexed")
            return []

        # Extract individual sentences from chunks
        from ..utils.nlp import split_into_sentences
        all_sentences = []
        for chunk in chunks:
            sentences = split_into_sentences(chunk)
            all_sentences.extend(sentences)

        # Build emotion word list dynamically from enhanced profile
        emotion_words = []
        if self._enhanced_profile and self._enhanced_profile.vocabulary:
            emotion_words = list(self._enhanced_profile.vocabulary.emotional)

        # Filter for high-quality exemplar sentences
        exemplars = []
        for sent in all_sentences:
            words = sent.split()
            # Good length: 10-40 words (interesting but not overwhelming)
            if 10 <= len(words) <= 40:
                # Structural features (author-agnostic)
                has_dash = '—' in sent or ' – ' in sent
                has_semicolon = ';' in sent
                # Dynamic emotion detection from corpus vocabulary
                has_emotion = bool(emotion_words) and any(
                    w in sent.lower() for w in emotion_words
                )
                # Sentence length variety is itself a signal
                word_count = len(words)
                has_length_variety = word_count <= 15 or word_count >= 30
                # Prioritize sentences with distinctive features
                score = (int(has_dash) + int(has_semicolon)
                         + int(has_emotion) * 2 + int(has_length_variety))
                if score > 0 or len(exemplars) < 3:  # Always get at least 3
                    exemplars.append((score, sent))

        # Sort by score and return top N
        exemplars.sort(key=lambda x: -x[0])
        return [sent for _, sent in exemplars[:n]]


# Cache key is (author, id(services)) so different Services containers get
# distinct RAGs — otherwise a test container would inherit the process-wide
# default's analyzers from an earlier call.
_rag_cache: Dict[tuple, StructuralRAG] = {}


def get_structural_rag(author: str) -> StructuralRAG:
    """Get or create structural RAG for author.

    The cache is partitioned by the active Services container so an injected
    container does not see a StructuralRAG built against the process-wide
    default (which would bypass DI).
    """
    from ..services import get_default_services
    services = get_default_services()
    key = (author, id(services))
    if key not in _rag_cache:
        _rag_cache[key] = StructuralRAG(author)
    return _rag_cache[key]


def get_structural_guidance(author: str, input_text: str) -> str:
    """Convenience function to get formatted structural guidance."""
    rag = get_structural_rag(author)
    guidance = rag.get_guidance(input_text)
    return guidance.format_for_prompt()
