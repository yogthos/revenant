"""
Cadence Analyzer Module

Analyzes sample text to extract paragraph cadence patterns including:
- Sequence patterns (e.g., short-medium-long sequences)
- Position-based patterns (early/middle/late paragraph lengths)
- Semantic density (claims/concepts per paragraph)
- Role-based patterns (section_openers vs body vs closers)
"""

import spacy
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import numpy as np

from semantic_extractor import SemanticExtractor, SemanticContent


@dataclass
class ParagraphTarget:
    """Target structure for a paragraph at a given position."""
    target_length: int  # Target word count
    target_sentence_count: int  # Target sentence count
    position: float  # Position in document (0.0-1.0)
    role: str  # section_opener, paragraph_opener, body, closer
    semantic_density: float  # Expected claims per paragraph


@dataclass
class CadenceProfile:
    """Complete cadence profile from sample text."""
    sequence_patterns: List[List[int]]  # Sequences of word counts
    position_based_lengths: Dict[str, int]  # Segment -> avg length
    semantic_density_patterns: Dict[str, float]  # Role -> claims per para
    role_based_lengths: Dict[str, Dict[str, float]]  # Role -> length stats
    paragraph_count: int
    avg_paragraph_length: float
    position_segments: int  # Number of segments used for position analysis


class CadenceAnalyzer:
    """
    Analyzes sample text to extract paragraph cadence patterns.

    Identifies:
    - Recurring paragraph length sequences
    - Position-based length patterns
    - Semantic density by role
    - Role-based length distributions
    """

    def __init__(self, sample_text: str, semantic_extractor: SemanticExtractor,
                 sequence_window_size: int = 5, position_segments: int = 4):
        """
        Initialize cadence analyzer.

        Args:
            sample_text: The sample text to analyze
            semantic_extractor: SemanticExtractor instance for analyzing semantic density
            sequence_window_size: Size of sliding window for sequence detection
            position_segments: Number of position segments (e.g., 4 = quartiles)
        """
        self.sample_text = sample_text
        self.semantic_extractor = semantic_extractor
        self.sequence_window_size = sequence_window_size
        self.position_segments = position_segments

        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

        # Extract semantic content from sample for density analysis
        self.sample_semantics: Optional[SemanticContent] = None
        self._extract_sample_semantics()

    def _extract_sample_semantics(self):
        """Extract semantic content from sample text."""
        try:
            self.sample_semantics = self.semantic_extractor.extract(self.sample_text)
        except Exception as e:
            print(f"  [CadenceAnalyzer] Warning: Could not extract semantics: {e}")
            self.sample_semantics = None

    def analyze_cadence(self) -> CadenceProfile:
        """
        Analyze sample text and extract all cadence patterns.

        Returns:
            CadenceProfile with all detected patterns
        """
        paragraphs = [p.strip() for p in self.sample_text.split('\n\n') if p.strip()]

        if not paragraphs:
            # Return default profile if no paragraphs
            return CadenceProfile(
                sequence_patterns=[],
                position_based_lengths={},
                semantic_density_patterns={},
                role_based_lengths={},
                paragraph_count=0,
                avg_paragraph_length=0,
                position_segments=self.position_segments
            )

        # Extract paragraph word counts and roles
        para_lengths = []
        para_roles = []
        substantial_paras = []

        for i, para in enumerate(paragraphs):
            word_count = len(para.split())
            # Only analyze substantial paragraphs (skip headers)
            if word_count >= 20:
                para_lengths.append(word_count)
                position_ratio = i / max(len(paragraphs) - 1, 1)
                role = self._determine_role(i, len(paragraphs), para)
                para_roles.append(role)
                substantial_paras.append((para, position_ratio, role, word_count))

        if not substantial_paras:
            # Return default profile if no substantial paragraphs
            return CadenceProfile(
                sequence_patterns=[],
                position_based_lengths={},
                semantic_density_patterns={},
                role_based_lengths={},
                paragraph_count=0,
                avg_paragraph_length=0,
                position_segments=self.position_segments
            )

        # Analyze sequence patterns
        sequence_patterns = self._detect_sequence_patterns(para_lengths)

        # Analyze position-based patterns
        position_based_lengths = self._analyze_position_based_patterns(substantial_paras)

        # Analyze semantic density
        semantic_density_patterns = self._analyze_semantic_density(substantial_paras)

        # Analyze role-based lengths
        role_based_lengths = self._analyze_role_based_lengths(substantial_paras)

        avg_length = np.mean(para_lengths) if para_lengths else 0

        return CadenceProfile(
            sequence_patterns=sequence_patterns,
            position_based_lengths=position_based_lengths,
            semantic_density_patterns=semantic_density_patterns,
            role_based_lengths=role_based_lengths,
            paragraph_count=len(substantial_paras),
            avg_paragraph_length=avg_length,
            position_segments=self.position_segments
        )

    def _determine_role(self, index: int, total: int, para: str) -> str:
        """Determine structural role of a paragraph."""
        # Check if section start
        if self._is_section_start(para):
            return 'section_opener'

        # Check position
        if index == 0:
            return 'section_opener'
        elif index >= total - 2:  # Last 2 paragraphs
            return 'closer'
        elif index < 3:  # First few paragraphs
            return 'paragraph_opener'
        else:
            return 'body'

    def _is_section_start(self, para: str) -> bool:
        """Check if paragraph starts a new section."""
        first_line = para.split('\n')[0].strip()
        section_patterns = [
            r'^[0-9]+\)',
            r'^[a-z]\)',
            r'^\d+\.',
            r'^[IVX]+\.',
            r'^#+\s',
        ]
        return any(re.match(p, first_line) for p in section_patterns)

    def _detect_sequence_patterns(self, para_lengths: List[int]) -> List[List[int]]:
        """
        Detect recurring sequence patterns in paragraph lengths.

        Uses sliding window to find common subsequences.
        """
        if len(para_lengths) < self.sequence_window_size:
            return []

        # Normalize lengths to categories (short/medium/long)
        if not para_lengths:
            return []

        q1, q2, q3 = np.percentile(para_lengths, [25, 50, 75])

        def categorize(length):
            if length <= q1:
                return 'short'
            elif length <= q3:
                return 'medium'
            else:
                return 'long'

        # Find sequences using sliding window
        sequences = []
        for i in range(len(para_lengths) - self.sequence_window_size + 1):
            window = para_lengths[i:i + self.sequence_window_size]
            sequences.append(window)

        # Find most common sequences (at least 2 occurrences)
        sequence_counts = Counter(tuple(seq) for seq in sequences)
        common_sequences = [list(seq) for seq, count in sequence_counts.items()
                           if count >= 2]

        # Also return normalized sequences
        normalized_sequences = []
        for seq in sequences[:5]:  # Limit to first 5 for variety
            normalized_sequences.append(seq)

        return normalized_sequences[:10]  # Return top 10 patterns

    def _analyze_position_based_patterns(self, substantial_paras: List[Tuple]) -> Dict[str, int]:
        """
        Analyze paragraph lengths by position in document.

        Divides document into segments and calculates average length per segment.
        """
        if not substantial_paras:
            return {}

        position_lengths = defaultdict(list)

        for para, position_ratio, role, word_count in substantial_paras:
            # Determine segment (0-based)
            segment = int(position_ratio * self.position_segments)
            segment = min(segment, self.position_segments - 1)
            segment_name = f"segment_{segment}"
            position_lengths[segment_name].append(word_count)

        # Calculate averages
        position_based = {}
        for segment, lengths in position_lengths.items():
            position_based[segment] = int(np.mean(lengths))

        return position_based

    def _analyze_semantic_density(self, substantial_paras: List[Tuple]) -> Dict[str, float]:
        """
        Analyze semantic density (claims per paragraph) by role.

        Maps semantic claims to paragraphs and counts density.
        """
        if not self.sample_semantics or not self.sample_semantics.claims:
            # Return default densities if no semantics
            return {
                'section_opener': 2.0,
                'paragraph_opener': 1.5,
                'body': 1.0,
                'closer': 1.5
            }

        # Map claims to paragraphs (simplified: use paragraph structure if available)
        role_claims = defaultdict(list)

        # Use paragraph structure from semantic extraction if available
        if self.sample_semantics.paragraph_structure:
            for i, para_info in enumerate(self.sample_semantics.paragraph_structure):
                if i < len(substantial_paras):
                    _, _, role, _ = substantial_paras[i]
                    # Estimate claims for this paragraph (use sentence count as proxy)
                    sentence_count = para_info.get('sentence_count', 1)
                    # Rough estimate: 1 claim per 2-3 sentences
                    estimated_claims = max(1, sentence_count // 2)
                    role_claims[role].append(estimated_claims)
        else:
            # Fallback: distribute claims evenly by role
            total_claims = len(self.sample_semantics.claims)
            role_counts = Counter(role for _, _, role, _ in substantial_paras)
            for role, count in role_counts.items():
                avg_claims = total_claims / len(substantial_paras) if substantial_paras else 1.0
                role_claims[role].append(avg_claims)

        # Calculate average density per role
        density_patterns = {}
        for role in ['section_opener', 'paragraph_opener', 'body', 'closer']:
            if role_claims[role]:
                density_patterns[role] = np.mean(role_claims[role])
            else:
                # Default values
                density_patterns[role] = {
                    'section_opener': 2.0,
                    'paragraph_opener': 1.5,
                    'body': 1.0,
                    'closer': 1.5
                }.get(role, 1.0)

        return density_patterns

    def _analyze_role_based_lengths(self, substantial_paras: List[Tuple]) -> Dict[str, Dict[str, float]]:
        """
        Analyze length statistics by role.

        Returns mean, median, std for each role.
        """
        role_lengths = defaultdict(list)

        for _, _, role, word_count in substantial_paras:
            role_lengths[role].append(word_count)

        role_stats = {}
        for role in ['section_opener', 'paragraph_opener', 'body', 'closer']:
            if role_lengths[role]:
                lengths = role_lengths[role]
                role_stats[role] = {
                    'mean': float(np.mean(lengths)),
                    'median': float(np.median(lengths)),
                    'std': float(np.std(lengths)) if len(lengths) > 1 else 0.0,
                    'min': float(np.min(lengths)),
                    'max': float(np.max(lengths))
                }
            else:
                # Default stats if no examples
                role_stats[role] = {
                    'mean': 150.0,
                    'median': 150.0,
                    'std': 50.0,
                    'min': 100.0,
                    'max': 200.0
                }

        return role_stats

    def get_target_paragraph_structure(self, position: float, role: str,
                                       cadence_profile: CadenceProfile) -> ParagraphTarget:
        """
        Get target paragraph structure for a given position and role.

        Args:
            position: Position in document (0.0-1.0)
            role: Structural role (section_opener, paragraph_opener, body, closer)
            cadence_profile: The cadence profile to use

        Returns:
            ParagraphTarget with target structure
        """
        # Determine segment
        segment = int(position * cadence_profile.position_segments)
        segment = min(segment, cadence_profile.position_segments - 1)
        segment_name = f"segment_{segment}"

        # Get target length from position-based or role-based
        if segment_name in cadence_profile.position_based_lengths:
            target_length = cadence_profile.position_based_lengths[segment_name]
        elif role in cadence_profile.role_based_lengths:
            target_length = int(cadence_profile.role_based_lengths[role]['mean'])
        else:
            target_length = int(cadence_profile.avg_paragraph_length)

        # Estimate sentence count (rough: 15-20 words per sentence)
        target_sentence_count = max(1, int(target_length / 17))

        # Get semantic density
        semantic_density = cadence_profile.semantic_density_patterns.get(role, 1.0)

        return ParagraphTarget(
            target_length=target_length,
            target_sentence_count=target_sentence_count,
            position=position,
            role=role,
            semantic_density=semantic_density
        )


# Test function
if __name__ == '__main__':
    from pathlib import Path
    from semantic_extractor import SemanticExtractor

    sample_path = Path(__file__).parent / "prompts" / "sample_mao.txt"
    if sample_path.exists():
        with open(sample_path, 'r', encoding='utf-8') as f:
            sample_text = f.read()

        print("=== Cadence Analyzer Test ===\n")

        extractor = SemanticExtractor()
        analyzer = CadenceAnalyzer(sample_text, extractor)

        profile = analyzer.analyze_cadence()

        print(f"Paragraph count: {profile.paragraph_count}")
        print(f"Average paragraph length: {profile.avg_paragraph_length:.1f} words")
        print(f"\nSequence patterns (first 3):")
        for i, seq in enumerate(profile.sequence_patterns[:3], 1):
            print(f"  {i}: {seq}")

        print(f"\nPosition-based lengths:")
        for segment, length in sorted(profile.position_based_lengths.items()):
            print(f"  {segment}: {length} words")

        print(f"\nSemantic density by role:")
        for role, density in profile.semantic_density_patterns.items():
            print(f"  {role}: {density:.2f} claims/paragraph")

        print(f"\nRole-based length stats:")
        for role, stats in profile.role_based_lengths.items():
            print(f"  {role}: mean={stats['mean']:.1f}, std={stats['std']:.1f}")

        # Test target structure
        print(f"\n\nExample target structures:")
        for pos in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for role in ['section_opener', 'body', 'closer']:
                target = analyzer.get_target_paragraph_structure(pos, role, profile)
                print(f"  Position {pos:.0%}, {role}: {target.target_length} words, "
                      f"{target.target_sentence_count} sentences, "
                      f"{target.semantic_density:.1f} claims/para")
    else:
        print("No sample file found.")

