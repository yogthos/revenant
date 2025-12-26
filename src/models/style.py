"""Style-related data models."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class AuthorProfile:
    """Profile for a single author's writing style."""
    name: str
    style_dna: str  # LLM-generated description of style
    top_vocab: List[str] = field(default_factory=list)  # Top 50 characteristic words
    avg_sentence_length: float = 15.0
    burstiness: float = 0.3  # Coefficient of variation of sentence lengths
    punctuation_freq: Dict[str, float] = field(default_factory=dict)
    perspective: str = "third_person"  # first_person_singular, first_person_plural, third_person

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "style_dna": self.style_dna,
            "top_vocab": self.top_vocab,
            "avg_sentence_length": self.avg_sentence_length,
            "burstiness": self.burstiness,
            "punctuation_freq": self.punctuation_freq,
            "perspective": self.perspective
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "AuthorProfile":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            style_dna=data.get("style_dna", ""),
            top_vocab=data.get("top_vocab", []),
            avg_sentence_length=data.get("avg_sentence_length", 15.0),
            burstiness=data.get("burstiness", 0.3),
            punctuation_freq=data.get("punctuation_freq", {}),
            perspective=data.get("perspective", "third_person")
        )


@dataclass
class StyleProfile:
    """Style profile that can represent single author or blended style.

    Designed to support future multi-author blending while currently
    operating in single-author mode.
    """
    primary_author: AuthorProfile
    secondary_author: Optional[AuthorProfile] = None
    blend_ratio: float = 1.0  # 1.0 = 100% primary, 0.5 = 50/50 blend

    def get_effective_style_dna(self) -> str:
        """Get the effective style DNA, handling blending if applicable."""
        if self.secondary_author is None or self.blend_ratio >= 1.0:
            return self.primary_author.style_dna
        # Future: implement style DNA blending
        return self._blend_style_dnas()

    def get_effective_vocab(self) -> List[str]:
        """Get the effective vocabulary, handling blending if applicable."""
        if self.secondary_author is None or self.blend_ratio >= 1.0:
            return self.primary_author.top_vocab
        # Future: implement vocabulary blending
        return self._blend_vocabularies()

    def get_effective_avg_sentence_length(self) -> float:
        """Get the effective average sentence length."""
        if self.secondary_author is None or self.blend_ratio >= 1.0:
            return self.primary_author.avg_sentence_length
        # Weighted average for blending
        return (
            self.blend_ratio * self.primary_author.avg_sentence_length +
            (1 - self.blend_ratio) * self.secondary_author.avg_sentence_length
        )

    def get_effective_burstiness(self) -> float:
        """Get the effective burstiness."""
        if self.secondary_author is None or self.blend_ratio >= 1.0:
            return self.primary_author.burstiness
        # Weighted average for blending
        return (
            self.blend_ratio * self.primary_author.burstiness +
            (1 - self.blend_ratio) * self.secondary_author.burstiness
        )

    def get_author_name(self) -> str:
        """Get the primary author name for display/logging."""
        return self.primary_author.name

    def _blend_style_dnas(self) -> str:
        """Blend style DNAs from two authors. Future implementation."""
        # For now, just return primary
        # Future: Use LLM to generate blended style description
        return self.primary_author.style_dna

    def _blend_vocabularies(self) -> List[str]:
        """Blend vocabularies from two authors. Future implementation."""
        # For now, just return primary
        # Future: Merge with weighting based on blend_ratio
        return self.primary_author.top_vocab

    @classmethod
    def from_author(cls, author: AuthorProfile) -> "StyleProfile":
        """Create a single-author style profile."""
        return cls(primary_author=author)
