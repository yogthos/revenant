"""Triplet extraction using spaCy dependency parsing.

Extracts (subject, relation, object) triplets from text using
spaCy's dependency parser for entity-role extraction.
"""

from dataclasses import dataclass
from typing import List, Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Triplet:
    """A subject-relation-object triplet."""
    subject: str
    relation: str
    object: str
    confidence: float = 1.0

    def __str__(self) -> str:
        return f"({self.subject} | {self.relation} | {self.object})"

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "confidence": self.confidence,
        }


class SpaCyTripletExtractor:
    """Triplet extractor using spaCy dependency parsing."""

    def __init__(self):
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            from ..utils.nlp import get_nlp
            self._nlp = get_nlp()
        return self._nlp

    def extract(self, text: str) -> List[Triplet]:
        """Extract triplets using spaCy dependency parsing.

        Args:
            text: Input text to extract triplets from.

        Returns:
            List of Triplet objects.
        """
        if not text or not text.strip():
            return []

        triplets = []
        doc = self.nlp(text)

        for sent in doc.sents:
            # Find main verb
            verb = None
            for token in sent:
                if token.pos_ == "VERB" and token.dep_ in ("ROOT", "relcl", "advcl"):
                    verb = token
                    break

            if not verb:
                continue

            # Find subject
            subject = None
            for child in verb.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subject = self._get_span_text(child)
                    break

            # Find object
            obj = None
            for child in verb.children:
                if child.dep_ in ("dobj", "pobj", "attr", "oprd"):
                    obj = self._get_span_text(child)
                    break

            # Also check for prepositional objects
            if not obj:
                for child in verb.children:
                    if child.dep_ == "prep":
                        for grandchild in child.children:
                            if grandchild.dep_ == "pobj":
                                obj = self._get_span_text(grandchild)
                                break

            if subject and obj:
                triplets.append(Triplet(
                    subject=subject,
                    relation=verb.lemma_,
                    object=obj,
                    confidence=0.8,
                ))

        return triplets

    def _get_span_text(self, token) -> str:
        """Get the full noun phrase for a token."""
        # Get subtree for compound nouns and modifiers
        tokens = []
        for t in token.subtree:
            if t.dep_ not in ("punct", "cc", "conj", "prep"):
                tokens.append(t.text)
            if len(tokens) > 8:  # Limit length
                break
        return " ".join(tokens)


# Global extractor instance (singleton)
_extractor: Optional[SpaCyTripletExtractor] = None


def get_triplet_extractor() -> SpaCyTripletExtractor:
    """Get the triplet extractor (singleton).

    Returns:
        SpaCyTripletExtractor instance.
    """
    global _extractor

    if _extractor is None:
        _extractor = SpaCyTripletExtractor()
    return _extractor


def extract_triplets(text: str) -> List[Triplet]:
    """Convenience function to extract triplets from text.

    Args:
        text: Input text.

    Returns:
        List of triplets.
    """
    extractor = get_triplet_extractor()
    return extractor.extract(text)
