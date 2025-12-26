"""Relationship detection between propositions."""

import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

from ..models.graph import PropositionNode, RelationshipEdge, RelationshipType
from ..llm.provider import LLMProvider
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DetectedRelationship:
    """A detected relationship with metadata."""
    source_idx: int
    target_idx: int
    relationship: RelationshipType
    confidence: float
    evidence: str  # The marker or reason for detection


# Relationship markers organized by type
RELATIONSHIP_MARKERS = {
    RelationshipType.CAUSES: [
        "because", "therefore", "thus", "hence", "consequently",
        "as a result", "due to", "since", "so that", "leads to",
        "causes", "results in", "for this reason", "accordingly",
    ],
    RelationshipType.CONTRASTS: [
        "but", "however", "although", "despite", "yet",
        "on the other hand", "in contrast", "nevertheless",
        "whereas", "while", "conversely", "instead",
        "on the contrary", "nonetheless", "even though",
    ],
    RelationshipType.ELABORATES: [
        "for example", "for instance", "such as", "namely",
        "specifically", "in particular", "that is", "i.e.",
        "to illustrate", "in other words", "more specifically",
    ],
    RelationshipType.REFERENCES: [
        "according to", "as mentioned", "as noted",
        "as stated", "refers to", "regarding",
        "with respect to", "concerning",
    ],
}


class RelationshipDetector:
    """Detects relationships between propositions using markers and LLM fallback.

    Uses a two-stage approach:
    1. Rule-based detection using discourse markers
    2. LLM-based detection for ambiguous cases (optional)
    """

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        use_llm_fallback: bool = True,
        min_confidence: float = 0.5
    ):
        """Initialize detector.

        Args:
            llm_provider: Optional LLM provider for fallback detection.
            use_llm_fallback: Whether to use LLM for ambiguous cases.
            min_confidence: Minimum confidence for relationships.
        """
        self.llm_provider = llm_provider
        self.use_llm_fallback = use_llm_fallback and llm_provider is not None
        self.min_confidence = min_confidence

        # Compile marker patterns
        self._marker_patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[RelationshipType, re.Pattern]:
        """Compile regex patterns for each relationship type."""
        patterns = {}
        for rel_type, markers in RELATIONSHIP_MARKERS.items():
            # Create pattern that matches any marker as a word boundary
            marker_pattern = "|".join(
                r"\b" + re.escape(m) + r"\b"
                for m in markers
            )
            patterns[rel_type] = re.compile(marker_pattern, re.IGNORECASE)
        return patterns

    def detect_relationships(
        self,
        propositions: List[PropositionNode]
    ) -> List[RelationshipEdge]:
        """Detect relationships between propositions.

        Args:
            propositions: List of propositions to analyze.

        Returns:
            List of detected relationship edges.
        """
        if len(propositions) < 2:
            return []

        relationships = []

        # Stage 1: Rule-based detection
        rule_based = self._detect_rule_based(propositions)
        relationships.extend(rule_based)

        # Stage 2: Add sequential FOLLOWS relationships for adjacent propositions
        # that don't have another relationship
        connected_pairs = set()
        for edge in relationships:
            connected_pairs.add((edge.source_id, edge.target_id))

        for i in range(len(propositions) - 1):
            prop1 = propositions[i]
            prop2 = propositions[i + 1]

            # Check if already connected
            if (prop1.id, prop2.id) not in connected_pairs:
                relationships.append(RelationshipEdge(
                    source_id=prop1.id,
                    target_id=prop2.id,
                    relationship=RelationshipType.FOLLOWS,
                    confidence=0.8
                ))

        # Stage 3: LLM fallback for complex patterns (optional)
        if self.use_llm_fallback:
            llm_detected = self._detect_with_llm(propositions, relationships)
            relationships.extend(llm_detected)

        logger.debug(f"Detected {len(relationships)} relationships")
        return relationships

    def _detect_rule_based(
        self,
        propositions: List[PropositionNode]
    ) -> List[DetectedRelationship]:
        """Detect relationships using discourse markers.

        Args:
            propositions: List of propositions.

        Returns:
            List of detected relationships.
        """
        relationships = []

        for i, prop in enumerate(propositions):
            text_lower = prop.text.lower()

            # Check each relationship type
            for rel_type, pattern in self._marker_patterns.items():
                match = pattern.search(text_lower)
                if match:
                    # This proposition has a marker suggesting relationship
                    # with the previous proposition
                    if i > 0:
                        relationships.append(RelationshipEdge(
                            source_id=propositions[i - 1].id,
                            target_id=prop.id,
                            relationship=rel_type,
                            confidence=0.9
                        ))

        return relationships

    def _detect_with_llm(
        self,
        propositions: List[PropositionNode],
        existing: List[RelationshipEdge]
    ) -> List[RelationshipEdge]:
        """Use LLM to detect additional relationships.

        Args:
            propositions: List of propositions.
            existing: Already detected relationships.

        Returns:
            List of new relationship edges.
        """
        if not self.llm_provider or len(propositions) < 2:
            return []

        # Find pairs without strong relationships
        existing_pairs = {
            (e.source_id, e.target_id)
            for e in existing
            if e.relationship != RelationshipType.FOLLOWS
        }

        # Only query LLM for unconnected pairs
        pairs_to_check = []
        for i in range(len(propositions)):
            for j in range(i + 1, min(i + 3, len(propositions))):  # Check within 2 sentences
                if (propositions[i].id, propositions[j].id) not in existing_pairs:
                    pairs_to_check.append((i, j))

        if not pairs_to_check:
            return []

        # Build prompt for LLM
        new_relationships = []
        for i, j in pairs_to_check[:5]:  # Limit to avoid too many LLM calls
            rel = self._query_llm_for_relationship(
                propositions[i],
                propositions[j]
            )
            if rel:
                new_relationships.append(rel)

        return new_relationships

    def _query_llm_for_relationship(
        self,
        prop1: PropositionNode,
        prop2: PropositionNode
    ) -> Optional[RelationshipEdge]:
        """Query LLM to determine relationship between two propositions.

        Args:
            prop1: First proposition.
            prop2: Second proposition.

        Returns:
            RelationshipEdge if relationship found, None otherwise.
        """
        system_prompt = """You are analyzing relationships between two statements.
Determine the logical relationship between Statement A and Statement B.

Respond with ONLY one of these relationship types:
- CAUSES: A causes or leads to B
- CONTRASTS: A contrasts with or opposes B
- ELABORATES: B provides more detail about A
- REFERENCES: B refers back to something in A
- FOLLOWS: B simply follows A in sequence
- NONE: No clear relationship

Response format: RELATIONSHIP_TYPE"""

        user_prompt = f"""Statement A: {prop1.text}
Statement B: {prop2.text}

What is the relationship from A to B?"""

        try:
            response = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1,
                max_tokens=20
            )

            # Parse response
            response_upper = response.strip().upper()
            for rel_type in RelationshipType:
                if rel_type.value in response_upper:
                    if rel_type != RelationshipType.FOLLOWS:  # Skip FOLLOWS, already handled
                        return RelationshipEdge(
                            source_id=prop1.id,
                            target_id=prop2.id,
                            relationship=rel_type,
                            confidence=0.7  # Lower confidence for LLM-detected
                        )

        except Exception as e:
            logger.warning(f"LLM relationship detection failed: {e}")

        return None

    def detect_between_sentences(
        self,
        sentence1: str,
        sentence2: str
    ) -> Optional[RelationshipType]:
        """Detect relationship between two sentences (simple interface).

        Args:
            sentence1: First sentence.
            sentence2: Second sentence.

        Returns:
            RelationshipType if found, None otherwise.
        """
        # Check sentence2 for markers
        text_lower = sentence2.lower()

        for rel_type, pattern in self._marker_patterns.items():
            if pattern.search(text_lower):
                return rel_type

        return None
