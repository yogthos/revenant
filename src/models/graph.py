"""Semantic graph data models.

These models represent the semantic structure extracted from input text.
Full implementation in Phase 2.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class RelationshipType(Enum):
    """Types of relationships between propositions."""
    CAUSES = "CAUSES"
    CONTRASTS = "CONTRASTS"
    ELABORATES = "ELABORATES"
    REFERENCES = "REFERENCES"
    FOLLOWS = "FOLLOWS"


class ParagraphRole(Enum):
    """Role of a paragraph in the document."""
    INTRO = "INTRO"
    BODY = "BODY"
    CONCLUSION = "CONCLUSION"


class RhetoricalIntent(Enum):
    """Rhetorical intent of text."""
    DEFINITION = "DEFINITION"
    ARGUMENT = "ARGUMENT"
    NARRATIVE = "NARRATIVE"
    INTERROGATIVE = "INTERROGATIVE"
    IMPERATIVE = "IMPERATIVE"


@dataclass
class PropositionNode:
    """An atomic proposition extracted from text."""
    id: str
    text: str  # The atomic proposition
    subject: str
    verb: str
    object: Optional[str] = None
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    source_sentence_idx: int = 0
    is_citation: bool = False
    is_quotation: bool = False
    attached_citations: List[str] = field(default_factory=list)  # ["[^1]", "[^2]"]


@dataclass
class RelationshipEdge:
    """A relationship between two propositions."""
    source_id: str
    target_id: str
    relationship: RelationshipType
    confidence: float = 1.0


@dataclass
class SemanticGraph:
    """Semantic graph for a paragraph."""
    nodes: List[PropositionNode] = field(default_factory=list)
    edges: List[RelationshipEdge] = field(default_factory=list)
    paragraph_idx: int = 0
    role: ParagraphRole = ParagraphRole.BODY
    intent: RhetoricalIntent = RhetoricalIntent.DEFINITION

    def to_summary(self) -> str:
        """Create a compact summary for LLM context."""
        node_texts = [f"P{i+1}: {n.text}" for i, n in enumerate(self.nodes)]
        edge_texts = [
            f"P{self._node_idx(e.source_id)+1} {e.relationship.value} P{self._node_idx(e.target_id)+1}"
            for e in self.edges
        ]
        return f"Propositions: {'; '.join(node_texts)}\nRelationships: {', '.join(edge_texts) or 'none'}"

    def _node_idx(self, node_id: str) -> int:
        """Get index of node by ID."""
        for i, n in enumerate(self.nodes):
            if n.id == node_id:
                return i
        return -1


@dataclass
class DocumentGraph:
    """Document-level graph containing paragraph graphs."""
    paragraphs: List[SemanticGraph] = field(default_factory=list)
    thesis: str = ""
    intent: str = ""
    keywords: List[str] = field(default_factory=list)
    perspective: str = "third_person"
