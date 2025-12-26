"""Sentence plan data models.

These models represent the planned structure for generating styled text.
Full implementation in Phase 3.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

from .graph import PropositionNode, SemanticGraph


class SentenceRole(Enum):
    """Role of a sentence in the paragraph."""
    THESIS = "THESIS"
    ELABORATION = "ELABORATION"
    CONTRAST = "CONTRAST"
    EXAMPLE = "EXAMPLE"
    CONCLUSION = "CONCLUSION"


class TransitionType(Enum):
    """Type of transition from previous sentence."""
    NONE = "NONE"
    CAUSAL = "CAUSAL"  # therefore, thus, consequently
    ADVERSATIVE = "ADVERSATIVE"  # however, but, although
    ADDITIVE = "ADDITIVE"  # moreover, furthermore, also
    TEMPORAL = "TEMPORAL"  # then, next, finally


@dataclass
class SentenceNode:
    """A planned sentence in the output."""
    id: str
    propositions: List[PropositionNode] = field(default_factory=list)
    role: SentenceRole = SentenceRole.ELABORATION
    transition: TransitionType = TransitionType.NONE
    target_length: int = 15  # Target word count
    target_skeleton: Optional[str] = None  # From matched style graph
    style_template: Optional[str] = None  # RAG example sentence
    keywords: List[str] = field(default_factory=list)  # Must include these

    def get_proposition_text(self) -> str:
        """Get combined text of all propositions."""
        return " ".join(p.text for p in self.propositions)


@dataclass
class SentencePlan:
    """Complete plan for generating a paragraph."""
    nodes: List[SentenceNode] = field(default_factory=list)
    paragraph_intent: str = ""  # DEFINITION, ARGUMENT, NARRATIVE
    paragraph_signature: str = ""  # CONTRAST, CAUSALITY, SEQUENCE
    paragraph_role: str = "BODY"  # INTRO, BODY, CONCLUSION
    source_graph: Optional[SemanticGraph] = None
    matched_style_graph_id: Optional[str] = None

    def to_summary(self) -> str:
        """Create a compact summary for LLM context."""
        parts = []
        for i, node in enumerate(self.nodes):
            role = node.role.value
            length = node.target_length
            trans = node.transition.value if node.transition != TransitionType.NONE else ""
            trans_str = f" ({trans})" if trans else ""
            parts.append(f"S{i+1}: {role}{trans_str}, ~{length}w")
        return f"Plan: {'; '.join(parts)}"

    def __len__(self) -> int:
        return len(self.nodes)


@dataclass
class ParagraphTransition:
    """Transition context between paragraphs."""
    previous_closing_sentence: str = ""
    previous_paragraph_summary: str = ""
    transition_type: str = "CONTINUATION"  # CONTINUATION, CONTRAST, NEW_TOPIC
