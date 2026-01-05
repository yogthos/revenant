"""Validation modules for semantic preservation.

Core validation for the LoRA pipeline:
- SemanticGraphBuilder: Builds semantic graphs from text
- SemanticGraphComparator: Compares semantic graphs for meaning preservation
- NLIAuditor: Sentence-level NLI verification for fact integrity
"""

from .semantic_graph import (
    SemanticGraphBuilder,
    SemanticGraphComparator,
)
from .nli_auditor import (
    NLIAuditor,
    AuditResult,
    SentenceIssue,
    get_nli_auditor,
)

__all__ = [
    "SemanticGraphBuilder",
    "SemanticGraphComparator",
    "NLIAuditor",
    "AuditResult",
    "SentenceIssue",
    "get_nli_auditor",
]
