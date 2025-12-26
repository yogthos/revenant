"""Semantic ingestion and graph construction module."""

from .proposition_extractor import PropositionExtractor, SVOTriple
from .relationship_detector import RelationshipDetector, RELATIONSHIP_MARKERS
from .graph_builder import SemanticGraphBuilder, DocumentGraphBuilder
from .context_analyzer import GlobalContextAnalyzer, GlobalContext, ParagraphContext

__all__ = [
    # Proposition extraction
    "PropositionExtractor",
    "SVOTriple",
    # Relationship detection
    "RelationshipDetector",
    "RELATIONSHIP_MARKERS",
    # Graph building
    "SemanticGraphBuilder",
    "DocumentGraphBuilder",
    # Context analysis
    "GlobalContextAnalyzer",
    "GlobalContext",
    "ParagraphContext",
]
