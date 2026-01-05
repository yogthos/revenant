"""Structural RAG module for rhythm and syntax guidance.

This module provides structural guidance for style transfer based on
author corpus analysis, without few-shot content injection.

Architecture:
    Corpus Files → Style Analyzer → ChromaDB → Structural RAG
                   (spaCy metrics)  (vectors)  (rhythm/syntax patterns)

Usage:
    # 1. Index a corpus (one-time)
    from src.rag import CorpusIndexer
    indexer = CorpusIndexer("data/rag_index")
    indexer.index_corpus("data/corpus/author.txt", "Author Name")

    # 2. Get structural guidance
    from src.rag import get_structural_rag
    rag = get_structural_rag("Author Name")
    guidance = rag.get_guidance(input_text)
"""

from .style_analyzer import (
    StyleMetrics,
    StyleAnalyzer,
    get_style_analyzer,
    analyze_style,
)

from .corpus_indexer import (
    CorpusIndexer,
    IndexedChunk,
    get_indexer,
)

from .structural_analyzer import (
    StructuralAnalyzer,
    RhythmFingerprint,
    StructuralStyle,
    SentencePattern,
    get_structural_analyzer,
    extract_rhythm_instruction,
)

from .structural_rag import (
    StructuralRAG,
    StructuralGuidance,
    get_structural_rag,
    get_structural_guidance,
)

from .enhanced_analyzer import (
    EnhancedStructuralAnalyzer,
    EnhancedStyleProfile,
    SyntacticTemplate,
    VocabularyCluster,
    TransitionInventory,
    StanceProfile,
    OpeningPatterns,
    get_enhanced_analyzer,
)

__all__ = [
    # Style analysis
    "StyleMetrics",
    "StyleAnalyzer",
    "get_style_analyzer",
    "analyze_style",
    # Corpus indexing
    "CorpusIndexer",
    "IndexedChunk",
    "get_indexer",
    # Structural analysis
    "StructuralAnalyzer",
    "RhythmFingerprint",
    "StructuralStyle",
    "SentencePattern",
    "get_structural_analyzer",
    "extract_rhythm_instruction",
    # Structural RAG
    "StructuralRAG",
    "StructuralGuidance",
    "get_structural_rag",
    "get_structural_guidance",
    # Enhanced analyzer
    "EnhancedStructuralAnalyzer",
    "EnhancedStyleProfile",
    "SyntacticTemplate",
    "VocabularyCluster",
    "TransitionInventory",
    "StanceProfile",
    "OpeningPatterns",
    "get_enhanced_analyzer",
]
