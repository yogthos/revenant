"""Style RAG module for dynamic few-shot prompting.

This module provides retrieval-augmented generation (RAG) for style transfer,
enabling dynamic injection of author style examples into prompts.

Architecture:
    Corpus Files → Style Analyzer → ChromaDB → Retriever → Session Context
                   (spaCy metrics)  (vectors)  (2-channel)  (prompt injection)

Usage:
    # 1. Index a corpus (one-time)
    from src.rag import CorpusIndexer
    indexer = CorpusIndexer("data/rag_index")
    indexer.index_corpus("data/corpus/author.txt", "Author Name")

    # 2. Create RAG context for a session
    from src.rag import create_rag_context
    context = create_rag_context("Author Name", sample_text="...")

    # 3. Get formatted examples for prompt
    style_examples = context.format_for_prompt()
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

from .style_retriever import (
    StyleRetriever,
    RetrievedChunk,
    get_retriever,
)

from .session_context import (
    StyleRAGContext,
    RAGContextManager,
    get_context_manager,
    create_rag_context,
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
    # Retrieval
    "StyleRetriever",
    "RetrievedChunk",
    "get_retriever",
    # Session context
    "StyleRAGContext",
    "RAGContextManager",
    "get_context_manager",
    "create_rag_context",
]
