"""Vocabulary control for style transfer.

The LoRA pipeline uses post-processing modules:
- RepetitionReducer: Tracks word usage, replaces overused words with synonyms
- GrammarCorrector: Style-safe grammar/spelling fixes using LanguageTool
- SentenceSplitter: Splits run-on sentences at natural conjunction points
- AsymmetryAnalyzer: Evaluates organic complexity vs mechanical patterns
"""

from .repetition_reducer import RepetitionReducer, ReductionStats
from .grammar_corrector import (
    GrammarCorrector,
    GrammarCorrectorConfig,
    GrammarStats,
    get_grammar_corrector,
    correct_grammar,
)
from .sentence_splitter import (
    SentenceSplitter,
    SentenceSplitterConfig,
    SplitStats,
    get_sentence_splitter,
    split_sentences,
)
from .asymmetry_analyzer import (
    AsymmetryAnalyzer,
    AsymmetryStats,
    get_asymmetry_analyzer,
    analyze_organic_complexity,
)
from .sentence_restructurer import (
    SentenceRestructurer,
    RestructureStats,
    get_sentence_restructurer,
    restructure_sentences,
)

__all__ = [
    "RepetitionReducer",
    "ReductionStats",
    "GrammarCorrector",
    "GrammarCorrectorConfig",
    "GrammarStats",
    "get_grammar_corrector",
    "correct_grammar",
    "SentenceSplitter",
    "SentenceSplitterConfig",
    "SplitStats",
    "get_sentence_splitter",
    "split_sentences",
    "AsymmetryAnalyzer",
    "AsymmetryStats",
    "get_asymmetry_analyzer",
    "analyze_organic_complexity",
    "SentenceRestructurer",
    "RestructureStats",
    "get_sentence_restructurer",
    "restructure_sentences",
]
