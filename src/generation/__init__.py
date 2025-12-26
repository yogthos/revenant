"""Generation module for styled text output."""

from .prompt_builder import (
    PromptBuilder,
    MultiSentencePromptBuilder,
    GenerationPrompt,
    TRANSITION_WORDS,
)
from .sentence_generator import (
    SentenceGenerator,
    MultiPassGenerator,
    SessionBasedGenerator,
    GeneratedSentence,
    GeneratedParagraph,
)
from .critics import (
    Critic,
    CriticPanel,
    CriticFeedback,
    CriticType,
    ValidationResult,
    LengthCritic,
    KeywordCritic,
    FluencyCritic,
    SemanticCritic,
    StyleCritic,
)

__all__ = [
    # Prompt building
    "PromptBuilder",
    "MultiSentencePromptBuilder",
    "GenerationPrompt",
    "TRANSITION_WORDS",
    # Sentence generation
    "SentenceGenerator",
    "MultiPassGenerator",
    "SessionBasedGenerator",
    "GeneratedSentence",
    "GeneratedParagraph",
    # Validation critics
    "Critic",
    "CriticPanel",
    "CriticFeedback",
    "CriticType",
    "ValidationResult",
    "LengthCritic",
    "KeywordCritic",
    "FluencyCritic",
    "SemanticCritic",
    "StyleCritic",
]
