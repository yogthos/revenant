"""Template-based style transfer components."""

from .models import (
    SentenceTemplate,
    TemplateSlot,
    SlotType,
    SentenceType,
    RhetoricalRole,
    LogicalRelation,
    WordType,
    VocabularyProfile,
    CorpusStatistics,
    ValidationIssue,
    RepairAction,
    SentenceValidationResult,
    ParagraphRepairAction,
    ParagraphValidationResult,
    SentenceRequirements,
)

from .statistics import (
    CorpusStatisticsExtractor,
    SentenceClassifier,
    FUNCTION_WORDS,
    CONNECTOR_PATTERNS,
)

from .extractor import (
    SkeletonExtractor,
    TemplateLibrary,
    ExtractedSlot,
)

from .vocabulary import (
    WordClassifier,
    TechnicalTermExtractor,
    GeneralWordMapper,
    WordClassification,
    VocabularyMapping,
    build_vocabulary_profile_from_corpus,
)

from .storage import (
    TemplateStore,
    TemplateStoreConfig,
)

from .filler import (
    SlotFiller,
    TemplateMatcher,
    Proposition,
    FilledSentence,
    FilledSlot,
)

from .validator import (
    SentenceValidator,
    SentenceRepairer,
    ParagraphValidator,
    ParagraphRepairer,
)

from .pipeline import (
    TemplateBasedTransfer,
    TransferConfig,
    TransferResult,
    create_transfer_pipeline,
)

__all__ = [
    # Models
    "SentenceTemplate",
    "TemplateSlot",
    "SlotType",
    "SentenceType",
    "RhetoricalRole",
    "LogicalRelation",
    "WordType",
    "VocabularyProfile",
    "CorpusStatistics",
    "ValidationIssue",
    "RepairAction",
    "SentenceValidationResult",
    "ParagraphRepairAction",
    "ParagraphValidationResult",
    "SentenceRequirements",
    # Statistics
    "CorpusStatisticsExtractor",
    "SentenceClassifier",
    "FUNCTION_WORDS",
    "CONNECTOR_PATTERNS",
    # Extractor
    "SkeletonExtractor",
    "TemplateLibrary",
    "ExtractedSlot",
    # Vocabulary
    "WordClassifier",
    "TechnicalTermExtractor",
    "GeneralWordMapper",
    "WordClassification",
    "VocabularyMapping",
    "build_vocabulary_profile_from_corpus",
    # Storage
    "TemplateStore",
    "TemplateStoreConfig",
    # Filler
    "SlotFiller",
    "TemplateMatcher",
    "Proposition",
    "FilledSentence",
    "FilledSlot",
    # Validator
    "SentenceValidator",
    "SentenceRepairer",
    "ParagraphValidator",
    "ParagraphRepairer",
    # Pipeline
    "TemplateBasedTransfer",
    "TransferConfig",
    "TransferResult",
    "create_transfer_pipeline",
]
