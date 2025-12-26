"""Vocabulary management for template-based style transfer.

Handles:
- Classification of words as technical vs general
- Extraction of technical terms to preserve
- Mapping of general words to author's vocabulary
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
import re

from ..utils.nlp import get_nlp, compute_semantic_similarity
from ..utils.logging import get_logger
from .models import WordType, VocabularyProfile
from .statistics import FUNCTION_WORDS, CONNECTOR_PATTERNS

logger = get_logger(__name__)


@dataclass
class WordClassification:
    """Classification result for a word."""
    word: str
    lemma: str
    word_type: WordType
    pos_tag: str
    confidence: float = 1.0

    # For technical terms
    is_domain_specific: bool = False
    domain_hint: Optional[str] = None


@dataclass
class VocabularyMapping:
    """Mapping from source word to target author's word."""
    source_word: str
    source_lemma: str
    target_word: str
    target_lemma: str
    similarity: float
    pos_match: bool = True


class WordClassifier:
    """Classifies words by type for vocabulary management."""

    # Words that commonly appear in technical/academic contexts
    # but are not themselves domain-specific
    ACADEMIC_GENERAL = {
        "analysis", "approach", "aspect", "basis", "concept", "context",
        "data", "definition", "development", "effect", "element", "evidence",
        "example", "factor", "feature", "finding", "form", "framework",
        "function", "idea", "impact", "implication", "importance", "issue",
        "level", "method", "model", "nature", "pattern", "perspective",
        "point", "principle", "problem", "process", "question", "reason",
        "relationship", "research", "result", "role", "significance",
        "source", "structure", "study", "system", "term", "theory", "type",
        "understanding", "value", "variable", "view", "work",
    }

    def __init__(self, domain_terms: Set[str] = None):
        """Initialize classifier.

        Args:
            domain_terms: Optional set of known domain-specific terms.
        """
        self._nlp = None
        self.domain_terms = domain_terms or set()
        self._all_connectors = self._build_connector_set()

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def _build_connector_set(self) -> Set[str]:
        """Build set of all connector words."""
        connectors = set()
        for conn_list in CONNECTOR_PATTERNS.values():
            connectors.update(conn_list)
        return connectors

    def classify(self, word: str, context: str = "") -> WordClassification:
        """Classify a single word.

        Args:
            word: The word to classify.
            context: Optional surrounding context for disambiguation.

        Returns:
            WordClassification with type and metadata.
        """
        # Parse word (and context if provided)
        if context:
            doc = self.nlp(context)
            # Find the word in context
            token = None
            for t in doc:
                if t.text.lower() == word.lower():
                    token = t
                    break
            if not token:
                # Word not found in context, parse alone
                doc = self.nlp(word)
                token = doc[0] if doc else None
        else:
            doc = self.nlp(word)
            token = doc[0] if doc else None

        if not token:
            return WordClassification(
                word=word,
                lemma=word.lower(),
                word_type=WordType.GENERAL,
                pos_tag="X",
                confidence=0.5,
            )

        lemma = token.lemma_.lower()
        pos_tag = token.pos_
        word_lower = word.lower()

        # Check for proper nouns first
        if pos_tag == "PROPN" or token.ent_type_:
            return WordClassification(
                word=word,
                lemma=lemma,
                word_type=WordType.PROPER,
                pos_tag=pos_tag,
                confidence=0.95,
            )

        # Check connectors BEFORE function words (connectors are more specific)
        if word_lower in self._all_connectors:
            return WordClassification(
                word=word,
                lemma=lemma,
                word_type=WordType.CONNECTOR,
                pos_tag=pos_tag,
                confidence=0.95,
            )

        # Check function words
        if word_lower in FUNCTION_WORDS or token.is_stop:
            return WordClassification(
                word=word,
                lemma=lemma,
                word_type=WordType.FUNCTION,
                pos_tag=pos_tag,
                confidence=0.95,
            )

        # Check known domain terms
        if word_lower in self.domain_terms or lemma in self.domain_terms:
            return WordClassification(
                word=word,
                lemma=lemma,
                word_type=WordType.TECHNICAL,
                pos_tag=pos_tag,
                confidence=0.9,
                is_domain_specific=True,
            )

        # Heuristic: technical terms often have specific characteristics
        is_technical, confidence = self._is_technical_heuristic(
            word, lemma, pos_tag, token
        )

        if is_technical:
            return WordClassification(
                word=word,
                lemma=lemma,
                word_type=WordType.TECHNICAL,
                pos_tag=pos_tag,
                confidence=confidence,
                is_domain_specific=True,
            )

        # Default to general word
        return WordClassification(
            word=word,
            lemma=lemma,
            word_type=WordType.GENERAL,
            pos_tag=pos_tag,
            confidence=0.8,
        )

    def _is_technical_heuristic(
        self,
        word: str,
        lemma: str,
        pos_tag: str,
        token
    ) -> Tuple[bool, float]:
        """Apply heuristics to detect technical terms.

        Returns:
            Tuple of (is_technical, confidence).
        """
        confidence = 0.5
        is_technical = False

        # Technical indicators
        word_lower = word.lower()

        # Contains digits (e.g., "CO2", "H2O")
        if any(c.isdigit() for c in word):
            return True, 0.85

        # Contains hyphens or underscores (compound technical terms)
        if "-" in word or "_" in word:
            # But not if it's a common hyphenated word
            if len(word) > 5:
                return True, 0.7

        # Unusual capitalization patterns (e.g., "DNA", "RNA", "API")
        if word.isupper() and len(word) >= 2:
            return True, 0.85

        # Mixed case in middle (e.g., "iPhone", "JavaScript")
        if any(c.isupper() for c in word[1:]):
            return True, 0.8

        # Latin/Greek prefixes common in scientific terms
        scientific_prefixes = [
            "bio", "geo", "hydro", "thermo", "electro", "neuro", "psycho",
            "astro", "cosmo", "photo", "chrono", "micro", "macro", "nano",
            "meta", "para", "poly", "mono", "multi", "quasi", "pseudo",
            "hyper", "hypo", "anti", "proto", "retro",
        ]
        for prefix in scientific_prefixes:
            if word_lower.startswith(prefix) and len(word) > len(prefix) + 3:
                return True, 0.75

        # Scientific suffixes
        scientific_suffixes = [
            "ology", "ism", "tion", "osis", "itis", "emia", "ectomy",
            "scope", "graph", "meter", "phyte", "cyte", "plasm",
        ]
        for suffix in scientific_suffixes:
            if word_lower.endswith(suffix) and len(word) > len(suffix) + 3:
                # But check if it's a common academic word
                if lemma not in self.ACADEMIC_GENERAL:
                    return True, 0.7

        # Check word frequency - rare words are more likely technical
        # spaCy's is_oov (out of vocabulary) is a proxy for rarity
        if hasattr(token, "is_oov") and token.is_oov:
            confidence += 0.15

        # Nouns are more likely to be technical terms
        if pos_tag == "NOUN" and lemma not in self.ACADEMIC_GENERAL:
            confidence += 0.1

        return is_technical, confidence

    def classify_sentence(
        self,
        sentence: str
    ) -> List[WordClassification]:
        """Classify all words in a sentence.

        Args:
            sentence: The sentence to analyze.

        Returns:
            List of WordClassification for each token.
        """
        doc = self.nlp(sentence)
        classifications = []

        for token in doc:
            if token.is_punct or token.is_space:
                continue

            classification = self.classify(token.text, sentence)
            classifications.append(classification)

        return classifications


class TechnicalTermExtractor:
    """Extracts technical terms that should be preserved during style transfer."""

    def __init__(self, domain_hints: List[str] = None):
        """Initialize extractor.

        Args:
            domain_hints: Optional list of domain keywords to help identify
                         related technical terms.
        """
        self._nlp = None
        self.domain_hints = domain_hints or []
        self.classifier = WordClassifier()

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def extract(self, text: str) -> Set[str]:
        """Extract technical terms from text.

        Args:
            text: The text to analyze.

        Returns:
            Set of technical terms to preserve.
        """
        doc = self.nlp(text)
        technical_terms = set()

        # Extract named entities (often technical)
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW"):
                technical_terms.add(ent.text)

        # Extract noun phrases that look technical
        for chunk in doc.noun_chunks:
            # Clean up whitespace and newlines
            chunk_text = " ".join(chunk.text.split())
            if self._is_technical_phrase(chunk, doc):
                technical_terms.add(chunk_text)

        # Extract individual technical words
        for token in doc:
            if token.is_punct or token.is_space:
                continue

            classification = self.classifier.classify(token.text, text)
            if classification.word_type == WordType.TECHNICAL:
                technical_terms.add(token.text)

        # Also preserve proper nouns
        for token in doc:
            if token.pos_ == "PROPN":
                technical_terms.add(token.text)

        return technical_terms

    def extract_with_context(
        self,
        text: str
    ) -> List[Tuple[str, str]]:
        """Extract technical terms with their surrounding context.

        Args:
            text: The text to analyze.

        Returns:
            List of (term, context_snippet) tuples.
        """
        doc = self.nlp(text)
        results = []

        terms = self.extract(text)

        for term in terms:
            # Find term in document and get context
            for sent in doc.sents:
                if term.lower() in sent.text.lower():
                    results.append((term, sent.text))
                    break

        return results

    def _is_technical_phrase(self, chunk, doc) -> bool:
        """Determine if a noun phrase is technical."""
        text = chunk.text.lower()

        # Single common word is not technical
        if len(chunk) == 1:
            token = chunk.root
            if token.is_stop or token.text.lower() in FUNCTION_WORDS:
                return False
            # Check if it's classified as technical
            classification = self.classifier.classify(token.text)
            return classification.word_type == WordType.TECHNICAL

        # Multi-word phrases with technical indicators
        words = text.split()

        # Check for compound technical terms
        if any(self._has_technical_indicators(w) for w in words):
            return True

        # Check if head noun is technical
        root_classification = self.classifier.classify(chunk.root.text)
        if root_classification.word_type == WordType.TECHNICAL:
            return True

        return False

    def _has_technical_indicators(self, word: str) -> bool:
        """Check if a word has technical term indicators."""
        word_lower = word.lower()

        # Digits
        if any(c.isdigit() for c in word):
            return True

        # Unusual capitalization
        if word.isupper() and len(word) >= 2:
            return True

        # Scientific prefixes/suffixes
        technical_patterns = [
            r"^bio", r"^geo", r"^thermo", r"^electro", r"^neuro",
            r"ology$", r"ism$", r"osis$", r"itis$",
        ]
        for pattern in technical_patterns:
            if re.search(pattern, word_lower):
                return True

        return False


class GeneralWordMapper:
    """Maps general words from input to author's vocabulary."""

    def __init__(self, vocabulary_profile: VocabularyProfile):
        """Initialize mapper with author's vocabulary profile.

        Args:
            vocabulary_profile: The target author's vocabulary statistics.
        """
        self._nlp = None
        self.profile = vocabulary_profile
        self.classifier = WordClassifier()

        # Cache for computed mappings
        self._mapping_cache: Dict[str, VocabularyMapping] = {}

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def map_word(
        self,
        word: str,
        pos_tag: str = None,
        exclude: Set[str] = None
    ) -> Optional[VocabularyMapping]:
        """Map a general word to the author's vocabulary.

        Args:
            word: The source word to map.
            pos_tag: Optional POS tag to constrain mapping.
            exclude: Words to exclude from candidates.

        Returns:
            VocabularyMapping if a good match is found, None otherwise.
        """
        exclude = exclude or set()
        word_lower = word.lower()

        # Check cache
        cache_key = f"{word_lower}:{pos_tag or 'any'}"
        if cache_key in self._mapping_cache:
            mapping = self._mapping_cache[cache_key]
            if mapping.target_word not in exclude:
                return mapping

        # Get word info
        doc = self.nlp(word)
        if not doc:
            return None

        token = doc[0]
        source_lemma = token.lemma_.lower()
        actual_pos = pos_tag or token.pos_

        # Get candidates from author's vocabulary
        candidates = self._get_candidates(actual_pos, exclude)

        if not candidates:
            return None

        # Find best match using semantic similarity
        best_match = None
        best_similarity = -1.0

        for candidate, freq in candidates:
            if candidate in exclude:
                continue

            similarity = compute_semantic_similarity(word_lower, candidate)

            # Weight by frequency (prefer more common words)
            weighted_score = similarity * (1 + freq * 0.5)

            if weighted_score > best_similarity:
                best_similarity = weighted_score
                best_match = (candidate, similarity)

        if best_match and best_match[1] > 0.3:  # Minimum similarity threshold
            candidate, similarity = best_match

            mapping = VocabularyMapping(
                source_word=word,
                source_lemma=source_lemma,
                target_word=candidate,
                target_lemma=candidate,  # Already lemmatized in profile
                similarity=similarity,
                pos_match=True,
            )

            self._mapping_cache[cache_key] = mapping
            return mapping

        return None

    def map_sentence(
        self,
        sentence: str,
        technical_terms: Set[str] = None
    ) -> List[Tuple[str, Optional[VocabularyMapping]]]:
        """Map all general words in a sentence.

        Args:
            sentence: The sentence to process.
            technical_terms: Terms to preserve (not map).

        Returns:
            List of (original_word, mapping) tuples.
            Mapping is None if word should not be changed.
        """
        technical_terms = technical_terms or set()
        technical_lower = {t.lower() for t in technical_terms}

        doc = self.nlp(sentence)
        results = []
        used_words = set()  # Track words we've already mapped to

        for token in doc:
            if token.is_punct or token.is_space:
                continue

            word = token.text
            word_lower = word.lower()

            # Skip technical terms
            if word_lower in technical_lower or word in technical_terms:
                results.append((word, None))
                continue

            # Classify word
            classification = self.classifier.classify(word, sentence)

            # Only map general words
            if classification.word_type != WordType.GENERAL:
                results.append((word, None))
                continue

            # Try to find mapping
            mapping = self.map_word(word, token.pos_, exclude=used_words)

            if mapping:
                used_words.add(mapping.target_word)

            results.append((word, mapping))

        return results

    def _get_candidates(
        self,
        pos_tag: str,
        exclude: Set[str] = None
    ) -> List[Tuple[str, float]]:
        """Get candidate words for a given POS tag.

        Args:
            pos_tag: The POS tag to match.
            exclude: Words to exclude.

        Returns:
            List of (word, frequency) tuples.
        """
        exclude = exclude or set()

        # Map POS to vocabulary profile category
        if pos_tag == "VERB":
            return [
                (w, f) for w, f in self.profile.common_verbs.items()
                if w not in exclude
            ]
        elif pos_tag in ("ADJ", "ADV"):
            return [
                (w, f) for w, f in self.profile.modifiers.items()
                if w not in exclude
            ]
        elif pos_tag == "NOUN":
            return [
                (w, f) for w, f in self.profile.general_words.items()
                if w not in exclude
            ]
        else:
            # Try POS-specific vocab
            pos_words = self.profile.words_by_pos.get(pos_tag, {})
            return [
                (w, f) for w, f in pos_words.items()
                if w not in exclude
            ]

    def get_connector(
        self,
        relation_type: str,
        exclude: Set[str] = None
    ) -> Optional[str]:
        """Get a connector word for a logical relation.

        Args:
            relation_type: Type of relation (causal, adversative, etc.)
            exclude: Connectors to exclude.

        Returns:
            A connector word if available.
        """
        return self.profile.get_connector(
            relation_type,
            exclude=list(exclude) if exclude else None
        )


def build_vocabulary_profile_from_corpus(
    paragraphs: List[str]
) -> VocabularyProfile:
    """Build a vocabulary profile from a corpus.

    Convenience function that uses CorpusStatisticsExtractor.

    Args:
        paragraphs: List of paragraph texts.

    Returns:
        VocabularyProfile for the corpus.
    """
    from .statistics import CorpusStatisticsExtractor

    extractor = CorpusStatisticsExtractor()
    return extractor.extract_vocabulary_profile(paragraphs)
