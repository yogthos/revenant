"""NLP utilities using spaCy and NLTK."""

import re
from typing import List, Optional, Tuple

from .logging import get_logger

logger = get_logger(__name__)

# Lazy-loaded spaCy model
_nlp = None


def get_nlp():
    """Get the spaCy NLP model, loading it if necessary.

    Returns:
        spaCy Language model.

    Raises:
        RuntimeError: If spaCy model cannot be loaded.
    """
    global _nlp
    if _nlp is None:
        try:
            import spacy
            # Prefer large model with word vectors, fall back to smaller models
            models = ["en_core_web_lg", "en_core_web_md", "en_core_web_sm"]
            for model_name in models:
                try:
                    _nlp = spacy.load(model_name)
                    logger.info(f"Loaded spaCy model: {model_name}")
                    break
                except OSError:
                    continue
            else:
                # No model found, download and use large model
                logger.info("Downloading spaCy model en_core_web_lg...")
                from spacy.cli import download
                download("en_core_web_lg")
                _nlp = spacy.load("en_core_web_lg")
                logger.info("Downloaded and loaded spaCy model: en_core_web_lg")
        except ImportError:
            raise RuntimeError("spaCy is required. Install with: pip install spacy")
    return _nlp


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using spaCy.

    Preserves citations like [^1] and handles edge cases.

    Args:
        text: Input text.

    Returns:
        List of sentence strings.
    """
    if not text or not text.strip():
        return []

    nlp = get_nlp()
    doc = nlp(text)

    sentences = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if sent_text:
            sentences.append(sent_text)

    return sentences


def split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs.

    Args:
        text: Input text.

    Returns:
        List of paragraph strings (non-empty).
    """
    if not text or not text.strip():
        return []

    # Try double newlines first
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    # If no double newlines, try single newlines
    if len(paragraphs) == 1 and '\n' in text:
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

    return paragraphs


def extract_citations(text: str) -> List[Tuple[str, int]]:
    """Extract citations from text.

    Args:
        text: Input text.

    Returns:
        List of (citation, position) tuples.
    """
    pattern = r'\[\^\d+\]'
    citations = []
    for match in re.finditer(pattern, text):
        citations.append((match.group(), match.start()))
    return citations


def remove_citations(text: str) -> str:
    """Remove citations from text.

    Args:
        text: Input text.

    Returns:
        Text with citations removed.
    """
    return re.sub(r'\[\^\d+\]', '', text)


def extract_entities(text: str) -> List[Tuple[str, str]]:
    """Extract named entities from text.

    Args:
        text: Input text.

    Returns:
        List of (entity_text, entity_label) tuples.
    """
    nlp = get_nlp()
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """Extract keywords (nouns and verbs) from text.

    Args:
        text: Input text.
        top_n: Maximum number of keywords to return.

    Returns:
        List of lemmatized keywords.
    """
    nlp = get_nlp()
    doc = nlp(text)

    # Extract nouns and verbs, lemmatize them
    keywords = []
    for token in doc:
        if token.pos_ in ("NOUN", "VERB", "PROPN") and not token.is_stop:
            lemma = token.lemma_.lower()
            if lemma not in keywords and len(lemma) > 2:
                keywords.append(lemma)

    return keywords[:top_n]


def count_words(text: str) -> int:
    """Count words in text.

    Args:
        text: Input text.

    Returns:
        Word count.
    """
    if not text:
        return 0
    return len(text.split())


def calculate_burstiness(sentences: List[str]) -> float:
    """Calculate burstiness (coefficient of variation of sentence lengths).

    Args:
        sentences: List of sentences.

    Returns:
        Burstiness value (0 = uniform, higher = more variable).
    """
    if len(sentences) < 2:
        return 0.0

    lengths = [count_words(s) for s in sentences]
    mean_length = sum(lengths) / len(lengths)

    if mean_length == 0:
        return 0.0

    variance = sum((l - mean_length) ** 2 for l in lengths) / len(lengths)
    std_dev = variance ** 0.5

    return std_dev / mean_length


def get_pos_distribution(text: str) -> dict:
    """Get POS tag distribution for text.

    Args:
        text: Input text.

    Returns:
        Dictionary of POS tag to count.
    """
    nlp = get_nlp()
    doc = nlp(text)

    distribution = {}
    for token in doc:
        pos = token.pos_
        distribution[pos] = distribution.get(pos, 0) + 1

    return distribution


def get_dependency_depth(text: str) -> float:
    """Get average dependency tree depth for sentences.

    Higher values indicate more complex sentence structures.

    Args:
        text: Input text.

    Returns:
        Average dependency depth.
    """
    nlp = get_nlp()
    doc = nlp(text)

    depths = []
    for sent in doc.sents:
        # Find max depth in this sentence
        max_depth = 0
        for token in sent:
            depth = 0
            current = token
            while current.head != current:
                depth += 1
                current = current.head
            max_depth = max(max_depth, depth)
        depths.append(max_depth)

    if not depths:
        return 0.0

    return sum(depths) / len(depths)


def detect_perspective(text: str) -> str:
    """Detect the perspective (first/third person) of text.

    Args:
        text: Input text.

    Returns:
        One of: "first_person_singular", "first_person_plural", "third_person"
    """
    text_lower = text.lower()

    # Count first-person pronouns
    first_singular = len(re.findall(r'\b(i|me|my|mine|myself)\b', text_lower))
    first_plural = len(re.findall(r'\b(we|us|our|ours|ourselves)\b', text_lower))
    third = len(re.findall(r'\b(he|she|it|they|him|her|them|his|hers|its|their)\b', text_lower))

    if first_singular > first_plural and first_singular > third:
        return "first_person_singular"
    elif first_plural > first_singular and first_plural > third:
        return "first_person_plural"
    else:
        return "third_person"


def setup_nltk() -> None:
    """Download required NLTK data if not present."""
    try:
        import nltk

        required_packages = [
            'punkt',
            'averaged_perceptron_tagger',
            'wordnet',
            'vader_lexicon'
        ]

        for package in required_packages:
            try:
                nltk.data.find(f'tokenizers/{package}' if package == 'punkt'
                              else f'taggers/{package}' if 'tagger' in package
                              else f'corpora/{package}' if package == 'wordnet'
                              else f'sentiment/{package}')
            except LookupError:
                logger.info(f"Downloading NLTK package: {package}")
                nltk.download(package, quiet=True)

    except ImportError:
        logger.warning("NLTK not installed, some features may be limited")


class NLPManager:
    """Manager class for NLP operations.

    Provides a unified interface for NLP processing using spaCy.
    """

    def __init__(self):
        """Initialize NLP manager."""
        self._nlp = None

    @property
    def nlp(self):
        """Get the spaCy model, lazy-loading if needed."""
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def process(self, text: str):
        """Process text with spaCy.

        Args:
            text: Text to process.

        Returns:
            spaCy Doc object.
        """
        return self.nlp(text)

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences.

        Args:
            text: Input text.

        Returns:
            List of sentences.
        """
        return split_into_sentences(text)

    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """Extract named entities from text.

        Args:
            text: Input text.

        Returns:
            List of (entity_text, entity_label) tuples.
        """
        return extract_entities(text)

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract keywords from text.

        Args:
            text: Input text.
            top_n: Maximum keywords to return.

        Returns:
            List of keywords.
        """
        return extract_keywords(text, top_n)

    def get_pos_distribution(self, text: str) -> dict:
        """Get POS tag distribution.

        Args:
            text: Input text.

        Returns:
            Dictionary of POS tag to count.
        """
        return get_pos_distribution(text)

    def get_dependency_depth(self, text: str) -> float:
        """Get average dependency depth.

        Args:
            text: Input text.

        Returns:
            Average depth.
        """
        return get_dependency_depth(text)

    def calculate_burstiness(self, sentences: List[str]) -> float:
        """Calculate burstiness of sentence lengths.

        Args:
            sentences: List of sentences.

        Returns:
            Burstiness value.
        """
        return calculate_burstiness(sentences)

    def detect_perspective(self, text: str) -> str:
        """Detect perspective of text.

        Args:
            text: Input text.

        Returns:
            Perspective string.
        """
        return detect_perspective(text)
