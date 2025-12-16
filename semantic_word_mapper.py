"""
Semantic Word Mapper Module

Maps common, non-technical words to their closest semantic equivalents
in the sample text vocabulary using word embeddings. This ensures the
output text uses vocabulary from the sample text as much as possible.

Key features:
- Uses spaCy word vectors for semantic similarity
- Filters by part of speech for grammatical correctness
- Builds mapping upfront during initialization
- Applies mappings during text processing
"""

import re
import spacy
import numpy as np
from typing import Dict, List, Set, Optional, Tuple
from collections import Counter, defaultdict


class SemanticWordMapper:
    """
    Maps common words to sample text vocabulary using semantic similarity.

    Uses word embeddings to find the closest semantic match in the sample
    text for each common word, ensuring grammatical correctness through
    POS filtering.
    """

    # Common non-technical words to map (organized by POS)
    COMMON_WORDS = {
        'ADJ': {
            'important', 'significant', 'large', 'small', 'major', 'minor',
            'key', 'main', 'primary', 'secondary', 'essential', 'necessary',
            'vital', 'critical', 'crucial', 'fundamental', 'basic', 'central',
            'principal', 'chief', 'leading', 'prominent', 'notable', 'remarkable',
            'substantial', 'considerable', 'extensive', 'comprehensive', 'complete',
            'effective', 'efficient', 'successful', 'powerful', 'strong', 'weak',
            'clear', 'obvious', 'evident', 'apparent', 'visible', 'noticeable',
            'different', 'similar', 'same', 'various', 'diverse', 'multiple',
            'numerous', 'many', 'few', 'several', 'some', 'most', 'all',
            'new', 'old', 'recent', 'ancient', 'modern', 'contemporary',
            'good', 'bad', 'better', 'worse', 'best', 'worst', 'excellent',
            'poor', 'adequate', 'sufficient', 'insufficient', 'enough'
        },
        'VERB': {
            'show', 'indicate', 'demonstrate', 'reveal', 'suggest', 'imply',
            'appear', 'seem', 'look', 'become', 'remain', 'stay', 'keep',
            'make', 'do', 'get', 'take', 'give', 'put', 'set', 'go', 'come',
            'use', 'utilize', 'employ', 'apply', 'implement', 'execute',
            'create', 'produce', 'generate', 'develop', 'build', 'construct',
            'establish', 'found', 'form', 'shape', 'design', 'plan', 'prepare',
            'begin', 'start', 'continue', 'proceed', 'advance', 'progress',
            'end', 'finish', 'complete', 'conclude', 'terminate', 'stop',
            'change', 'modify', 'alter', 'transform', 'convert', 'adapt',
            'increase', 'decrease', 'grow', 'expand', 'reduce', 'shrink',
            'improve', 'enhance', 'strengthen', 'weaken', 'support', 'help',
            'require', 'need', 'demand', 'request', 'ask', 'seek', 'find',
            'obtain', 'acquire', 'gain', 'achieve', 'accomplish', 'reach',
            'understand', 'know', 'learn', 'recognize', 'realize', 'see',
            'think', 'believe', 'consider', 'regard', 'view', 'perceive'
        },
        'NOUN': {
            'way', 'method', 'approach', 'manner', 'means', 'process', 'system',
            'way', 'path', 'route', 'course', 'direction', 'strategy', 'tactic',
            'thing', 'item', 'object', 'element', 'component', 'part', 'piece',
            'aspect', 'feature', 'characteristic', 'attribute', 'property',
            'type', 'kind', 'sort', 'category', 'class', 'group', 'set',
            'number', 'amount', 'quantity', 'level', 'degree', 'extent',
            'time', 'period', 'duration', 'moment', 'instance', 'occasion',
            'place', 'location', 'position', 'site', 'spot', 'area', 'region',
            'person', 'people', 'individual', 'person', 'human', 'man', 'woman',
            'group', 'team', 'organization', 'institution', 'company', 'firm',
            'problem', 'issue', 'challenge', 'difficulty', 'obstacle', 'barrier',
            'solution', 'answer', 'response', 'reaction', 'result', 'outcome',
            'effect', 'impact', 'influence', 'consequence', 'implication',
            'reason', 'cause', 'factor', 'element', 'component', 'aspect',
            'purpose', 'goal', 'objective', 'aim', 'target', 'intention',
            'information', 'data', 'fact', 'detail', 'point', 'note', 'remark'
        }
    }

    # Flatten all common words for quick lookup
    ALL_COMMON_WORDS = set()
    for pos_words in COMMON_WORDS.values():
        ALL_COMMON_WORDS.update(pos_words)

    def __init__(self, sample_text: str, nlp_model=None, similarity_threshold: float = 0.5, min_sample_frequency: int = 2):
        """
        Initialize mapper with sample text vocabulary.

        Args:
            sample_text: The target style sample to derive mappings from
            nlp_model: spaCy model (if None, will load en_core_web_sm)
            similarity_threshold: Minimum cosine similarity for mapping (0.0-1.0)
            min_sample_frequency: Minimum occurrences in sample to consider a word
        """
        # Load spaCy model
        if nlp_model is None:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                self.nlp = spacy.load("en_core_web_sm")
        else:
            self.nlp = nlp_model

        self.similarity_threshold = similarity_threshold
        self.min_sample_frequency = min_sample_frequency

        # Extract sample vocabulary
        self.sample_vocabulary: Dict[str, Dict[str, Tuple[np.ndarray, int]]] = defaultdict(dict)
        self._extract_sample_vocabulary(sample_text)

        # Build mappings from common words to sample equivalents
        self.mappings: Dict[str, Dict[str, str]] = defaultdict(dict)  # POS -> {common_word -> sample_word}
        self._build_mappings()

        print(f"  [SemanticWordMapper] Built {sum(len(m) for m in self.mappings.values())} word mappings")

    def _extract_sample_vocabulary(self, text: str):
        """Extract vocabulary from sample text with word vectors and frequencies."""
        doc = self.nlp(text)

        # Count words by POS and store vectors
        pos_words: Dict[str, Counter] = defaultdict(Counter)
        word_vectors: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)

        for token in doc:
            if token.is_alpha and not token.is_stop and len(token.text) > 2:
                word = token.lemma_.lower()
                pos = token.pos_

                # Skip if word has no vector
                if not token.has_vector:
                    continue

                pos_words[pos][word] += 1

                # Store vector (use first occurrence's vector)
                if word not in word_vectors[pos]:
                    word_vectors[pos][word] = token.vector

        # Build vocabulary dictionary with vectors and frequencies
        for pos, words in pos_words.items():
            for word, freq in words.items():
                if freq >= self.min_sample_frequency and word in word_vectors[pos]:
                    self.sample_vocabulary[pos][word] = (word_vectors[pos][word], freq)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """Get word vector from spaCy model."""
        token = self.nlp.vocab[word]
        if token.has_vector:
            return token.vector
        return None

    def _build_mappings(self):
        """Build mappings from common words to sample vocabulary."""
        for pos, common_words in self.COMMON_WORDS.items():
            if pos not in self.sample_vocabulary:
                continue

            sample_words = self.sample_vocabulary[pos]

            for common_word in common_words:
                common_vec = self._get_word_vector(common_word)
                if common_vec is None:
                    continue

                # Find best match in sample vocabulary
                best_match = None
                best_similarity = -1.0

                for sample_word, (sample_vec, freq) in sample_words.items():
                    # Skip if it's the same word
                    if common_word == sample_word:
                        continue

                    similarity = self._cosine_similarity(common_vec, sample_vec)
                    if similarity > best_similarity and similarity >= self.similarity_threshold:
                        best_similarity = similarity
                        best_match = sample_word

                if best_match:
                    self.mappings[pos][common_word] = best_match

    def map_word(self, word: str, pos: str) -> Optional[str]:
        """
        Get sample equivalent for a word.

        Args:
            word: Word to map (should be lemmatized, lowercase)
            pos: Part of speech tag

        Returns:
            Sample equivalent word or None if no mapping found
        """
        word_lower = word.lower()
        if pos in self.mappings and word_lower in self.mappings[pos]:
            return self.mappings[pos][word_lower]
        return None

    def apply_mapping(self, text: str) -> str:
        """
        Apply semantic mappings to text.

        Args:
            text: Text to process

        Returns:
            Text with common words replaced by sample equivalents
        """
        if not self.mappings:
            return text

        doc = self.nlp(text)
        result_tokens = []

        for token in doc:
            word = token.lemma_.lower()
            pos = token.pos_

            # Check if this word should be mapped
            mapped_word = self.map_word(word, pos)

            if mapped_word:
                # Preserve original case
                if token.text.isupper():
                    replacement = mapped_word.upper()
                elif token.text[0].isupper():
                    replacement = mapped_word.capitalize()
                else:
                    replacement = mapped_word.lower()

                # Preserve original token shape (punctuation, spacing)
                if token.whitespace_:
                    result_tokens.append(replacement + token.whitespace_)
                else:
                    result_tokens.append(replacement)
            else:
                # Keep original token
                result_tokens.append(token.text_with_ws)

        return ''.join(result_tokens)

    def get_mapping_stats(self) -> Dict[str, int]:
        """Get statistics about built mappings."""
        return {
            'total_mappings': sum(len(m) for m in self.mappings.values()),
            'adj_mappings': len(self.mappings.get('ADJ', {})),
            'verb_mappings': len(self.mappings.get('VERB', {})),
            'noun_mappings': len(self.mappings.get('NOUN', {})),
            'sample_vocab_size': sum(len(v) for v in self.sample_vocabulary.values())
        }


# Test function
if __name__ == '__main__':
    from pathlib import Path

    sample_path = Path(__file__).parent / "prompts" / "sample_mao.txt"
    if sample_path.exists():
        with open(sample_path, 'r', encoding='utf-8') as f:
            sample_text = f.read()

        print("=== Semantic Word Mapper Test ===\n")

        mapper = SemanticWordMapper(sample_text, similarity_threshold=0.5)

        print("\nMapping statistics:")
        stats = mapper.get_mapping_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

        print("\nSample mappings (first 20):")
        count = 0
        for pos, mappings in mapper.mappings.items():
            for common, sample in list(mappings.items())[:10]:
                print(f"  {pos}: {common} -> {sample}")
                count += 1
                if count >= 20:
                    break
            if count >= 20:
                break

        # Test text
        test_text = """
        This is an important method that shows significant results.
        The system uses various approaches to demonstrate effective solutions.
        Many people find this process very useful for their work.
        """

        print("\n\nOriginal text:")
        print(test_text)

        print("\nMapped text:")
        mapped = mapper.apply_mapping(test_text)
        print(mapped)
    else:
        print("No sample file found.")

