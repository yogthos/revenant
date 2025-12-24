"""Style Profiler: Forensic linguistic analysis for dynamic style extraction.

This module extracts a unique "voice fingerprint" from any text corpus by analyzing
POV, burstiness, vocabulary, sentence starters, and punctuation patterns.
"""

import re
import statistics
from collections import Counter
from typing import Dict, List, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.nlp_manager import NLPManager


class StyleProfiler:
    """Extracts style characteristics from text corpus using forensic linguistic analysis."""

    def __init__(self):
        """Initialize the style profiler."""
        self.nlp = NLPManager.get_nlp()
        # Cache for word filtering to avoid redundant NLP calls
        self._word_filter_cache = {}

    def _sanitize_text(self, text: str) -> str:
        """Sanitize and normalize input text before analysis.

        Removes artifacts like literal newline sequences, normalizes whitespace,
        and ensures clean text for processing.

        Args:
            text: Raw input text

        Returns:
            Sanitized text ready for analysis
        """
        if not text:
            return ""

        # 1. Replace literal "\n" strings with actual newlines (in case they're escaped)
        text = text.replace("\\n", "\n")
        text = text.replace("\\r", "\r")
        text = text.replace("\\t", "\t")

        # 2. Normalize line breaks (Windows, Mac, Unix)
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        # 3. Replace multiple newlines with single newline (preserve paragraph breaks)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 4. Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)

        # 5. Replace tabs with spaces
        text = text.replace("\t", " ")

        # 6. Remove control characters (except newlines and tabs which we've handled)
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')

        # 7. Strip leading/trailing whitespace from each line
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        text = '\n'.join(lines)

        # 8. Remove empty lines at start/end
        text = text.strip()

        # 9. Final cleanup: ensure no literal "\n" strings remain
        text = text.replace("\\n", " ")
        text = text.replace("\\r", " ")
        text = text.replace("\\t", " ")

        return text

    def analyze_style(self, text: str) -> Dict[str, Any]:
        """Analyze style characteristics from text corpus.

        Extracts 5 dimensions:
        1. POV & Address (Point of View)
        2. Burstiness (Jaggedness Index)
        3. Signature Vocabulary (Keywords)
        4. Connective Tissue (Sentence Starters)
        5. Punctuation Signature

        Args:
            text: Full corpus text to analyze

        Returns:
            Dictionary containing all style characteristics
        """
        if not text or not text.strip():
            return self._empty_profile()

        # Sanitize and normalize text before processing
        text = self._sanitize_text(text)

        if not text or not text.strip():
            return self._empty_profile()

        # Parse text with spaCy (limit to 200000 chars for performance and stats accuracy)
        doc = self.nlp(text[:200000])
        sentences = list(doc.sents)

        if not sentences:
            return self._empty_profile()

        # Build entity blocklist first (from original case-sensitive text)
        entity_blocklist = self._build_entity_blocklist(doc)

        # Extract all dimensions
        pov_data = self._analyze_pov(doc)
        burstiness_data = self._analyze_burstiness(sentences)
        keywords_data = self._analyze_keywords(doc, entity_blocklist)
        openers_data = self._analyze_sentence_starters(sentences, entity_blocklist)
        punctuation_data = self._analyze_punctuation(doc, len(sentences))
        vocabulary_palette = self._extract_vocabulary_palette(doc)
        structure_data = self._analyze_structure(sentences)  # NEW: Structural DNA

        # NEW: DNA Analysis
        dna = self._analyze_stylistic_dna(doc)

        return {
            **pov_data,
            **burstiness_data,
            **keywords_data,
            **openers_data,
            **punctuation_data,
            "vocabulary_palette": vocabulary_palette,
            "structural_dna": structure_data,  # NEW: Structural DNA
            "stylistic_dna": dna  # NEW: Stylistic DNA
        }

    def _build_entity_blocklist(self, doc) -> set:
        """
        Scans the document for Proper Nouns and Named Entities to create a
        strict blocklist for keyword extraction.

        Args:
            doc: spaCy Doc object (must be from original case-sensitive text)

        Returns:
            Set of lowercase words/lemmas to block
        """
        blocklist = set()

        for token in doc:
            # 1. Catch Standard Proper Nouns
            if token.pos_ == "PROPN":
                blocklist.add(token.text.lower())
                blocklist.add(token.lemma_.lower())

            # 2. Catch Named Entities (PERSON, ORG, GPE, etc.)
            # ent_type_ is often more accurate than POS for names
            if token.ent_type_ in {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "NORP"}:
                blocklist.add(token.text.lower())
                blocklist.add(token.lemma_.lower())

        return blocklist

    def _is_general_word(self, word: str, blocklist: set = None) -> bool:
        """Check if a word is a general word (not a proper noun or specific term).

        Args:
            word: Word to check (lowercase string) OR token object
            blocklist: Set of lowercase words/lemmas to block (from _build_entity_blocklist)

        Returns:
            True if word is general (should be kept), False if it's a proper noun/specific term (should be filtered)
        """
        # Handle both string and token inputs
        if hasattr(word, 'text'):
            # It's a token object
            token = word
            text_lower = token.text.lower()
            lemma_lower = token.lemma_.lower()
        else:
            # It's a string
            if not word or len(word) < 3:
                return False
            text_lower = word.lower()
            lemma_lower = text_lower  # For string input, use word as lemma
            token = None

        # 1. Check strict blocklist first (Names detected by NER from original text)
        if blocklist:
            if text_lower in blocklist or lemma_lower in blocklist:
                return False

        # If we have a token, use direct checks (more accurate)
        if token is not None:
            # 2. Standard Filters
            if token.is_stop or token.is_punct or token.is_digit or token.like_num:
                return False

            # 2.5. Filter out whitespace-only tokens and artifacts
            token_text = token.text.strip()
            if not token_text or len(token_text) < 2:
                return False
            # Filter out tokens that are only whitespace, newlines, or control characters
            if token_text.isspace() or any(ord(c) < 32 and c not in '\n\t' for c in token_text):
                return False
            # Filter out literal escape sequences
            if '\\n' in token_text or '\\r' in token_text or '\\t' in token_text:
                return False

            # 3. Fallback POS Check (in case NER missed it but it's tagged PROPN)
            if token.pos_ == "PROPN":
                return False

            # 4. Named Entity Check (double-check)
            if token.ent_type_ in {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "NORP"}:
                return False

            return True

        # For string input, use the old method (for backward compatibility)
        # Additional checks for string input
        if not text_lower or text_lower.isspace():
            return False
        # Filter out literal escape sequences in string input
        if '\\n' in text_lower or '\\r' in text_lower or '\\t' in text_lower:
            return False

        # Check cache first
        cache_key = f"{text_lower}:{bool(blocklist)}"
        if cache_key in self._word_filter_cache:
            return self._word_filter_cache[cache_key]

        # Default to True (conservative: include if unsure)
        result = True

        try:
            # Process word in minimal sentence context to get accurate POS tags
            test_sentence = f"The {text_lower}."
            doc = self.nlp(test_sentence)

            # Find the word token (skip "The" and punctuation)
            for token in doc:
                if token.text.lower() == text_lower and not token.is_punct:
                    # 1. Basic Filters
                    if token.is_stop or token.is_punct or token.is_digit or token.like_num:
                        result = False
                        break

                    # 2. Part of Speech Filter (Exclude Proper Nouns)
                    if token.pos_ == "PROPN":
                        result = False
                        break

                    # 3. Named Entity Filter
                    if token.ent_type_ in {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "NORP"}:
                        result = False
                        break

                    # Keep common parts of speech (general vocabulary)
                    if token.pos_ in ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "CONJ", "SCONJ", "ADP", "PART"]:
                        result = True
                        break
                    # For other POS tags, default to keeping (conservative)
                    result = True
                    break
        except Exception:
            # If processing fails, be conservative and include the word
            result = True

        # Cache the result
        self._word_filter_cache[cache_key] = result
        return result

    def _empty_profile(self) -> Dict[str, Any]:
        """Return an empty profile structure."""
        return {
            "pov": "Third Person",
            "pov_breakdown": {"first_singular": 0, "first_plural": 0, "third_person": 0},
            "burstiness": 0.0,
            "rhythm_desc": "Unknown",
            "keywords": [],
            "keyword_frequencies": {},
            "common_openers": [],
            "opener_pattern": "Unknown",
            "semicolons_per_100": 0.0,
            "dashes_per_100": 0.0,
            "exclamations_per_100": 0.0,
            "punctuation_preference": "Standard",
            "vocabulary_palette": {
                "general": [],
                "sensory_verbs": [],
                "connectives": [],
                "intensifiers": []
            },
            "structural_dna": {
                "avg_words_per_sentence": 0.0,
                "complexity_ratio": 0.0,
                "sentence_length_std_dev": 0.0
            },
            "stylistic_dna": {
                "force_imperfection": False,
                "use_fragments": False,
                "allow_contractions": True,
                "sensory_grounding": False,
                "banned_transitions": [],
                "allow_complex_connectors": True,
                "force_active_voice": False,
                "allow_intro_participles": True,
                "allow_relative_clauses": True,
                "allow_serial_gerunds": True,
                "sentence_structure": "balanced"
            }
        }

    def _analyze_pov(self, doc) -> Dict[str, Any]:
        """Analyze Point of View by counting pronouns.

        Args:
            doc: spaCy document

        Returns:
            Dictionary with POV classification and breakdown
        """
        from src.utils.spacy_linguistics import get_pov_pronouns

        # Get POV pronouns using spaCy
        pov_dict = get_pov_pronouns(doc)
        first_singular = pov_dict["first_singular"]
        first_plural = pov_dict["first_plural"]
        third_person = pov_dict["third_person"]

        counts = {
            "first_singular": 0,
            "first_plural": 0,
            "third_person": 0
        }

        # Count pronouns (case-insensitive)
        for token in doc:
            if token.pos_ == "PRON" or token.pos_ == "DET":
                token_lower = token.text.lower()
                if token_lower in first_singular:
                    counts["first_singular"] += 1
                elif token_lower in first_plural:
                    counts["first_plural"] += 1
                elif token_lower in third_person:
                    counts["third_person"] += 1

        # Determine dominant POV
        max_count = max(counts.values())
        if max_count == 0:
            pov = "Third Person"  # Default
        elif counts["first_singular"] == max_count:
            pov = "First Person Singular"
        elif counts["first_plural"] == max_count:
            pov = "First Person Plural"
        else:
            pov = "Third Person"

        return {
            "pov": pov,
            "pov_breakdown": counts
        }

    def _analyze_burstiness(self, sentences: List) -> Dict[str, Any]:
        """Analyze sentence length variation (burstiness).

        Args:
            sentences: List of spaCy sentence spans

        Returns:
            Dictionary with burstiness score and description
        """
        if not sentences or len(sentences) < 2:
            return {
                "burstiness": 0.0,
                "rhythm_desc": "Unknown"
            }

        # Calculate sentence lengths (in words)
        sentence_lengths = [len([token for token in sent if not token.is_punct]) for sent in sentences]

        if not sentence_lengths:
            return {
                "burstiness": 0.0,
                "rhythm_desc": "Unknown"
            }

        mean_len = np.mean(sentence_lengths)
        if mean_len == 0:
            return {
                "burstiness": 0.0,
                "rhythm_desc": "Unknown"
            }

        std_dev = np.std(sentence_lengths)
        burstiness = std_dev / mean_len  # Coefficient of Variation

        # Classify
        if burstiness < 0.4:
            rhythm_desc = "Smooth/Monotonous"
        elif burstiness < 0.6:
            rhythm_desc = "Moderate Variation"
        else:
            rhythm_desc = "Jagged/Volatile"

        return {
            "burstiness": round(float(burstiness), 3),
            "rhythm_desc": rhythm_desc
        }

    def _analyze_keywords(self, doc, entity_blocklist: set) -> Dict[str, Any]:
        """Extract signature vocabulary using token-level frequency analysis.

        Args:
            doc: spaCy Doc object (from original case-sensitive text)
            entity_blocklist: Set of lowercase words/lemmas to block

        Returns:
            Dictionary with keywords and frequencies
        """
        if not doc or len(doc) < 10:
            return {
                "keywords": [],
                "keyword_frequencies": {}
            }

        try:
            # Count words using token-level filtering with blocklist
            word_counts = Counter()
            for token in doc:
                # Use token-level check with blocklist
                if self._is_general_word(token, entity_blocklist):
                    lemma = token.lemma_.lower()
                    word_counts[lemma] += 1

            # Calculate frequencies (Total clean words)
            total_words = sum(word_counts.values())
            if total_words == 0:
                return {
                    "keywords": [],
                    "keyword_frequencies": {}
                }

            # Get top 30 keywords with frequencies
            top_keywords = word_counts.most_common(30)
            # Filter out any remaining artifacts (whitespace, escape sequences, etc.)
            keywords = []
            keyword_frequencies = {}
            for word, count in top_keywords:
                # Final sanitization check
                word_clean = word.strip()
                if (word_clean and
                    len(word_clean) >= 2 and
                    not word_clean.isspace() and
                    '\\n' not in word_clean and
                    '\\r' not in word_clean and
                    '\\t' not in word_clean and
                    not any(ord(c) < 32 and c not in '\n\t' for c in word_clean)):
                    keywords.append(word_clean)
                    keyword_frequencies[word_clean] = count / total_words

            return {
                "keywords": keywords,
                "keyword_frequencies": keyword_frequencies
            }
        except Exception as e:
            # Fallback: return empty if processing fails
            return {
                "keywords": [],
                "keyword_frequencies": {}
            }

    def _extract_vocabulary_palette(self, doc) -> Dict[str, List[str]]:
        """
        Extracts the author's preferred words categorized by function.

        Args:
            doc: spaCy Doc object

        Returns:
            Dictionary with categorized vocabulary lists (lemmas)
        """
        palette = {
            "general": [],      # High-frequency content words (Nouns/Adjs/Verbs)
            "sensory_verbs": [], # Verbs related to perception
            "connectives": [],   # Transition words
            "intensifiers": []  # Adverbs modifying adjectives
        }

        # Specialized lists
        sensory_lemmas = {'see', 'hear', 'feel', 'grasp', 'watch', 'listen', 'touch', 'smell', 'sense', 'perceive'}
        connective_deps = {'cc', 'mark', 'advmod'}

        word_counts = Counter()

        for token in doc:
            # 1. Basic Filters
            if token.is_stop or token.is_punct or token.is_digit or token.like_num:
                continue

            # 2. Part of Speech Filter (Exclude Proper Nouns)
            # PROPN = Proper Noun (e.g., "Rainer", "London")
            if token.pos_ == "PROPN":
                continue

            # 3. Named Entity Filter (Double-check)
            # Sometimes 'Rainer' might be tagged as NOUN if lowercased,
            # so we check if it's part of a named entity (PERSON, ORG, GPE).
            if token.ent_type_ in {"PERSON", "ORG", "GPE", "LOC", "PRODUCT"}:
                continue

            lemma = token.lemma_.lower()

            # Note: We skip the _is_general_word check here because we've already
            # done all the necessary filtering directly on the token (PROPN, named entities, etc.).
            # The _is_general_word method is used for keyword extraction where we only have
            # the word string and need to test it in isolation.

            # 1. Sensory Verbs
            if token.pos_ == "VERB" and lemma in sensory_lemmas:
                if lemma not in palette["sensory_verbs"]:
                    palette["sensory_verbs"].append(lemma)

            # 2. Connectives (simplified heuristic)
            if token.dep_ in connective_deps and token.pos_ in {'ADV', 'CCONJ', 'SCONJ'}:
                if lemma not in palette["connectives"]:
                    palette["connectives"].append(lemma)

            # 3. Intensifiers (adverbs modifying adjectives)
            if token.dep_ == "advmod" and token.head.pos_ == "ADJ":
                if lemma not in palette["intensifiers"]:
                    palette["intensifiers"].append(lemma)

            # 4. General Palette (Nouns/Adjs/Verbs)
            if token.pos_ in {'NOUN', 'VERB', 'ADJ'}:
                word_counts[lemma] += 1

        # Keep top N for general palette
        palette["general"] = [w for w, c in word_counts.most_common(50)]

        # Deduplicate lists (already done above, but ensure)
        palette["sensory_verbs"] = list(set(palette["sensory_verbs"]))
        palette["connectives"] = list(set(palette["connectives"]))
        palette["intensifiers"] = list(set(palette["intensifiers"]))

        return palette

    def _analyze_sentence_starters(self, sentences: List, entity_blocklist: set) -> Dict[str, Any]:
        """Extract common sentence openers (smart extraction skipping punctuation).

        Args:
            sentences: List of spaCy sentence spans
            entity_blocklist: Set of lowercase words/lemmas to block

        Returns:
            Dictionary with common openers and pattern classification
        """
        if not sentences:
            return {
                "common_openers": [],
                "opener_pattern": "Unknown"
            }

        openers = []
        # Use token-level extraction to get accurate POS and entity info
        for sent in sentences:
            # Get first non-punctuation token
            for token in sent:
                if not token.is_punct:
                    # Check if it's a general word (not a name) using blocklist
                    if self._is_general_word(token, entity_blocklist):
                        openers.append(token.text.lower())
                    break  # Only take the first valid token

        if not openers:
            return {
                "common_openers": [],
                "opener_pattern": "Unknown"
            }

        # Count and get top 10
        opener_counts = Counter(openers)
        top_openers = [word for word, count in opener_counts.most_common(10)]

        # Classify pattern
        from src.utils.spacy_linguistics import get_discourse_markers, get_conjunctions

        # Get doc from first sentence if available
        doc = sentences[0].doc if sentences else None

        if doc is not None:
            # Get discourse markers using spaCy
            discourse_markers_list = get_discourse_markers(doc)
            conjunctions_list = get_conjunctions(doc)

            # Convert to sets for membership testing
            logical_markers = set(discourse_markers_list)  # Discourse markers are typically logical
            narrative_markers = set(conjunctions_list)  # Conjunctions are typically narrative
        else:
            # Fallback to hardcoded lists if doc not available
            logical_markers = {"therefore", "however", "moreover", "furthermore", "consequently", "thus", "hence", "accordingly", "nevertheless", "nonetheless"}
            narrative_markers = {"and", "but", "then", "so", "or", "nor", "yet"}

        logical_count = sum(1 for opener in top_openers if opener in logical_markers)
        narrative_count = sum(1 for opener in top_openers if opener in narrative_markers)

        if logical_count > narrative_count:
            opener_pattern = "Logical"
        elif narrative_count > logical_count:
            opener_pattern = "Narrative"
        else:
            opener_pattern = "Mixed"

        return {
            "common_openers": top_openers,
            "opener_pattern": opener_pattern
        }

    def _analyze_punctuation(self, doc, num_sentences: int) -> Dict[str, Any]:
        """Analyze punctuation signature.

        Args:
            doc: spaCy document
            num_sentences: Number of sentences in the document

        Returns:
            Dictionary with punctuation statistics
        """
        if num_sentences == 0:
            return {
                "semicolons_per_100": 0.0,
                "dashes_per_100": 0.0,
                "exclamations_per_100": 0.0,
                "punctuation_preference": "Standard"
            }

        # Count punctuation marks
        semicolons = sum(1 for token in doc if token.text == ';')
        dashes = sum(1 for token in doc if token.text in ['—', '–', '-'] and token.pos_ == 'PUNCT')
        exclamations = sum(1 for token in doc if token.text == '!')

        # Calculate per 100 sentences
        multiplier = 100.0 / num_sentences
        semicolons_per_100 = semicolons * multiplier
        dashes_per_100 = dashes * multiplier
        exclamations_per_100 = exclamations * multiplier

        # Determine preference
        max_punct = max(semicolons_per_100, dashes_per_100, exclamations_per_100)
        if max_punct < 1.0:
            preference = "Standard"
        elif semicolons_per_100 == max_punct:
            preference = "Semicolons"
        elif dashes_per_100 == max_punct:
            preference = "Dashes"
        elif exclamations_per_100 == max_punct:
            preference = "Exclamations"
        else:
            preference = "Standard"

        return {
            "semicolons_per_100": round(semicolons_per_100, 2),
            "dashes_per_100": round(dashes_per_100, 2),
            "exclamations_per_100": round(exclamations_per_100, 2),
            "punctuation_preference": preference
        }

    def _analyze_structure(self, sentences: List) -> Dict[str, Any]:
        """Analyze structural characteristics of sentences.

        Calculates average words per sentence, complexity ratio, and sentence length
        standard deviation for use in dynamic inflation calculation.

        Args:
            sentences: List of spaCy sentence spans

        Returns:
            Dictionary with structural_dna metrics
        """
        if not sentences:
            return {
                "avg_words_per_sentence": 0.0,
                "complexity_ratio": 0.0,
                "sentence_length_std_dev": 0.0
            }

        # Calculate sentence lengths (in words, excluding punctuation)
        sentence_lengths = [len([token for token in sent if not token.is_punct]) for sent in sentences]

        if not sentence_lengths:
            return {
                "avg_words_per_sentence": 0.0,
                "complexity_ratio": 0.0,
                "sentence_length_std_dev": 0.0
            }

        avg_len = np.mean(sentence_lengths)

        # Calculate complexity ratio (sentences with commas or semicolons)
        complex_count = sum(1 for sent in sentences if ',' in sent.text or ';' in sent.text)
        complexity_ratio = complex_count / len(sentences) if sentences else 0.0

        # Calculate standard deviation for burstiness helper
        std_dev = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0.0

        return {
            "avg_words_per_sentence": round(float(avg_len), 1),
            "complexity_ratio": round(float(complexity_ratio), 2),
            "sentence_length_std_dev": round(float(std_dev), 1)
        }

    def _get_tree_depth(self, token) -> int:
        """Recursively calculate the maximum depth of a dependency tree from a token.

        Args:
            token: spaCy token (usually the root of a sentence)

        Returns:
            Maximum depth of the dependency tree (0 for simple sentences)
        """
        if not list(token.children):
            return 0
        return 1 + max((self._get_tree_depth(child) for child in token.children), default=0)

    def _analyze_stylistic_dna(self, doc) -> Dict[str, Any]:
        """
        Infers 'Human Texture' parameters (Voice, Rhythm, Connectors, Openers) from the doc.

        Analyzes the corpus to determine:
        - allow_complex_connectors: Whether to allow academic transitions
        - force_active_voice: Whether to enforce active voice
        - allow_intro_participles: Whether to allow sentences starting with -ing participles
        - sentence_structure: jagged, flowing, or balanced
        - avg_depth: Average dependency tree depth

        Args:
            doc: spaCy document object

        Returns:
            Dictionary with stylistic_dna parameters
        """
        sentences = list(doc.sents)
        total_sents = len(sentences)
        if total_sents == 0:
            return {
                "allow_complex_connectors": True,
                "force_active_voice": False,
                "allow_intro_participles": True,
                "allow_relative_clauses": True,
                "allow_serial_gerunds": True,
                "sentence_structure": "balanced",
                "force_imperfection": True,
                "sensory_grounding": True
            }

        # 1. Connector Analysis ("Academic Glue")
        # Academic: transition words that imply formal logical flow
        academic_connectors = {'moreover', 'thus', 'therefore', 'hence', 'thereby', 'furthermore', 'consequently'}
        # Simple: strong/direct pivots
        simple_connectors = {'but', 'and', 'yet', 'so'}

        academic_count = 0

        for token in doc:
            if token.lower_ in academic_connectors:
                academic_count += 1

        # If academic usage is frequent (> 1% of sentences), allow them. Otherwise ban.
        # Mao/Hemingway: Low. Marx/Academic: High.
        allow_complex_connectors = (academic_count / total_sents) > 0.01

        # 2. Voice Analysis (Adjusted Threshold)
        # Standard English is ~15-20% passive. AI is often higher.
        # We want to force ACTIVE voice unless the author is extremely passive (>20%).
        # Count passive auxiliaries (e.g., "was" in "was eaten")
        passive_count = sum(1 for token in doc if token.dep_ == "auxpass")
        passive_ratio = passive_count / total_sents if total_sents > 0 else 0

        # If passive ratio is low (< 20%), force active voice instructions
        force_active_voice = passive_ratio < 0.20

        # 2a. Relative Clause Analysis (The "Which/That" Glue)
        # Count 'relcl' dependencies with specific words
        rel_clause_count = sum(1 for token in doc if token.dep_ == "relcl" and token.text.lower() in ['which', 'who', 'that'])
        # If relative clauses are rare (< 1% of tokens), ban them to force simple syntax
        allow_relative_clauses = (rel_clause_count / len(doc)) > 0.01 if len(doc) > 0 else True

        # 2b. Serial Gerund Analysis (The "Laundry List" Pattern)
        # Pattern: VBG -> conj -> VBG (e.g., "running, jumping, and playing")
        serial_gerund_count = 0
        for token in doc:
            if token.tag_ == "VBG" and token.dep_ == "conj":
                if token.head.tag_ == "VBG":
                    serial_gerund_count += 1
        # Strict Ban: If author uses < 1 per 200 sentences, ban it.
        allow_serial_gerunds = (serial_gerund_count / total_sents) > 0.005 if total_sents > 0 else True

        # 3. Opener Analysis (The "Serving as..." Test)
        # Count sentences starting with -ing participles (VBG tag with advcl/acl dependency)
        intro_participle_count = 0
        for sent in sentences:
            if len(sent) > 0 and sent[0].tag_ == "VBG" and sent[0].dep_ in ("advcl", "acl"):
                intro_participle_count += 1
        allow_intro_participles = (intro_participle_count / total_sents) > 0.02 if total_sents > 0 else True

        # 4. Rhythm Analysis (Jagged vs Flowing)
        # Jagged = High standard deviation in sentence length + frequent semicolons/stops
        lengths = [len([token for token in sent if not token.is_punct]) for sent in sentences]

        if len(lengths) > 1:
            length_std_dev = statistics.stdev(lengths)
        else:
            length_std_dev = 0

        avg_len = statistics.mean(lengths) if lengths else 0

        semicolon_count = sum(1 for token in doc if token.text == ';')
        semicolons_per_sent = semicolon_count / total_sents if total_sents > 0 else 0

        if length_std_dev > 12 or semicolons_per_sent > 0.15:
            structure = "jagged"  # Mao style (stops and pivots)
        elif avg_len > 25:
            structure = "flowing"  # Marx/Academic style (subordination)
        else:
            structure = "balanced"  # Standard

        # 5. Tree Depth Analysis (Dependency Parse Depth)
        # Measure average dependency tree depth to determine target complexity
        depths = []
        for sent in sentences:
            root = sent.root
            depth = self._get_tree_depth(root)
            depths.append(depth)

        avg_depth = statistics.mean(depths) if depths else 3.0

        # Add rhetoric analysis
        rhetoric = self._analyze_rhetoric(doc)

        return {
            "allow_complex_connectors": allow_complex_connectors,
            "force_active_voice": force_active_voice,
            "allow_intro_participles": allow_intro_participles,
            "allow_relative_clauses": allow_relative_clauses,
            "allow_serial_gerunds": allow_serial_gerunds,
            "sentence_structure": structure,
            "structural_stats": {
                "avg_words_per_sentence": round(avg_len, 1),
                "complexity_ratio": round(length_std_dev / avg_len, 2) if avg_len else 0,
                "avg_depth": round(avg_depth, 1)
            },
            # Default Toggles (can be refined later)
            "force_imperfection": True,
            "sensory_grounding": True,
            "rhetoric": rhetoric  # Add rhetoric analysis
        }

    def _analyze_rhetoric(self, doc) -> Dict[str, bool]:
        """
        Detects rhetorical devices (Anaphora, Asyndeton) from the corpus.

        Args:
            doc: spaCy document object

        Returns:
            Dictionary with rhetoric flags (use_anaphora, use_asyndeton)
        """
        sentences = list(doc.sents)
        if len(sentences) < 2:
            return {
                "use_anaphora": False,
                "use_asyndeton": False
            }

        # 1. ANAPHORA (Repeating the start of sentences)
        # Compare first tokens of consecutive sentences
        anaphora_count = 0
        exclusion_list = ["the", "a", "an", "and", "but", "or"]  # Minimal, conservative list

        for i in range(1, len(sentences)):
            prev_sent = sentences[i-1]
            curr_sent = sentences[i]

            # Get first non-punctuation token from each sentence
            def get_start_token(sent):
                for token in sent:
                    if not token.is_punct and not token.is_space:
                        return token.text.lower()
                return None

            prev_start = get_start_token(prev_sent)
            curr_start = get_start_token(curr_sent)

            # If first words match (and not common starters), count as anaphora
            if prev_start and curr_start and prev_start == curr_start:
                if prev_start not in exclusion_list:
                    anaphora_count += 1

        # Threshold: > 2% of sentence pairs (e.g., 1 in 50 pairs)
        use_anaphora = (anaphora_count / (len(sentences) - 1)) > 0.02 if len(sentences) > 1 else False

        # 2. ASYNDETON (Clauses without conjunctions)
        # Check for sentences with commas/semicolons but NO 'and/but/or'
        asyndeton_count = 0
        for sent in sentences:
            has_comma_semi = any(t.text in [',', ';'] for t in sent)
            has_conjunction = any(t.dep_ == 'cc' or t.text.lower() in ['and', 'but', 'or'] for t in sent)
            if has_comma_semi and not has_conjunction:
                asyndeton_count += 1

        # Threshold: > 10% of sentences
        use_asyndeton = (asyndeton_count / len(sentences)) > 0.10 if len(sentences) > 0 else False

        return {
            "use_anaphora": use_anaphora,
            "use_asyndeton": use_asyndeton
        }

