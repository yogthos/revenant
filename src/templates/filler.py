"""Slot filling for template-based style transfer.

Fills template skeletons with content from propositions while:
- Mapping general words to author's vocabulary
- Preserving technical terms
- Maintaining grammatical correctness
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any

from ..utils.nlp import get_nlp, split_into_sentences
from ..utils.logging import get_logger
from .models import (
    SentenceTemplate,
    TemplateSlot,
    SlotType,
    VocabularyProfile,
)
from .vocabulary import (
    WordClassifier,
    TechnicalTermExtractor,
    GeneralWordMapper,
    WordClassification,
    WordType,
)

logger = get_logger(__name__)


@dataclass
class Proposition:
    """A semantic proposition to be expressed."""
    subject: str
    predicate: str  # The verb/action
    object: Optional[str] = None
    modifiers: List[str] = field(default_factory=list)
    prepositional_phrases: List[str] = field(default_factory=list)
    clauses: List[str] = field(default_factory=list)

    # Technical terms to preserve
    technical_terms: Set[str] = field(default_factory=set)

    @classmethod
    def from_sentence(cls, sentence: str) -> "Proposition":
        """Extract a proposition from a sentence using spaCy.

        Args:
            sentence: The sentence to parse.

        Returns:
            Proposition with extracted components.
        """
        nlp = get_nlp()
        doc = nlp(sentence)

        subject = ""
        predicate = ""
        obj = ""
        modifiers = []
        prep_phrases = []
        clauses = []

        # Find root verb
        root = None
        for token in doc:
            if token.dep_ == "ROOT":
                root = token
                predicate = token.text
                break

        if not root:
            # No clear root, return sentence as subject
            return cls(subject=sentence, predicate="")

        # Extract subject
        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass"):
                # Get full subtree
                subject = " ".join([t.text for t in token.subtree])
                break

        # Extract objects
        objects = []
        for token in doc:
            if token.dep_ in ("dobj", "pobj", "attr"):
                obj_text = " ".join([t.text for t in token.subtree])
                objects.append(obj_text)

        obj = objects[0] if objects else ""

        # Extract modifiers
        for token in doc:
            if token.pos_ in ("ADJ", "ADV") and token.dep_ not in ("ROOT",):
                modifiers.append(token.text)

        # Extract prepositional phrases
        for token in doc:
            if token.dep_ == "prep":
                pp = " ".join([t.text for t in token.subtree])
                prep_phrases.append(pp)

        # Extract subordinate clauses
        for token in doc:
            if token.dep_ in ("advcl", "relcl", "ccomp", "xcomp"):
                clause = " ".join([t.text for t in token.subtree])
                clauses.append(clause)

        # Extract technical terms
        extractor = TechnicalTermExtractor()
        technical_terms = extractor.extract(sentence)

        return cls(
            subject=subject,
            predicate=predicate,
            object=obj,
            modifiers=modifiers,
            prepositional_phrases=prep_phrases,
            clauses=clauses,
            technical_terms=technical_terms,
        )


@dataclass
class FilledSlot:
    """A slot that has been filled with content."""
    slot_type: SlotType
    original_placeholder: str
    filled_content: str
    words_substituted: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class FilledSentence:
    """A sentence with all slots filled."""
    template: SentenceTemplate
    text: str
    filled_slots: List[FilledSlot]
    propositions_used: List[Proposition]
    technical_terms_preserved: Set[str] = field(default_factory=set)


class SlotFiller:
    """Fills template slots with content from propositions."""

    # Regex to find placeholders like [SUBJECT], [VERB], etc.
    PLACEHOLDER_PATTERN = re.compile(r'\[([A-Z_]+)\]')

    def __init__(
        self,
        vocabulary_profile: Optional[VocabularyProfile] = None,
        technical_terms: Optional[Set[str]] = None
    ):
        """Initialize slot filler.

        Args:
            vocabulary_profile: Author's vocabulary for word substitution.
            technical_terms: Terms to preserve (not substitute).
        """
        self._nlp = None
        self.vocabulary = vocabulary_profile
        self.mapper = GeneralWordMapper(vocabulary_profile) if vocabulary_profile else None
        self.classifier = WordClassifier()
        self.global_technical_terms = technical_terms or set()

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def fill_template(
        self,
        template: SentenceTemplate,
        propositions: List[Proposition],
        used_words: Optional[Set[str]] = None
    ) -> Optional[FilledSentence]:
        """Fill a template with content from propositions.

        Args:
            template: The template to fill.
            propositions: Propositions to extract content from.
            used_words: Words already used (to avoid repetition).

        Returns:
            FilledSentence or None if filling fails.
        """
        used_words = used_words or set()
        skeleton = template.skeleton

        logger.debug(f"Filling template: '{skeleton[:50]}...'")
        logger.debug(f"  Propositions: {len(propositions)}, Used words: {len(used_words)}")

        # Find all placeholders in skeleton
        placeholders = self._find_placeholders(skeleton)
        logger.debug(f"  Found {len(placeholders)} placeholders: {[p[1].value for p in placeholders]}")

        if not placeholders:
            # No placeholders, return skeleton as-is
            return FilledSentence(
                template=template,
                text=skeleton,
                filled_slots=[],
                propositions_used=propositions,
            )

        # Merge technical terms from all propositions
        all_technical = self.global_technical_terms.copy()
        for prop in propositions:
            all_technical.update(prop.technical_terms)

        # Build content pool from propositions
        content_pool = self._build_content_pool(propositions)

        # Track used content to avoid repetition
        used_content: Set[str] = set()

        # Fill each placeholder
        filled_slots = []
        result = skeleton

        for placeholder, slot_type in placeholders:
            content = self._select_content_for_slot(
                slot_type, content_pool, all_technical, used_words, used_content
            )

            if content:
                # Mark content as used
                used_content.add(content.lower())
                logger.debug(f"  Slot {placeholder}: selected '{content[:30]}...'")

                # Apply vocabulary mapping to general words
                mapped_content, substitutions = self._apply_vocabulary_mapping(
                    content, all_technical, used_words
                )

                if substitutions:
                    logger.debug(f"    Vocab substitutions: {substitutions[:3]}")

                filled_slot = FilledSlot(
                    slot_type=slot_type,
                    original_placeholder=placeholder,
                    filled_content=mapped_content,
                    words_substituted=substitutions,
                )
                filled_slots.append(filled_slot)

                # Replace placeholder in result
                result = result.replace(placeholder, mapped_content, 1)

                # Track used words
                for word in mapped_content.split():
                    used_words.add(word.lower())
            else:
                # No content available, remove placeholder entirely
                logger.debug(f"  Slot {placeholder}: no content available, removing")
                result = result.replace(placeholder, "", 1)

        # Clean up the result
        result = self._clean_sentence(result)

        logger.debug(f"  Final result: '{result[:60]}...'")

        return FilledSentence(
            template=template,
            text=result,
            filled_slots=filled_slots,
            propositions_used=propositions,
            technical_terms_preserved=all_technical,
        )

    def fill_from_sentence(
        self,
        template: SentenceTemplate,
        source_sentence: str,
        used_words: Optional[Set[str]] = None
    ) -> Optional[FilledSentence]:
        """Fill a template using content from a source sentence.

        Args:
            template: The template to fill.
            source_sentence: Source sentence to extract content from.
            used_words: Words already used.

        Returns:
            FilledSentence or None if filling fails.
        """
        proposition = Proposition.from_sentence(source_sentence)
        return self.fill_template(template, [proposition], used_words)

    def _find_placeholders(
        self,
        skeleton: str
    ) -> List[Tuple[str, SlotType]]:
        """Find all placeholders in a skeleton.

        Args:
            skeleton: Template skeleton string.

        Returns:
            List of (placeholder_string, SlotType) tuples.
        """
        placeholders = []

        for match in self.PLACEHOLDER_PATTERN.finditer(skeleton):
            placeholder = match.group(0)  # e.g., "[SUBJECT]"
            type_name = match.group(1).lower()  # e.g., "subject"

            try:
                slot_type = SlotType(type_name)
            except ValueError:
                # Unknown slot type, default to SUBJECT
                slot_type = SlotType.SUBJECT

            placeholders.append((placeholder, slot_type))

        return placeholders

    def _build_content_pool(
        self,
        propositions: List[Proposition]
    ) -> Dict[SlotType, List[str]]:
        """Build a pool of content items by slot type.

        Args:
            propositions: Source propositions.

        Returns:
            Dictionary mapping SlotType to list of content items.
        """
        pool = {slot_type: [] for slot_type in SlotType}

        for prop in propositions:
            if prop.subject:
                pool[SlotType.SUBJECT].append(prop.subject)

            if prop.predicate:
                pool[SlotType.VERB].append(prop.predicate)

            if prop.object:
                pool[SlotType.OBJECT].append(prop.object)

            for modifier in prop.modifiers:
                pool[SlotType.MODIFIER].append(modifier)

            for pp in prop.prepositional_phrases:
                pool[SlotType.PREPOSITIONAL].append(pp)

            for clause in prop.clauses:
                pool[SlotType.CLAUSE].append(clause)

        # Log pool contents
        non_empty = {k.value: len(v) for k, v in pool.items() if v}
        logger.debug(f"  Content pool: {non_empty}")

        return pool

    def _select_content_for_slot(
        self,
        slot_type: SlotType,
        content_pool: Dict[SlotType, List[str]],
        technical_terms: Set[str],
        used_words: Set[str],
        used_content: Set[str] = None
    ) -> Optional[str]:
        """Select the best content for a slot.

        Args:
            slot_type: Type of slot to fill.
            content_pool: Available content by type.
            technical_terms: Terms to preserve.
            used_words: Words to avoid.
            used_content: Content already used (to avoid repetition).

        Returns:
            Selected content or None.
        """
        used_content = used_content or set()
        candidates = content_pool.get(slot_type, [])

        if not candidates:
            # Try fallback slot types
            fallbacks = self._get_fallback_types(slot_type)
            for fallback in fallbacks:
                candidates = content_pool.get(fallback, [])
                if candidates:
                    break

        if not candidates:
            return None

        # Filter out already used content
        candidates = [c for c in candidates if c.lower() not in used_content]

        if not candidates:
            return None

        # Score candidates based on:
        # 1. Contains technical terms (prefer)
        # 2. Length appropriateness
        # 3. Not already used words

        best_candidate = None
        best_score = -1

        for candidate in candidates:
            score = 0

            # Bonus for containing technical terms
            for term in technical_terms:
                if term.lower() in candidate.lower():
                    score += 2

            # Penalty for word repetition
            candidate_words = set(w.lower() for w in candidate.split())
            overlap = candidate_words & used_words
            score -= len(overlap) * 0.5

            # Prefer appropriate length
            words = candidate.split()
            if 2 <= len(words) <= 10:
                score += 1

            if score > best_score:
                best_score = score
                best_candidate = candidate

        return best_candidate

    def _get_fallback_types(self, slot_type: SlotType) -> List[SlotType]:
        """Get fallback slot types when primary is empty.

        Args:
            slot_type: Primary slot type.

        Returns:
            List of fallback types to try.
        """
        fallbacks = {
            SlotType.SUBJECT: [SlotType.OBJECT],
            SlotType.OBJECT: [SlotType.SUBJECT],
            SlotType.MODIFIER: [SlotType.PREPOSITIONAL],
            SlotType.PREPOSITIONAL: [SlotType.MODIFIER, SlotType.CLAUSE],
            SlotType.CLAUSE: [SlotType.PREPOSITIONAL],
            SlotType.VERB: [],
            SlotType.CONNECTOR: [],
            SlotType.COMPLEMENT: [SlotType.OBJECT],
        }
        return fallbacks.get(slot_type, [])

    def _apply_vocabulary_mapping(
        self,
        content: str,
        technical_terms: Set[str],
        used_words: Set[str]
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """Apply vocabulary mapping to content.

        Args:
            content: Content to map.
            technical_terms: Terms to preserve.
            used_words: Words to avoid.

        Returns:
            Tuple of (mapped_content, list of substitutions).
        """
        if not self.mapper:
            return content, []

        technical_lower = {t.lower() for t in technical_terms}
        substitutions = []

        doc = self.nlp(content)
        result_tokens = []

        for token in doc:
            word = token.text
            word_lower = word.lower()

            # Preserve technical terms
            if word_lower in technical_lower or word in technical_terms:
                result_tokens.append(word)
                continue

            # Preserve punctuation and whitespace
            if token.is_punct or token.is_space:
                result_tokens.append(word)
                continue

            # Classify word
            classification = self.classifier.classify(word, content)

            # Only map general words
            if classification.word_type != WordType.GENERAL:
                result_tokens.append(word)
                continue

            # Try to find mapping
            mapping = self.mapper.map_word(word, token.pos_, exclude=used_words)

            if mapping and mapping.similarity > 0.4:
                # Use mapped word, preserve case
                mapped = mapping.target_word
                if word[0].isupper():
                    mapped = mapped.capitalize()

                result_tokens.append(mapped)
                substitutions.append((word, mapped))
                used_words.add(mapped.lower())
            else:
                result_tokens.append(word)

        # Reconstruct with proper spacing
        result = self._reconstruct_text(result_tokens, doc)
        return result, substitutions

    def _reconstruct_text(self, tokens: List[str], doc) -> str:
        """Reconstruct text from tokens with proper spacing.

        Args:
            tokens: List of token strings.
            doc: Original spaCy doc for spacing info.

        Returns:
            Reconstructed string.
        """
        if not tokens:
            return ""

        result = tokens[0]
        for i, token in enumerate(tokens[1:], 1):
            # Check if we need a space before this token
            if i < len(doc) and doc[i].is_punct:
                # No space before punctuation
                result += token
            else:
                result += " " + token

        return result

    def _get_generic_filler(self, slot_type: SlotType) -> str:
        """Get generic filler for a slot type.

        Args:
            slot_type: Type of slot.

        Returns:
            Generic filler text.
        """
        fillers = {
            SlotType.SUBJECT: "this",
            SlotType.VERB: "is",
            SlotType.OBJECT: "it",
            SlotType.MODIFIER: "",
            SlotType.CLAUSE: "",
            SlotType.CONNECTOR: "",
            SlotType.PREPOSITIONAL: "",
            SlotType.COMPLEMENT: "",
        }
        return fillers.get(slot_type, "")

    def _clean_sentence(self, text: str) -> str:
        """Clean up a filled sentence.

        Args:
            text: Raw filled text.

        Returns:
            Cleaned text.
        """
        # Normalize whitespace
        text = " ".join(text.split())

        # Remove empty placeholders that weren't filled
        text = re.sub(r'\[\w+\]', '', text)

        # Fix double commas
        text = re.sub(r',\s*,+', ',', text)

        # Fix double periods
        text = re.sub(r'\.\s*\.+', '.', text)

        # Fix double punctuation
        text = re.sub(r'([.!?])\s*\1+', r'\1', text)

        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)

        # Fix punctuation followed by space then punctuation
        text = re.sub(r'([.,;:])\s+([.,;:])', r'\1', text)

        # Remove leading/trailing commas and extra spaces
        text = re.sub(r'^[,\s]+', '', text)
        text = re.sub(r'[,\s]+$', '', text)

        # Fix "word ,word" -> "word, word"
        text = re.sub(r'(\w)\s*,\s*(\w)', r'\1, \2', text)

        # Normalize whitespace again
        text = " ".join(text.split())

        # Ensure sentence ends with proper punctuation
        if text and text[-1] not in '.!?':
            text += '.'

        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]

        return text


class TemplateMatcher:
    """Matches source content to templates based on structure."""

    def __init__(self):
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def compute_match_score(
        self,
        template: SentenceTemplate,
        proposition: Proposition
    ) -> float:
        """Compute how well a proposition matches a template.

        Args:
            template: Template to match against.
            proposition: Source proposition.

        Returns:
            Match score (0-1).
        """
        score = 0.0
        total_weight = 0.0

        # Check subject match
        if proposition.subject:
            total_weight += 2.0
            if "[SUBJECT]" in template.skeleton:
                score += 2.0

        # Check verb match
        if proposition.predicate:
            total_weight += 2.0
            if "[VERB]" in template.skeleton:
                score += 2.0

        # Check object match
        if proposition.object:
            total_weight += 1.5
            if "[OBJECT]" in template.skeleton:
                score += 1.5

        # Check modifier match
        if proposition.modifiers:
            total_weight += 1.0
            if "[MODIFIER]" in template.skeleton:
                score += 1.0

        # Check prepositional phrase match
        if proposition.prepositional_phrases:
            total_weight += 1.0
            if "[PREPOSITIONAL]" in template.skeleton:
                score += 1.0

        # Check clause match
        if proposition.clauses:
            total_weight += 1.5
            if "[CLAUSE]" in template.skeleton:
                score += 1.5

        # Complexity alignment bonus
        prop_complexity = self._estimate_proposition_complexity(proposition)
        if abs(prop_complexity - template.complexity_score) < 2:
            score += 0.5
            total_weight += 0.5

        return score / max(total_weight, 1.0)

    def _estimate_proposition_complexity(self, proposition: Proposition) -> float:
        """Estimate complexity of a proposition.

        Args:
            proposition: Proposition to analyze.

        Returns:
            Estimated complexity score.
        """
        complexity = 1.0  # Base complexity

        # Add for each component
        if proposition.subject:
            complexity += len(proposition.subject.split()) * 0.2

        if proposition.object:
            complexity += len(proposition.object.split()) * 0.2

        complexity += len(proposition.modifiers) * 0.3
        complexity += len(proposition.prepositional_phrases) * 0.5
        complexity += len(proposition.clauses) * 1.0

        return complexity

    def find_best_template(
        self,
        proposition: Proposition,
        templates: List[SentenceTemplate],
        min_score: float = 0.4
    ) -> Optional[SentenceTemplate]:
        """Find the best matching template for a proposition.

        Args:
            proposition: Source proposition.
            templates: Available templates.
            min_score: Minimum match score required.

        Returns:
            Best matching template or None.
        """
        best_template = None
        best_score = min_score

        for template in templates:
            score = self.compute_match_score(template, proposition)
            if score > best_score:
                best_score = score
                best_template = template

        return best_template
