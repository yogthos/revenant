"""
AI Word Replacer Module

Replaces AI-typical words with contextually appropriate alternatives
derived from the sample text vocabulary. Instead of hardcoded replacements,
this module learns replacement candidates from the target style.

Key features:
- Maintains list of AI-typical words to detect
- Builds vocabulary from sample text at runtime
- Uses semantic similarity to find appropriate replacements
- Falls back to simple alternatives if no good match found
"""

import re
import spacy
from typing import Dict, List, Set, Optional, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass, field


@dataclass
class ReplacementCandidate:
    """A potential replacement word with context."""
    word: str
    pos: str  # Part of speech
    frequency: int
    example_context: str


class AIWordReplacer:
    """
    Replaces AI-typical words with vocabulary from the sample text.

    This ensures replacements match the target style rather than
    using generic hardcoded alternatives.
    """

    # AI-typical words to detect and replace (organized by semantic category)
    AI_WORDS = {
        # Verbs - action words AI overuses
        'verbs': {
            'leverage', 'leveraged', 'leveraging', 'leverages',
            'delve', 'delved', 'delving', 'delves',
            'navigate', 'navigated', 'navigating', 'navigates',
            'harness', 'harnessed', 'harnessing', 'harnesses',
            'foster', 'fostered', 'fostering', 'fosters',
            'empower', 'empowered', 'empowering', 'empowers',
            'streamline', 'streamlined', 'streamlining', 'streamlines',
            'optimize', 'optimized', 'optimizing', 'optimizes',
            'orchestrate', 'orchestrated', 'orchestrating', 'orchestrates',
            'spearhead', 'spearheaded', 'spearheading', 'spearheads',
            'bolster', 'bolstered', 'bolstering', 'bolsters',
            'catalyze', 'catalyzed', 'catalyzing', 'catalyzes',
            'underpin', 'underpinned', 'underpinning', 'underpins',
        },
        # Adjectives - descriptive words AI overuses
        'adjectives': {
            'seamless', 'seamlessly',
            'robust', 'robustly',
            'holistic', 'holistically',
            'pivotal', 'pivotally',
            'crucial', 'crucially',
            'innovative', 'innovatively',
            'transformative',
            'scalable',
            'nuanced',
            'comprehensive', 'comprehensively',
            'vibrant', 'vibrantly',
            'dynamic', 'dynamically',
            'cutting-edge',
            'groundbreaking',
            'game-changing',
        },
        # Nouns - concepts AI overuses
        'nouns': {
            'tapestry',
            'landscape',
            'realm',
            'paradigm',
            'synergy', 'synergies',
            'ecosystem',
            'catalyst',
            'stakeholder', 'stakeholders',
            'framework',
            'trajectory',
            'myriad',
            'plethora',
            'gamut',
            'symphony',
        },
        # Phrases - multi-word patterns
        'phrases': {
            'at the end of the day',
            'in today\'s world',
            'first and foremost',
            'it goes without saying',
            'needless to say',
            'a wide range of',
            'a variety of',
            'in order to',
            'due to the fact',
            'for the purpose of',
            'with regard to',
            'plays a crucial role',
            'is of utmost importance',
            'serves as a',
            'acts as a reminder',
            'stands as a',
            'rich tapestry',
            'key insight',
            'unique perspective',
            'broader context',
            'teeming with',
            'sits within',
            'operates under',
            # Hedging/qualifying phrases
            'such as',
            'for example',
            'for instance',
            'to some extent',
            'to a certain degree',
            'in some ways',
            'in many ways',
            'in a sense',
            'relatively speaking',
            'generally speaking',
            'broadly speaking',
            # Buzzword phrases
            'paradigm shift',
            'game changer',
            'cutting edge',
            'state of the art',
            'best practices',
            'moving forward',
            'going forward',
            'at its core',
            'when it comes to',
            # Weak/hedging patterns
            'we can see',
            'we observe',
            'we find',
            'we note',
            'this suggests',
            'this implies',
            'this indicates',
            'it is important to',
            'it is essential to',
            'it is necessary to',
            'it is clear that',
            'it is evident that',
            'could potentially',
            'might possibly',
            'seems to suggest',
            'appears to be',
            'tends to be',
            'in essence',
            'essentially',
            'basically',
            'importantly',
            'interestingly',
            'notably',
            'it should be noted',
            'it is worth noting',
        },
    }

    # Flatten all AI words for quick lookup
    ALL_AI_WORDS = set()
    for category in AI_WORDS.values():
        ALL_AI_WORDS.update(w.lower() for w in category)

    # Simple fallbacks for common AI words (used if no sample match found)
    FALLBACK_REPLACEMENTS = {
        'leverage': 'use',
        'leveraged': 'used',
        'leveraging': 'using',
        'delve': 'examine',
        'delving': 'examining',
        'seamless': 'smooth',
        'robust': 'strong',
        'holistic': 'complete',
        'pivotal': 'central',
        'crucial': 'important',
        'innovative': 'new',
        'transformative': 'significant',
        'scalable': 'extensible',
        'nuanced': 'subtle',
        'comprehensive': 'thorough',
        'tapestry': 'structure',
        'landscape': 'field',
        'realm': 'domain',
        'paradigm': 'model',
        'synergy': 'cooperation',
        'ecosystem': 'system',
        'catalyst': 'cause',
        'stakeholder': 'participant',
        'myriad': 'many',
        'plethora': 'abundance',
        'navigate': 'traverse',
        'harness': 'use',
        'foster': 'encourage',
        'empower': 'enable',
        'streamline': 'simplify',
        'optimize': 'improve',
        'orchestrate': 'coordinate',
        'vibrant': 'active',
        'dynamic': 'changing',
    }

    def __init__(self, sample_text: str):
        """
        Initialize replacer with sample text vocabulary.

        Args:
            sample_text: The target style sample to derive replacements from
        """
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

        # Build vocabulary from sample
        self.sample_vocabulary: Dict[str, List[ReplacementCandidate]] = defaultdict(list)
        self._analyze_sample(sample_text)

        # Precompile regex patterns for AI phrases
        self._phrase_patterns = self._compile_phrase_patterns()

    def _analyze_sample(self, text: str):
        """Extract vocabulary from sample text organized by POS."""
        doc = self.nlp(text)

        # Count words by POS
        pos_words: Dict[str, Counter] = defaultdict(Counter)
        word_contexts: Dict[str, str] = {}

        for sent in doc.sents:
            sent_text = sent.text.strip()
            for token in sent:
                if token.is_alpha and not token.is_stop and len(token.text) > 2:
                    word = token.lemma_.lower()
                    pos = token.pos_
                    pos_words[pos][word] += 1
                    # Store example context (first occurrence)
                    if word not in word_contexts:
                        word_contexts[word] = sent_text[:100]

        # Build vocabulary by POS category
        for pos, words in pos_words.items():
            for word, freq in words.most_common(100):  # Top 100 per POS
                # Skip if it's an AI word
                if word.lower() in self.ALL_AI_WORDS:
                    continue

                candidate = ReplacementCandidate(
                    word=word,
                    pos=pos,
                    frequency=freq,
                    example_context=word_contexts.get(word, "")
                )
                self.sample_vocabulary[pos].append(candidate)

        print(f"  [AIWordReplacer] Built vocabulary: {sum(len(v) for v in self.sample_vocabulary.values())} words from sample")

    def _compile_phrase_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """Compile regex patterns for AI phrases."""
        patterns = []
        for phrase in self.AI_WORDS['phrases']:
            # Escape special chars and create pattern
            pattern = re.compile(r'\b' + re.escape(phrase) + r'\b', re.IGNORECASE)
            patterns.append((pattern, phrase))
        return patterns

    def _get_pos_for_ai_word(self, word: str) -> str:
        """Determine the likely POS for an AI word."""
        word_lower = word.lower()

        if word_lower in self.AI_WORDS['verbs']:
            return 'VERB'
        elif word_lower in self.AI_WORDS['adjectives']:
            return 'ADJ'
        elif word_lower in self.AI_WORDS['nouns']:
            return 'NOUN'

        # Use spaCy for unknown words
        doc = self.nlp(word)
        if doc:
            return doc[0].pos_
        return 'NOUN'  # Default

    def _find_replacement(self, word: str, context: str = "") -> str:
        """
        Find a replacement word from sample vocabulary.

        Strategy:
        1. Check semantic hints for preferred replacements
        2. Look for those in sample vocabulary
        3. Fall back to sample vocabulary by POS
        4. Final fallback to hardcoded alternatives

        Args:
            word: The AI word to replace
            context: Surrounding text for context-aware selection

        Returns:
            Replacement word from sample vocabulary or fallback
        """
        word_lower = word.lower()
        base_word = re.sub(r'(ed|ing|s|ly)$', '', word_lower)

        # Semantic hints map AI words to preferred alternatives
        # These are ordered by preference - first match in sample wins
        semantic_hints = {
            'leverage': ['use', 'employ', 'apply', 'utilize', 'exploit'],
            'leveraged': ['used', 'employed', 'applied', 'utilized'],
            'leveraging': ['using', 'employing', 'applying', 'utilizing'],
            'delve': ['examine', 'analyze', 'study', 'investigate', 'explore'],
            'delving': ['examining', 'analyzing', 'studying', 'investigating'],
            'robust': ['strong', 'solid', 'firm', 'stable', 'powerful'],
            'holistic': ['complete', 'total', 'whole', 'entire', 'comprehensive'],
            'pivotal': ['central', 'key', 'main', 'primary', 'essential'],
            'crucial': ['important', 'essential', 'necessary', 'vital', 'critical'],
            'crucially': ['importantly', 'essentially', 'necessarily'],
            'innovative': ['new', 'novel', 'original', 'fresh', 'advanced'],
            'navigate': ['move', 'pass', 'traverse', 'cross', 'proceed'],
            'navigating': ['moving', 'passing', 'traversing', 'crossing'],
            'navigated': ['moved', 'passed', 'traversed', 'crossed'],
            'foster': ['encourage', 'promote', 'support', 'advance', 'develop'],
            'fostered': ['encouraged', 'promoted', 'supported', 'developed'],
            'fostering': ['encouraging', 'promoting', 'supporting', 'developing'],
            'fosters': ['encourages', 'promotes', 'supports', 'develops'],
            'empower': ['enable', 'allow', 'permit', 'authorize', 'give'],
            'empowered': ['enabled', 'allowed', 'permitted'],
            'empowering': ['enabling', 'allowing', 'permitting'],
            'streamline': ['simplify', 'improve', 'reduce', 'ease'],
            'optimize': ['improve', 'enhance', 'better', 'refine', 'advance'],
            'orchestrate': ['coordinate', 'organize', 'arrange', 'manage', 'direct'],
            'orchestrated': ['coordinated', 'organized', 'arranged', 'managed'],
            'tapestry': ['structure', 'fabric', 'system', 'pattern', 'whole'],
            'landscape': ['field', 'area', 'domain', 'sphere', 'terrain'],
            'realm': ['domain', 'sphere', 'area', 'field', 'world'],
            'paradigm': ['model', 'pattern', 'system', 'framework', 'mode'],
            'synergy': ['cooperation', 'collaboration', 'unity', 'combination'],
            'ecosystem': ['system', 'environment', 'network', 'structure', 'world'],
            'catalyst': ['cause', 'agent', 'stimulus', 'force', 'factor'],
            'stakeholder': ['participant', 'party', 'member', 'actor', 'agent'],
            'stakeholders': ['participants', 'parties', 'members', 'actors'],
            'myriad': ['many', 'numerous', 'various', 'countless', 'multiple'],
            'plethora': ['abundance', 'wealth', 'mass', 'quantity', 'number'],
            'seamless': ['smooth', 'continuous', 'unbroken', 'fluid', 'easy'],
            'seamlessly': ['smoothly', 'continuously', 'easily', 'naturally'],
            'transformative': ['significant', 'major', 'important', 'profound', 'great'],
            'scalable': ['extensible', 'expandable', 'flexible', 'adaptable'],
            'nuanced': ['subtle', 'delicate', 'fine', 'complex', 'detailed'],
            'comprehensive': ['thorough', 'complete', 'full', 'total', 'extensive'],
            'vibrant': ['active', 'lively', 'dynamic', 'energetic', 'strong'],
            'dynamic': ['changing', 'active', 'moving', 'variable', 'fluid'],
            'harness': ['use', 'employ', 'utilize', 'apply', 'exploit'],
            'harnessing': ['using', 'employing', 'utilizing', 'applying'],
            'harnessed': ['used', 'employed', 'utilized', 'applied'],
            'bolster': ['support', 'strengthen', 'reinforce', 'boost', 'help'],
            'bolstered': ['supported', 'strengthened', 'reinforced'],
            'underpin': ['support', 'sustain', 'maintain', 'uphold'],
            'underpinned': ['supported', 'sustained', 'maintained'],
            'underpinning': ['supporting', 'sustaining', 'maintaining'],
            'spearhead': ['lead', 'head', 'direct', 'guide', 'front'],
        }

        # Get hints for this word
        hints = semantic_hints.get(word_lower, semantic_hints.get(base_word, []))

        # Get expected POS
        expected_pos = self._get_pos_for_ai_word(word_lower)

        # Look for hint words in sample vocabulary
        candidates = self.sample_vocabulary.get(expected_pos, [])
        candidate_words = {c.word: c for c in candidates}

        # Try each hint in order of preference - look for exact match in sample
        for hint in hints:
            if hint in candidate_words:
                return self._match_case(hint, word)

        # Try lemmatized forms
        for hint in hints:
            hint_lemma = hint.rstrip('edingsly')
            for cand_word in candidate_words:
                if cand_word.startswith(hint_lemma) and len(cand_word) <= len(hint_lemma) + 3:
                    return self._match_case(cand_word, word)

        # If no hints found in sample, use first hint (which is the preferred replacement)
        # This ensures we use sensible words even if sample lacks them
        if hints:
            return self._match_case(hints[0], word)

        # Final fallback to hardcoded replacements
        fallback = self.FALLBACK_REPLACEMENTS.get(
            word_lower,
            self.FALLBACK_REPLACEMENTS.get(base_word, word_lower)
        )
        return self._match_case(fallback, word)

    def _match_case(self, replacement: str, original: str) -> str:
        """Match the case of replacement to original word."""
        if original.isupper():
            return replacement.upper()
        elif original[0].isupper():
            return replacement.capitalize()
        return replacement.lower()

    def replace_ai_words(self, text: str) -> str:
        """
        Replace all AI-typical words in text with sample-derived alternatives.

        Args:
            text: Text to process

        Returns:
            Text with AI words replaced
        """
        result = text

        # First, replace phrases (longer patterns first)
        for pattern, phrase in self._phrase_patterns:
            if pattern.search(result):
                # Find contextual replacement for phrase
                replacement = self._get_phrase_replacement(phrase)
                result = pattern.sub(replacement, result)

        # Then replace individual words
        # Use word boundary regex for each AI word
        for word in self.ALL_AI_WORDS:
            if len(word) > 2:  # Skip very short words
                pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)

                def replace_match(match):
                    original = match.group(0)
                    # Get surrounding context
                    start = max(0, match.start() - 50)
                    end = min(len(result), match.end() + 50)
                    context = result[start:end]
                    return self._find_replacement(original, context)

                result = pattern.sub(replace_match, result)

        return result

    def _get_phrase_replacement(self, phrase: str) -> str:
        """Get replacement for an AI phrase."""
        phrase_replacements = {
            'at the end of the day': 'ultimately',
            'in today\'s world': 'under present conditions',
            'first and foremost': 'primarily',
            'it goes without saying': 'clearly',
            'needless to say': 'evidently',
            'a wide range of': 'various',
            'a variety of': 'various',
            'in order to': 'to',
            'due to the fact': 'because',
            'for the purpose of': 'for',
            'with regard to': 'concerning',
            'plays a crucial role': 'is essential',
            'is of utmost importance': 'is essential',
            'serves as a': 'constitutes a',
            'acts as a reminder': 'demonstrates',
            'stands as a': 'represents a',
            'rich tapestry': 'complex structure',
            'key insight': 'central point',
            'unique perspective': 'particular viewpoint',
            'broader context': 'wider circumstances',
            'teeming with': 'containing',
            'sits within': 'exists within',
            'operates under': 'functions according to',
            # Hedging phrases - remove or simplify
            'such as': 'like',
            'for example': '—',  # Em-dash works as a transition
            'for instance': '—',
            'to some extent': 'partially',
            'to a certain degree': 'partially',
            'in some ways': 'partially',
            'in many ways': 'in several respects',
            'in a sense': '',
            'relatively speaking': '',
            'generally speaking': 'generally',
            'broadly speaking': 'broadly',
            # Buzzword phrases
            'paradigm shift': 'transformation',
            'game changer': 'decisive factor',
            'cutting edge': 'advanced',
            'state of the art': 'modern',
            'best practices': 'methods',
            'moving forward': 'henceforth',
            'going forward': 'henceforth',
            'at its core': 'fundamentally',
            'when it comes to': 'regarding',
            # Weak/hedging patterns - make more assertive
            'we can see': 'it is apparent',
            'we observe': 'one observes',
            'we find': 'one finds',
            'we note': 'one notes',
            'this suggests': 'this demonstrates',
            'this implies': 'this shows',
            'this indicates': 'this demonstrates',
            'it is important to': 'it is necessary to',
            'it is essential to': 'it is necessary to',
            'it is necessary to': 'one must',
            'it is clear that': 'clearly',
            'it is evident that': 'evidently',
            'could potentially': 'may',
            'might possibly': 'may',
            'seems to suggest': 'indicates',
            'appears to be': 'is',
            'tends to be': 'is generally',
            'in essence': '',
            'essentially': '',
            'basically': '',
            'importantly': '',
            'interestingly': '',
            'notably': '',
            'it should be noted': '',
            'it is worth noting': '',
        }
        return phrase_replacements.get(phrase.lower(), '')

    def get_detected_ai_words(self, text: str) -> List[str]:
        """
        Get list of AI-typical words found in text.

        Args:
            text: Text to scan

        Returns:
            List of detected AI words
        """
        found = []
        text_lower = text.lower()

        # Check phrases first
        for pattern, phrase in self._phrase_patterns:
            if pattern.search(text_lower):
                found.append(phrase)

        # Check individual words
        for word in self.ALL_AI_WORDS:
            if len(word) > 2:
                pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                if pattern.search(text_lower):
                    found.append(word)

        return found


# Test function
if __name__ == '__main__':
    from pathlib import Path

    sample_path = Path(__file__).parent / "prompts" / "sample.txt"
    if sample_path.exists():
        with open(sample_path, 'r', encoding='utf-8') as f:
            sample_text = f.read()

        print("=== AI Word Replacer Test ===\n")

        replacer = AIWordReplacer(sample_text)

        # Test text with AI words
        test_text = """
        We need to leverage the synergy between our teams to delve into this
        transformative paradigm shift. The robust ecosystem fosters innovative
        solutions that navigate the complex landscape. This holistic approach
        serves as a catalyst for change in today's world.
        """

        print("Original text:")
        print(test_text)

        print("\nDetected AI words:")
        detected = replacer.get_detected_ai_words(test_text)
        for word in detected:
            print(f"  - {word}")

        print("\nReplaced text:")
        replaced = replacer.replace_ai_words(test_text)
        print(replaced)

        print("\nRemaining AI words:")
        remaining = replacer.get_detected_ai_words(replaced)
        if remaining:
            for word in remaining:
                print(f"  - {word}")
        else:
            print("  (none)")
    else:
        print("No sample.txt found.")

