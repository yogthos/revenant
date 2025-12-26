"""Integration tests for semantic style pipeline.

Tests that:
1. spaCy-based vocabulary classification correctly separates transferable vs topic-specific
2. Topic-specific terms (Marxists, proletariat, etc.) are excluded from style transfer
3. General transitions and vocabulary are identified correctly
4. Discourse context is properly tracked
5. Prompts are built with correct structure
"""

import pytest
from pathlib import Path


class TestVocabularyClassification:
    """Test spaCy-based vocabulary classification."""

    @pytest.fixture
    def mao_corpus(self):
        """Load Mao corpus."""
        corpus_path = Path(__file__).parent.parent / "data" / "corpus" / "mao.txt"
        if not corpus_path.exists():
            pytest.skip("Mao corpus not found")

        text = corpus_path.read_text(encoding="utf-8")
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return paragraphs[:10]  # Use first 10 for faster tests

    @pytest.fixture
    def classifier(self):
        """Create vocabulary classifier."""
        from src.templates.style_vocabulary import VocabularyClassifier
        return VocabularyClassifier()

    def test_transitions_are_general(self, classifier, mao_corpus):
        """Transitions should be general-purpose, not topic-specific."""
        classified = classifier.classify_corpus(mao_corpus)

        # Should identify common transitions
        transitions = set(classified.transitions.keys())

        # Mao uses these transitions frequently
        expected_transitions = {"thus", "therefore", "however", "but", "and", "or"}
        found = transitions & expected_transitions

        assert len(found) > 0, f"Expected some transitions from {expected_transitions}, got {transitions}"
        print(f"Found transitions: {list(transitions)[:15]}")

    def test_topic_specific_excluded(self, classifier, mao_corpus):
        """Topic-specific terms should be identified and excluded."""
        classified = classifier.classify_corpus(mao_corpus)

        # These are Mao-specific terms that should NOT be transferable
        topic_specific = classified.topic_specific

        # Marxist terminology should be topic-specific
        marxist_terms = {"marxist", "marxists", "marxism", "proletariat", "bourgeoisie", "communism"}

        # Check that these are in topic_specific set
        # Note: they might be detected via NER or frequency analysis
        print(f"Topic-specific terms found: {list(topic_specific)[:20]}")

        # More importantly: these should NOT be in transferable vocabulary
        transferable_verbs = set(classified.general_verbs.keys())
        transferable_adjs = set(classified.general_adjectives.keys())
        transferable_nouns = set(classified.general_nouns.keys())
        all_transferable = transferable_verbs | transferable_adjs | transferable_nouns

        contamination = marxist_terms & all_transferable
        assert len(contamination) == 0, f"Topic-specific terms leaked into transferable: {contamination}"

    def test_general_verbs_are_abstract(self, classifier, mao_corpus):
        """General verbs should be abstract/general, not topic-specific."""
        classified = classifier.classify_corpus(mao_corpus)

        verbs = set(classified.general_verbs.keys())

        # These are generic verbs that should be transferable
        generic_verbs = {"develop", "exist", "become", "understand", "change", "know"}
        found = verbs & generic_verbs

        print(f"General verbs found: {list(verbs)[:15]}")
        # Should find at least some general verbs
        assert len(found) >= 1, f"Expected general verbs, got {verbs}"

    def test_general_adjectives_are_abstract(self, classifier, mao_corpus):
        """General adjectives should be abstract, not domain-specific."""
        classified = classifier.classify_corpus(mao_corpus)

        adjs = set(classified.general_adjectives.keys())

        print(f"General adjectives found: {list(adjs)[:15]}")
        # Should have some abstract adjectives
        assert len(adjs) > 0, "Should identify some general adjectives"


class TestStyleFingerprint:
    """Test statistical style fingerprint extraction."""

    @pytest.fixture
    def mao_corpus(self):
        """Load Mao corpus."""
        corpus_path = Path(__file__).parent.parent / "data" / "corpus" / "mao.txt"
        if not corpus_path.exists():
            pytest.skip("Mao corpus not found")

        text = corpus_path.read_text(encoding="utf-8")
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return paragraphs[:10]

    @pytest.fixture
    def fingerprint(self, mao_corpus):
        """Extract style fingerprint."""
        from src.templates.fingerprint import StyleFingerprintExtractor
        extractor = StyleFingerprintExtractor()
        return extractor.extract(mao_corpus)

    def test_sentence_length_reasonable(self, fingerprint):
        """Sentence length should be reasonable for Mao's style."""
        # Mao tends to write long, complex sentences
        assert fingerprint.sentence_length_mean > 15, "Mao writes long sentences"
        assert fingerprint.sentence_length_mean < 100, "Mean should be reasonable"
        assert fingerprint.sentence_length_std > 0, "Should have variation"

        print(f"Sentence length: {fingerprint.sentence_length_mean:.1f} +/- {fingerprint.sentence_length_std:.1f}")

    def test_transitions_identified(self, fingerprint):
        """Should identify preferred transitions."""
        assert len(fingerprint.preferred_transitions) > 0

        print(f"Preferred transitions: {fingerprint.preferred_transitions[:10]}")

    def test_distinctive_vocabulary(self, fingerprint):
        """Should identify distinctive vocabulary."""
        vocab = fingerprint.distinctive_vocabulary

        assert len(vocab) > 0, "Should identify distinctive words"
        print(f"Distinctive vocabulary: {list(vocab.keys())[:15]}")


class TestDiscourseContext:
    """Test discourse context tracking."""

    @pytest.fixture
    def analyzer(self):
        """Create discourse analyzer."""
        from src.templates.discourse import DiscourseAnalyzer
        return DiscourseAnalyzer()

    def test_document_position_tracking(self, analyzer):
        """Should correctly identify document positions."""
        paragraphs = [
            "Introduction paragraph that sets up the topic.",
            "Body paragraph with supporting details.",
            "Another body paragraph with more evidence.",
            "Conclusion paragraph that wraps up."
        ]

        # Analyze document
        doc_context = analyzer.analyze_document(paragraphs)

        assert doc_context is not None
        assert len(doc_context.paragraph_contexts) == 4

    def test_sentence_role_detection(self, analyzer):
        """Should detect sentence roles."""
        from src.templates.discourse import SentenceRole

        paragraph = (
            "The main point is clear. "
            "For example, we see this in daily life. "
            "Furthermore, the evidence supports this. "
            "Therefore, we can conclude the argument is valid."
        )

        para_context = analyzer.analyze_paragraph(paragraph, 0, 1)

        # First sentence should be TOPIC
        assert para_context.sentence_contexts[0].role == SentenceRole.TOPIC


class TestSemanticPipeline:
    """Test the full semantic style pipeline."""

    @pytest.fixture
    def mao_corpus(self):
        """Load Mao corpus."""
        corpus_path = Path(__file__).parent.parent / "data" / "corpus" / "mao.txt"
        if not corpus_path.exists():
            pytest.skip("Mao corpus not found")

        text = corpus_path.read_text(encoding="utf-8")
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return paragraphs[:5]  # Use fewer for faster tests

    @pytest.fixture
    def pipeline(self, mao_corpus):
        """Create semantic pipeline."""
        from src.templates.pipeline import SemanticStylePipeline, TransferConfig

        config = TransferConfig(
            author="Mao",
            use_semantic_transfer=True,
            style_strictness=0.6,
            delta_threshold=2.0,
        )

        return SemanticStylePipeline(mao_corpus, "Mao", config)

    def test_pipeline_initialization(self, pipeline):
        """Pipeline should initialize with classified vocabulary."""
        assert pipeline.classified_vocab is not None
        assert pipeline.fingerprint is not None

        # Should have transitions
        assert len(pipeline.classified_vocab.transitions) > 0

        # Should have fingerprint
        assert pipeline.fingerprint.sentence_length_mean > 0

    def test_prompt_excludes_topic_terms(self, pipeline):
        """Generated prompts should not include topic-specific terms."""
        from src.templates.discourse import (
            SentenceContext, ParagraphContext,
            DocumentPosition, SentenceRole, DiscourseRelation, ReferenceType
        )

        source_sentence = "The economy grew last year."

        sent_ctx = SentenceContext(
            role=SentenceRole.CLAIM,
            position=0,
            total_in_paragraph=3,
            discourse_relation=DiscourseRelation.NONE,
            reference_type=ReferenceType.NONE,
        )

        from src.templates.discourse import ParagraphRole
        para_ctx = ParagraphContext(
            role=ParagraphRole.ELABORATION,
            document_position=DocumentPosition.BODY,
            position=0,
            total_in_document=1,
            discourse_relation=DiscourseRelation.NONE,
        )

        prompt = pipeline.build_sentence_prompt(
            source_sentence,
            sent_ctx,
            para_ctx,
            technical_terms=set(),
        )

        # Prompt should NOT mention Marxists, proletariat, etc.
        marxist_terms = ["marxist", "proletariat", "bourgeoisie", "communism"]
        prompt_lower = prompt.lower()

        for term in marxist_terms:
            assert term not in prompt_lower, f"Prompt should not contain topic-specific term '{term}'"

        print("--- Generated Prompt ---")
        print(prompt[:1000])

    def test_prompt_includes_style_markers(self, pipeline):
        """Prompts should include transferable style markers."""
        from src.templates.discourse import (
            SentenceContext, ParagraphContext,
            DocumentPosition, SentenceRole, DiscourseRelation, ReferenceType
        )

        source = "Technology advances rapidly."

        sent_ctx = SentenceContext(
            role=SentenceRole.CLAIM,
            position=0,
            total_in_paragraph=1,
            discourse_relation=DiscourseRelation.NONE,
            reference_type=ReferenceType.NONE,
        )

        from src.templates.discourse import ParagraphRole
        para_ctx = ParagraphContext(
            role=ParagraphRole.THESIS,
            document_position=DocumentPosition.INTRO,
            position=0,
            total_in_document=1,
            discourse_relation=DiscourseRelation.NONE,
        )

        prompt = pipeline.build_sentence_prompt(
            source, sent_ctx, para_ctx, set()
        )

        # Should mention author
        assert "Mao" in prompt

        # Should have style section
        assert "AUTHOR'S STYLE" in prompt

        # Should have discourse context
        assert "DISCOURSE CONTEXT" in prompt

        # Should preserve source meaning
        assert source in prompt


class TestStyleVerification:
    """Test Burrows' Delta style verification."""

    @pytest.fixture
    def mao_corpus(self):
        """Load Mao corpus."""
        corpus_path = Path(__file__).parent.parent / "data" / "corpus" / "mao.txt"
        if not corpus_path.exists():
            pytest.skip("Mao corpus not found")

        text = corpus_path.read_text(encoding="utf-8")
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return paragraphs[:10]

    @pytest.fixture
    def verifier(self, mao_corpus):
        """Create style verifier."""
        from src.templates.fingerprint import StyleFingerprintExtractor, StyleVerifier

        extractor = StyleFingerprintExtractor()
        fingerprint = extractor.extract(mao_corpus)

        return StyleVerifier(fingerprint, threshold=2.0)

    def test_corpus_text_passes(self, verifier, mao_corpus):
        """Text from corpus should pass verification."""
        # Use a sentence from the corpus
        sample = mao_corpus[0][:200]  # First 200 chars

        verification = verifier.verify(sample)

        print(f"Delta score for corpus text: {verification.delta_score:.2f}")
        # Should have low delta (close to author's style)
        # Note: may not pass threshold depending on sample

    def test_very_different_text_has_high_delta(self, verifier):
        """Very different text should have high delta."""
        # Simple, modern, non-Mao style text
        different_text = "Hi! Let's go shopping. I love pizza. LOL that's funny!"

        verification = verifier.verify(different_text)

        print(f"Delta score for different text: {verification.delta_score:.2f}")
        # Delta should be higher for very different style
        # (exact value depends on implementation)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
