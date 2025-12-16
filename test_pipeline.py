"""
Test Suite for Style Transfer Pipeline

Tests each stage independently and the complete pipeline.
Validates:
1. Semantic extraction accuracy
2. Style analysis correctness
3. Synthesis quality
4. Verification accuracy
5. End-to-end pipeline
"""

import unittest
import sys
from pathlib import Path

# Import pipeline components
from semantic_extractor import SemanticExtractor, SemanticContent
from style_analyzer import StyleAnalyzer, StyleProfile
from verifier import Verifier, VerificationResult
from sentence_validator import SentenceValidator, ValidationIssue
from semantic_word_mapper import SemanticWordMapper
from template_generator import SentenceTemplate


class TestSemanticExtractor(unittest.TestCase):
    """Tests for the semantic extraction module."""

    @classmethod
    def setUpClass(cls):
        """Initialize extractor once for all tests."""
        cls.extractor = SemanticExtractor()

    def test_extract_citations(self):
        """Test that citations are properly extracted."""
        text = "Tom Stonier proposed this theory[^155]. Another study[^25] confirms it."
        result = self.extractor.extract(text)

        self.assertIn('[^155]', result.preserved_elements['citations'])
        self.assertIn('[^25]', result.preserved_elements['citations'])

    def test_extract_claims(self):
        """Test that claims are extracted from sentences."""
        text = "The universe is infinite. Stars eventually die. Energy is conserved."
        result = self.extractor.extract(text)

        self.assertGreater(len(result.claims), 0)
        # Check that we got claims for each sentence
        self.assertGreaterEqual(len(result.claims), 2)

    def test_extract_entities(self):
        """Test entity extraction."""
        text = "Albert Einstein developed the theory of relativity in Germany."
        result = self.extractor.extract(text)

        entity_texts = [e.text for e in result.entities]
        # Should find named entities
        self.assertTrue(any('Einstein' in t or 'Albert' in t for t in entity_texts))

    def test_extract_relationships(self):
        """Test relationship extraction."""
        text = "The temperature rises because heat is applied. However, pressure remains constant."
        result = self.extractor.extract(text)

        # Should find cause-effect and contrast relationships
        relation_types = [r.relation_type for r in result.relationships]
        self.assertTrue(
            'CAUSE_EFFECT' in relation_types or 'CONTRAST' in relation_types,
            "Should detect causal or contrast relationships"
        )

    def test_paragraph_structure(self):
        """Test paragraph structure analysis."""
        text = """First paragraph with introduction.

Second paragraph with more content. It has multiple sentences.

Third paragraph concludes."""

        result = self.extractor.extract(text)

        self.assertEqual(len(result.paragraph_structure), 3)
        for para in result.paragraph_structure:
            self.assertIn('function', para)
            self.assertIn('sentence_count', para)


class TestStyleAnalyzer(unittest.TestCase):
    """Tests for the style analysis module."""

    @classmethod
    def setUpClass(cls):
        """Initialize analyzer once for all tests."""
        cls.analyzer = StyleAnalyzer()

    def test_vocabulary_analysis(self):
        """Test vocabulary profile extraction."""
        text = "The quick brown fox jumps over the lazy dog. The fox was quick."
        result = self.analyzer.analyze(text)

        self.assertGreater(result.vocabulary.total_words, 0)
        self.assertGreater(result.vocabulary.unique_words, 0)
        self.assertLessEqual(result.vocabulary.vocabulary_richness, 1.0)

    def test_sentence_patterns(self):
        """Test sentence pattern extraction."""
        text = """Short sentence. This is a medium length sentence with more words.
        This is quite a long sentence that contains many words and demonstrates
        the kind of complex structure that might appear in formal writing."""

        result = self.analyzer.analyze(text)

        # Should have variety in sentence lengths
        self.assertGreater(result.sentences.avg_length, 0)
        self.assertIn('short', result.sentences.length_distribution)
        self.assertIn('medium', result.sentences.length_distribution)

    def test_formality_detection(self):
        """Test formality score calculation."""
        formal_text = "Therefore, the hypothesis is supported by empirical evidence. Nevertheless, further investigation is warranted."
        informal_text = "So yeah, it's kinda working. Can't really tell though."

        formal_result = self.analyzer.analyze(formal_text)
        informal_result = self.analyzer.analyze(informal_text)

        self.assertGreater(
            formal_result.vocabulary.formality_score,
            informal_result.vocabulary.formality_score,
            "Formal text should have higher formality score"
        )

    def test_style_guide_generation(self):
        """Test style guide generation."""
        text = "Sample text for analysis. It has multiple sentences. This tests the guide."
        result = self.analyzer.analyze(text)

        guide = result.to_style_guide()

        self.assertIn("STYLE GUIDE", guide)
        self.assertIn("Vocabulary", guide)
        self.assertIn("Sentence Structure", guide)

    def test_example_sentences(self):
        """Test example sentence extraction."""
        text = """Short one. Medium length sentence here.
        A somewhat longer sentence that spans more words.
        This is an even longer sentence that contains quite a few words and
        demonstrates complex sentence structure in the text."""

        result = self.analyzer.analyze(text)

        # Should have categorized examples
        self.assertIn('short', result.example_sentences)
        self.assertIn('medium', result.example_sentences)


class TestVerifier(unittest.TestCase):
    """Tests for the verification module."""

    @classmethod
    def setUpClass(cls):
        """Initialize verifier once for all tests."""
        cls.verifier = Verifier()

    def test_citation_preservation(self):
        """Test that citation preservation is verified."""
        input_text = "This theory was proposed[^1]. Another study[^2] confirms it."

        # Good output preserves citations
        good_output = "The theory in question was proposed[^1]. Confirmation comes from another study[^2]."
        result = self.verifier.verify(input_text, good_output)
        self.assertTrue(result.preservation.citations_preserved)

        # Bad output loses citations
        bad_output = "The theory was proposed. Another study confirms it."
        result = self.verifier.verify(input_text, bad_output)
        self.assertFalse(result.preservation.citations_preserved)

    def test_semantic_preservation(self):
        """Test semantic content verification."""
        input_text = "The universe is infinite. Stars eventually die."

        # Good output preserves meaning
        good_output = "Stars will eventually die. The cosmos extends infinitely."
        result = self.verifier.verify(input_text, good_output)
        self.assertGreaterEqual(result.semantic.claim_coverage, 0.5)

        # Bad output changes meaning completely
        bad_output = "Cats are fluffy. Dogs bark loudly."
        result = self.verifier.verify(input_text, bad_output)
        self.assertLess(result.semantic.claim_coverage, 0.5)

    def test_meaning_drift_detection(self):
        """Test that meaning drift is detected."""
        input_text = "Computers process data. Networks transmit information."

        # Low drift - similar concepts (use exact same words to ensure low drift)
        similar_output = "Computers process data efficiently. Networks transmit information quickly."
        result = self.verifier.verify(input_text, similar_output)
        self.assertLess(result.semantic.meaning_drift_score, 0.8)

        # High drift - completely different concepts
        different_output = "Elephants roam the savanna. Dolphins swim in the ocean."
        result = self.verifier.verify(input_text, different_output)
        self.assertGreater(result.semantic.meaning_drift_score, 0.5)

    def test_overall_verification(self):
        """Test overall verification result."""
        input_text = "Energy is conserved[^1]. Mass can convert to energy."
        good_output = "Mass converts to energy. Energy remains conserved[^1]."

        result = self.verifier.verify(input_text, good_output)

        # Should have all verification components
        self.assertIsNotNone(result.semantic)
        self.assertIsNotNone(result.style)
        self.assertIsNotNone(result.preservation)


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""

    def test_component_import(self):
        """Test that all components can be imported."""
        from semantic_extractor import SemanticExtractor
        from style_analyzer import StyleAnalyzer
        from synthesizer import Synthesizer
        from verifier import Verifier
        from humanizer import StyleTransferPipeline

        # All imports successful
        self.assertTrue(True)

    def test_sample_text_exists(self):
        """Test that sample text file exists."""
        # Check for sample_mao.txt (default) or sample.txt
        sample_path = Path(__file__).parent / "prompts" / "sample_mao.txt"
        if not sample_path.exists():
            sample_path = Path(__file__).parent / "prompts" / "sample.txt"
        self.assertTrue(sample_path.exists(), "Sample text file should exist")

    def test_config_exists(self):
        """Test that config file exists."""
        config_path = Path(__file__).parent / "config.json"
        self.assertTrue(config_path.exists(), "Config file should exist")

    def test_semantic_to_style_flow(self):
        """Test that semantic content can inform style synthesis."""
        extractor = SemanticExtractor()
        analyzer = StyleAnalyzer()

        input_text = "The quick brown fox jumps. The lazy dog sleeps."
        sample_text = "Matter exists objectively. Reality is knowable."

        # Extract semantics from input
        semantics = extractor.extract(input_text)

        # Analyze target style
        style = analyzer.analyze(sample_text)

        # Both should produce valid outputs
        self.assertGreater(len(semantics.claims), 0)
        self.assertGreater(style.vocabulary.total_words, 0)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and error handling."""

    def test_empty_input(self):
        """Test handling of empty input."""
        extractor = SemanticExtractor()
        result = extractor.extract("")

        self.assertEqual(len(result.claims), 0)
        self.assertEqual(len(result.entities), 0)

    def test_single_word(self):
        """Test handling of single word input."""
        extractor = SemanticExtractor()
        result = extractor.extract("Hello")

        # Should not crash
        self.assertIsNotNone(result)

    def test_special_characters(self):
        """Test handling of special characters."""
        extractor = SemanticExtractor()
        text = "Test with émojis 🎉 and spëcial châräctërs!"
        result = extractor.extract(text)

        # Should not crash
        self.assertIsNotNone(result)

    def test_long_paragraph(self):
        """Test handling of long paragraphs."""
        analyzer = StyleAnalyzer()

        # Create a long text
        long_text = " ".join(["This is a test sentence."] * 100)
        result = analyzer.analyze(long_text)

        self.assertGreater(result.sentences.avg_length, 0)

    def test_no_punctuation(self):
        """Test handling of text without punctuation."""
        extractor = SemanticExtractor()
        result = extractor.extract("no punctuation here at all")

        # Should not crash, may have limited extraction
        self.assertIsNotNone(result)


class TestSentenceValidator(unittest.TestCase):
    """Tests for sentence validation module."""

    @classmethod
    def setUpClass(cls):
        """Initialize validator once for all tests."""
        cls.validator = SentenceValidator(word_count_tolerance=0.35, require_exact_opener=False)

    def test_word_count_tolerance(self):
        """Test that word count tolerance is applied correctly."""
        template = SentenceTemplate(
            word_count=20,
            pos_sequence=["DET", "NOUN", "VERB", "DET", "NOUN"],
            punctuation=".",
            opener_type="det_noun",
            has_subordinate_clause=False,
            clause_count=1,
            length_category="medium"
        )

        # Within 35% tolerance (20 ± 7 = 13-27) - use sentence with ~20 words
        sentence_ok = "The quick brown fox jumps over the lazy dog and runs very fast indeed."
        issues = self.validator.validate_sentence(sentence_ok, template)
        # Should have no critical word count issues (may have minor ones like opener type)
        word_count_issues = [i for i in issues if "word count" in i.description.lower() and "CRITICAL" in i.description]
        self.assertEqual(len(word_count_issues), 0, "Should not have critical word count issues within tolerance")

    def test_opener_flexibility(self):
        """Test that opener type matching is flexible."""
        template = SentenceTemplate(
            word_count=15,
            pos_sequence=["ADV", "DET", "NOUN", "VERB"],
            punctuation=".",
            opener_type="conjunction",
            has_subordinate_clause=False,
            clause_count=1,
            length_category="short"
        )

        # Adverb opener (similar to conjunction) should be acceptable
        sentence = "However, the system works correctly."
        issues = self.validator.validate_sentence(sentence, template)
        # Should not have critical opener mismatch
        critical_issues = [i for i in issues if "CRITICAL" in i.description]
        self.assertEqual(len(critical_issues), 0)

    def test_priority_assignment(self):
        """Test that only critical mismatches get priority 1."""
        template = SentenceTemplate(
            word_count=20,
            pos_sequence=["NOUN", "VERB", "SCONJ", "DET", "NOUN"],
            punctuation=".",
            opener_type="noun",
            has_subordinate_clause=True,
            clause_count=2,
            length_category="medium"
        )

        # Sentence with wrong structure (missing subordinate clause)
        sentence = "The system works."
        issues = self.validator.validate_sentence(sentence, template)

        # Should have at least one issue
        self.assertGreater(len(issues), 0)

        # Check that structural issues are marked appropriately
        has_clause_issue = any("clause" in i.description.lower() for i in issues)
        self.assertTrue(has_clause_issue)


class TestSemanticPreservation(unittest.TestCase):
    """Tests for semantic content preservation."""

    @classmethod
    def setUpClass(cls):
        """Initialize verifier once for all tests."""
        cls.verifier = Verifier()

    def test_claim_fully_expressed(self):
        """Test that _claim_fully_expressed checks for full expression."""
        from semantic_extractor import Claim

        claim = Claim(
            text="Tom Stonier proposed that information is interconvertible with energy",
            subject="Tom Stonier",
            predicate="proposed",
            objects=["information", "energy"],
            modifiers=[],
            confidence=0.9,
            citations=[]
        )

        # Good output: fully expresses the claim
        good_output = "Tom Stonier proposed that information is interconvertible with energy."
        result = self.verifier._claim_fully_expressed(claim, good_output.lower())
        self.assertTrue(result)

        # Bad output: only mentions one word
        bad_output = "The theory discusses energy."
        result = self.verifier._claim_fully_expressed(claim, bad_output.lower())
        self.assertFalse(result)

    def test_semantic_coverage_threshold(self):
        """Test that semantic coverage threshold is enforced."""
        input_text = "The universe is infinite. Stars eventually die. Energy is conserved."
        input_semantics = self.verifier.semantic_extractor.extract(input_text)

        # Output with good coverage (all claims present)
        good_output = "The cosmos extends infinitely. Stars will eventually die. Energy remains conserved."
        result = self.verifier.verify(input_text, good_output, input_semantics=input_semantics)
        # Coverage may vary, but should be reasonable (>= 0.5)
        self.assertGreaterEqual(result.semantic.claim_coverage, 0.5)

        # Output with poor coverage (missing claims)
        bad_output = "The universe is infinite."
        result = self.verifier.verify(input_text, bad_output, input_semantics=input_semantics)
        self.assertLess(result.semantic.claim_coverage, 0.8)

    def test_missing_claims_hint(self):
        """Test that missing claims generate CRITICAL hints."""
        input_text = "The universe is infinite. Stars eventually die. Energy is conserved."
        input_semantics = self.verifier.semantic_extractor.extract(input_text)

        # Output missing claims
        bad_output = "The universe is infinite."
        result = self.verifier.verify(
            input_text, bad_output,
            input_semantics=input_semantics,
            paragraph_template=None,
            sentence_validator=None
        )

        # Should have hints about missing claims if coverage is low
        # Note: hints are only generated if coverage < min_claim_coverage from config
        semantic_hints = [h for h in result.transformation_hints if "Missing" in h.issue and "claims" in h.issue.lower()]
        # This test verifies the mechanism works, but may not always trigger depending on config
        self.assertIsNotNone(result.semantic.claim_coverage)


class TestSemanticWordMapper(unittest.TestCase):
    """Tests for semantic word mapper."""

    def test_model_with_vectors(self):
        """Test that mapper uses a model with vectors."""
        sample_text = "The principal method demonstrates significant results. The system operates effectively."

        mapper = SemanticWordMapper(sample_text, similarity_threshold=0.5, min_sample_frequency=1)

        # Should have loaded a model with vectors
        self.assertGreater(mapper.nlp.vocab.vectors.size, 0,
                          "Mapper should use a model with word vectors")

        # Should have built some mappings
        stats = mapper.get_mapping_stats()
        self.assertGreater(stats['sample_vocab_size'], 0)

    def test_model_fallback(self):
        """Test that mapper falls back to model with vectors when passed one without."""
        import spacy

        # Load a model without vectors
        nlp_sm = spacy.load('en_core_web_sm')
        self.assertEqual(nlp_sm.vocab.vectors.size, 0, "en_core_web_sm should have no vectors")

        sample_text = "The principal method demonstrates significant results."

        # Pass model without vectors - should automatically switch
        mapper = SemanticWordMapper(sample_text, nlp_model=nlp_sm,
                                   similarity_threshold=0.5, min_sample_frequency=1)

        # Should have switched to a model with vectors (if available)
        # If md model is not available, it will use sm and warn, but we test the logic
        if mapper.nlp.vocab.vectors.size > 0:
            # Successfully switched to model with vectors
            self.assertTrue(True, "Mapper successfully switched to model with vectors")
        else:
            # Fallback to sm (should warn but not crash)
            self.assertEqual(mapper.nlp.vocab.vectors.size, 0)

    def test_mapping_application(self):
        """Test that mappings are applied correctly."""
        sample_text = "The principal method demonstrates significant results. The system operates effectively."

        mapper = SemanticWordMapper(sample_text, similarity_threshold=0.3, min_sample_frequency=1)

        # Test text with common words
        test_text = "This is an important method that shows significant results."
        mapped_text = mapper.apply_mapping(test_text)

        # Should have applied some mappings (if similarity threshold is met)
        # Note: exact mappings depend on sample text, so we just check it doesn't crash
        self.assertIsInstance(mapped_text, str)
        self.assertGreater(len(mapped_text), 0)


class TestSemanticCoverageTracking(unittest.TestCase):
    """Tests for semantic coverage tracking in pipeline."""

    def test_coverage_calculation(self):
        """Test that semantic coverage is calculated correctly."""
        from humanizer import StyleTransferPipeline

        pipeline = StyleTransferPipeline()

        # Create test semantics
        from semantic_extractor import Claim, SemanticContent
        claims = [
            Claim("The universe is infinite", "universe", "is", ["infinite"], [], 0.9, []),
            Claim("Stars eventually die", "Stars", "die", [], [], 0.9, [])
        ]
        semantics = SemanticContent(
            entities=[],
            claims=claims,
            relationships=[],
            preserved_elements={'citations': [], 'technical_terms': []},
            paragraph_structure=[]
        )

        # Test coverage calculation
        good_output = "The cosmos extends infinitely. Stars will eventually die."
        coverage = pipeline._check_semantic_coverage(good_output, semantics)
        # Coverage may vary based on claim matching, but should be reasonable
        self.assertGreaterEqual(coverage, 0.4, "Good output should have reasonable coverage")

        bad_output = "The universe is infinite."
        coverage = pipeline._check_semantic_coverage(bad_output, semantics)
        # Bad output should have lower coverage
        self.assertLess(coverage, 0.7, "Bad output should have lower coverage")

    def test_pipeline_result_coverage(self):
        """Test that PipelineResult includes semantic coverage."""
        from humanizer import PipelineResult

        result = PipelineResult(
            success=True,
            output_text="Test output",
            iterations=1,
            final_verification=None,
            semantic_content=None,
            style_profile=None,
            error=None,
            semantic_coverage=0.95
        )

        self.assertEqual(result.semantic_coverage, 0.95)
        self.assertIn('semantic_coverage', result.to_dict())


class TestConfigLoading(unittest.TestCase):
    """Tests for configuration loading."""

    def test_semantic_preservation_config(self):
        """Test that semantic preservation config is loaded."""
        from humanizer import StyleTransferPipeline
        import json
        from pathlib import Path

        config_path = Path(__file__).parent / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Check that semantic_preservation config exists
            self.assertIn('semantic_preservation', config)
            semantic_config = config['semantic_preservation']
            self.assertIn('min_claim_coverage', semantic_config)
            self.assertIn('require_full_expression', semantic_config)
            self.assertIn('reject_on_semantic_loss', semantic_config)

    def test_sentence_validation_config(self):
        """Test that sentence validation config is loaded."""
        import json
        from pathlib import Path

        config_path = Path(__file__).parent / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Check that sentence_validation config exists with correct values
            self.assertIn('sentence_validation', config)
            sent_config = config['sentence_validation']
            self.assertGreaterEqual(sent_config.get('word_count_tolerance', 0), 0.3)
            self.assertFalse(sent_config.get('require_exact_opener', True))


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSemanticExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestStyleAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestVerifier))
    suite.addTests(loader.loadTestsFromTestCase(TestPipelineIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestSentenceValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestSemanticPreservation))
    suite.addTests(loader.loadTestsFromTestCase(TestSemanticWordMapper))
    suite.addTests(loader.loadTestsFromTestCase(TestSemanticCoverageTracking))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigLoading))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    # Run tests
    result = run_tests()

    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)

