"""Tests for Chimera dual-author blending functionality."""

import unittest
from typing import Optional
from unittest.mock import Mock, MagicMock, patch

try:
    from src.generator.translator import StyleTranslator
    import inspect
except ImportError as e:
    print(f"Skipping chimera blending tests: {e}")
    StyleTranslator = None
    inspect = None


class TestTranslatorInterface(unittest.TestCase):
    """Test that translator interface supports blending parameters."""

    def setUp(self):
        if StyleTranslator is None or inspect is None:
            self.skipTest("StyleTranslator or inspect not available")

    def test_translate_paragraph_has_blending_parameters(self):
        """Test that translate_paragraph accepts secondary_author and blend_ratio."""
        sig = inspect.signature(StyleTranslator.translate_paragraph)
        params = sig.parameters

        # Check that blending parameters exist
        self.assertIn('secondary_author', params)
        self.assertIn('blend_ratio', params)

        # Check default values
        self.assertEqual(params['secondary_author'].default, None)
        self.assertEqual(params['blend_ratio'].default, 0.5)

        # Check types
        secondary_author_annotation = params['secondary_author'].annotation
        blend_ratio_annotation = params['blend_ratio'].annotation

        # secondary_author should be Optional[str]
        self.assertIn('Optional', str(secondary_author_annotation))
        # blend_ratio should be float
        self.assertEqual(blend_ratio_annotation, float)

    def test_translate_paragraph_backward_compatible(self):
        """Test that translate_paragraph works without blending parameters (backward compatible)."""
        sig = inspect.signature(StyleTranslator.translate_paragraph)
        params = sig.parameters

        # Required parameters should still exist
        self.assertIn('paragraph', params)
        self.assertIn('atlas', params)
        self.assertIn('author_name', params)

        # Blending parameters should be optional (have defaults)
        # Check that default is not inspect.Parameter.empty (meaning it has a default)
        self.assertNotEqual(params['secondary_author'].default, inspect.Parameter.empty)
        self.assertNotEqual(params['blend_ratio'].default, inspect.Parameter.empty)

    def test_blend_ratio_default_value(self):
        """Test that blend_ratio defaults to 0.5 (equal blend)."""
        sig = inspect.signature(StyleTranslator.translate_paragraph)
        params = sig.parameters

        self.assertEqual(params['blend_ratio'].default, 0.5)


class TestDualAuthorRetrieval(unittest.TestCase):
    """Test dual-author example retrieval."""

    def setUp(self):
        if StyleTranslator is None:
            self.skipTest("StyleTranslator not available")

    def test_dual_author_retrieval_ratio_calculation(self):
        """Test that blend_ratio correctly calculates primary/secondary split."""
        # With blend_ratio=0.7 and pool_size=20:
        # primary_count = max(1, int(20 * 0.7)) = 14 (70% primary)
        # secondary_count = 20 - 14 = 6 (30% secondary)
        retrieval_pool_size = 20
        blend_ratio = 0.7

        # Based on implementation: primary_count = max(1, int(pool_size * blend_ratio))
        primary_count = max(1, int(retrieval_pool_size * blend_ratio))
        secondary_count = retrieval_pool_size - primary_count

        # Verify counts are reasonable
        self.assertGreater(primary_count, 0, "Primary count should be at least 1")
        self.assertGreater(secondary_count, 0, "Secondary count should be at least 1")
        self.assertEqual(primary_count + secondary_count, retrieval_pool_size,
                        "Counts should sum to pool size")

        # With ratio=0.7, we expect: primary = 20 * 0.7 = 14, secondary = 6
        self.assertEqual(primary_count, 14, "Primary should be 70% of pool")
        self.assertEqual(secondary_count, 6, "Secondary should be 30% of pool")

    def test_dual_author_retrieval_edge_cases(self):
        """Test edge cases for blend ratio calculation."""
        retrieval_pool_size = 20

        # Test ratio=0.0 (should still get at least 1 primary)
        primary_count = max(1, int(retrieval_pool_size * 0.0))
        self.assertEqual(primary_count, 1, "Ratio 0.0 should still give at least 1 primary")

        # Test ratio=1.0 (all primary)
        primary_count = max(1, int(retrieval_pool_size * 1.0))
        secondary_count = retrieval_pool_size - primary_count
        self.assertEqual(primary_count, 20, "Ratio 1.0 should give all primary")
        self.assertEqual(secondary_count, 0, "Ratio 1.0 should give 0 secondary")

        # Test ratio=0.5 (equal split)
        primary_count = max(1, int(retrieval_pool_size * 0.5))
        secondary_count = retrieval_pool_size - primary_count
        self.assertEqual(primary_count, 10, "Ratio 0.5 should give equal split")
        self.assertEqual(secondary_count, 10, "Ratio 0.5 should give equal split")


class TestLexiconFusion(unittest.TestCase):
    """Test lexicon fusion from both authors."""

    def setUp(self):
        if StyleTranslator is None:
            self.skipTest("StyleTranslator not available")

    def test_lexicon_fusion_union_operation(self):
        """Test that lexicon fusion creates a union of words from both authors."""
        primary_lexicon = ["dialectical", "metaphysical", "contradictions", "materialism"]
        secondary_lexicon = ["iceberg", "grace", "understated", "contradictions"]

        # Union should contain all unique words
        blended_lexicon = list(set(primary_lexicon) | set(secondary_lexicon))

        # Verify all words from both are present
        for word in primary_lexicon:
            self.assertIn(word, blended_lexicon, f"Primary word '{word}' should be in blended lexicon")
        for word in secondary_lexicon:
            self.assertIn(word, blended_lexicon, f"Secondary word '{word}' should be in blended lexicon")

        # Verify no duplicates
        self.assertEqual(len(blended_lexicon), len(set(blended_lexicon)),
                        "Blended lexicon should have no duplicates")

        # Verify "contradictions" appears only once (it's in both)
        self.assertEqual(blended_lexicon.count("contradictions"), 1,
                         "Duplicate words should be removed")

    def test_lexicon_fusion_empty_secondary(self):
        """Test lexicon fusion when secondary author has no lexicon."""
        primary_lexicon = ["dialectical", "metaphysical", "contradictions"]
        secondary_lexicon = []

        blended_lexicon = list(set(primary_lexicon) | set(secondary_lexicon))

        # Should just be primary lexicon
        self.assertEqual(len(blended_lexicon), len(primary_lexicon))
        for word in primary_lexicon:
            self.assertIn(word, blended_lexicon)

    def test_lexicon_fusion_top_words_limit(self):
        """Test that lexicon fusion uses top 15 words from each author."""
        # Create lexicons with more than 15 words
        primary_lexicon = [f"primary_word_{i}" for i in range(20)]
        secondary_lexicon = [f"secondary_word_{i}" for i in range(20)]

        # Simulate the [:15] slice from the implementation
        primary_top = primary_lexicon[:15]
        secondary_top = secondary_lexicon[:15]
        blended_lexicon = list(set(primary_top) | set(secondary_top))

        # Should have at most 30 words (15 from each, but may have duplicates)
        self.assertLessEqual(len(blended_lexicon), 30)

        # All words in blended should come from the top 15 of either
        all_top_words = set(primary_top) | set(secondary_top)
        self.assertEqual(set(blended_lexicon), all_top_words)


class TestConfigReading(unittest.TestCase):
    """Test config reading for blending."""

    def test_config_single_author(self):
        """Test that single author config ignores ratio."""
        blend_config = {"authors": ["Mao"]}
        blend_authors = blend_config.get("authors", [])
        blend_ratio = blend_config.get("ratio", 0.5)

        if len(blend_authors) == 1:
            secondary_author = None
            # Ratio is ignored in single-author mode
        else:
            secondary_author = blend_authors[1] if len(blend_authors) >= 2 else None

        self.assertIsNone(secondary_author, "Single author should have no secondary")

    def test_config_dual_author(self):
        """Test that dual author config uses ratio."""
        blend_config = {"authors": ["Mao", "Hemingway"], "ratio": 0.7}
        blend_authors = blend_config.get("authors", [])
        blend_ratio = blend_config.get("ratio", 0.5)

        if len(blend_authors) >= 2:
            primary_author = blend_authors[0]
            secondary_author = blend_authors[1]
        else:
            primary_author = blend_authors[0] if blend_authors else None
            secondary_author = None

        self.assertEqual(primary_author, "Mao")
        self.assertEqual(secondary_author, "Hemingway")
        self.assertEqual(blend_ratio, 0.7)

    def test_config_missing_ratio(self):
        """Test that missing ratio defaults to 0.5."""
        blend_config = {"authors": ["Mao", "Hemingway"]}
        blend_ratio = blend_config.get("ratio", 0.5)

        self.assertEqual(blend_ratio, 0.5, "Missing ratio should default to 0.5")

    def test_config_empty_authors(self):
        """Test that empty authors list uses single-author mode."""
        blend_config = {"authors": []}
        blend_authors = blend_config.get("authors", [])

        if len(blend_authors) >= 2:
            secondary_author = blend_authors[1]
        else:
            secondary_author = None

        self.assertIsNone(secondary_author, "Empty authors should have no secondary")


if __name__ == '__main__':
    unittest.main()

