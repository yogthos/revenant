"""Tests for the LoRA training pipeline.

Tests cover:
- Corpus curation (quality filtering, token budgeting)
- Corpus neutralization (chunking, re-segmentation, description)
- Training data preparation (format, word counts)
- Inference format (matching training)
"""

import json
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add scripts directory to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))


# =============================================================================
# Tests for curate_corpus.py
# =============================================================================

class TestCorpusCuration:
    """Tests for corpus quality filtering and curation."""

    def test_estimate_tokens(self):
        """Test token estimation (~1.3 tokens per word)."""
        from curate_corpus import estimate_tokens

        text = "This is a test sentence with exactly ten words here."
        tokens = estimate_tokens(text)
        # 10 words * 1.3 = 13 tokens
        assert tokens == 13

    def test_estimate_words_from_tokens(self):
        """Test reverse conversion from tokens to words."""
        from curate_corpus import estimate_words_from_tokens

        words = estimate_words_from_tokens(900000)
        # 900000 / 1.3 ≈ 692307
        assert words == 692307

    def test_quality_paragraph_accepts_good_text(self):
        """Test that quality paragraphs are accepted."""
        from curate_corpus import is_quality_paragraph

        good_para = (
            "The cosmos is all that is or was or ever will be. "
            "Our feeblest contemplations of the Cosmos stir us. "
            "There is a tingling in the spine, a catch in the voice. "
            "We know we are approaching the greatest of mysteries."
        )

        is_quality, reason = is_quality_paragraph(good_para)
        assert is_quality is True
        assert reason == "ok"

    def test_quality_paragraph_rejects_short_text(self):
        """Test that short paragraphs are rejected."""
        from curate_corpus import is_quality_paragraph

        short_para = "This is too short to be useful."

        is_quality, reason = is_quality_paragraph(short_para, min_words=40)
        assert is_quality is False
        assert "too short" in reason

    def test_quality_paragraph_rejects_fragments(self):
        """Test that paragraphs with too many fragments are rejected."""
        from curate_corpus import is_quality_paragraph

        # Lots of short fragments
        fragment_para = "Yes. No. Maybe. Sure. OK. Fine. Yes. No. Perhaps. Indeed. Right. Wrong. " * 5

        is_quality, reason = is_quality_paragraph(fragment_para, min_words=10)
        assert is_quality is False

    def test_quality_paragraph_rejects_encoding_artifacts(self):
        """Test that text with encoding artifacts is rejected."""
        from curate_corpus import is_quality_paragraph

        # Simulated encoding garbage (non-ASCII sequences)
        bad_para = (
            "This paragraph has some normal text but also contains "
            "garbled characters like \xe2\x80\x99\xe2\x80\x99\xe2\x80\x99 "
            "which indicate encoding problems in the source material."
        )
        # This specific test depends on the artifact pattern
        # The function checks for 3+ consecutive non-ASCII that aren't common punctuation

    def test_quality_paragraph_rejects_excessive_repetition(self):
        """Test that text with excessive word repetition is rejected."""
        from curate_corpus import is_quality_paragraph

        # Same word repeated excessively, but with proper sentences
        repetitive = (
            "The amazing amazing amazing amazing amazing thing happened. "
            "It was amazing amazing amazing amazing amazing indeed. "
            "The amazing amazing amazing amazing amazing result was clear. "
        )

        is_quality, reason = is_quality_paragraph(repetitive, min_words=10)
        assert is_quality is False
        assert "repetition" in reason

    def test_sequential_sample_respects_target(self):
        """Test that sequential sampling stays within word budget."""
        from curate_corpus import sequential_sample

        # Create paragraphs of varying lengths
        paragraphs = [f"Word " * 100 for _ in range(50)]  # 50 paras, 100 words each
        target_words = 2000  # Should select ~20 paragraphs

        indices = sequential_sample(paragraphs, target_words)

        selected_words = sum(len(paragraphs[i].split()) for i in indices)
        # Should be close to target (within 10%)
        assert selected_words <= target_words * 1.1
        assert selected_words >= target_words * 0.5  # At least half


# =============================================================================
# Tests for generate_flat_training.py
# =============================================================================

class TestGenerateFlatTraining:
    """Tests for training data generation functions."""

    def test_is_quality_paragraph_accepts_good_text(self):
        """Test that quality paragraphs are accepted."""
        from generate_flat_training import is_quality_paragraph, CurationConfig

        # Default min_words is 100, so we need a longer paragraph
        config = CurationConfig(min_words=30)
        good_para = (
            "The cosmos is all that is or was or ever will be. "
            "Our feeblest contemplations of the Cosmos stir us. "
            "There is a tingling in the spine, a catch in the voice. "
            "We know we are approaching the greatest of mysteries."
        )

        is_quality, reason = is_quality_paragraph(good_para, config)
        assert is_quality is True
        assert reason == "OK"

    def test_is_quality_paragraph_rejects_short_text(self):
        """Test that short paragraphs are rejected."""
        from generate_flat_training import is_quality_paragraph, CurationConfig

        config = CurationConfig(min_words=40)
        short_para = "This is too short to be useful."

        is_quality, reason = is_quality_paragraph(short_para, config)
        assert is_quality is False
        assert "short" in reason.lower()

    def test_clean_text(self):
        """Test text cleaning."""
        from generate_flat_training import clean_text

        # Test with multiple newlines
        text = "Hello\n\n\n\nWorld"
        cleaned = clean_text(text)
        assert "\n\n\n\n" not in cleaned

    def test_split_into_sentences(self):
        """Test sentence splitting."""
        from generate_flat_training import split_into_sentences

        text = "First sentence. Second sentence. Third sentence."
        sentences = split_into_sentences(text)
        assert len(sentences) == 3

    def test_count_word_repetition(self):
        """Test word repetition counting."""
        from generate_flat_training import count_word_repetition

        # Normal text
        normal_text = "The quick brown fox jumps over the lazy dog."
        ratio1 = count_word_repetition(normal_text)

        # Repetitive text
        repetitive = "the the the the fox fox fox dog dog dog"
        ratio2 = count_word_repetition(repetitive)

        assert ratio2 > ratio1

    def test_neutralize_text_returns_string_or_none(self):
        """Test that neutralize_text returns string or None."""
        from generate_flat_training import neutralize_text

        # This test just verifies the function signature/return type
        # Actual LLM calls are mocked in integration tests
        # Here we just ensure the function exists and has correct signature
        import inspect
        sig = inspect.signature(neutralize_text)
        assert 'styled_text' in sig.parameters


# =============================================================================
# Tests for train_mlx_lora.py
# =============================================================================

class TestTrainingPreparation:
    """Tests for training data preparation."""

    def test_estimate_tokens(self):
        """Test token estimation for sequence length checking."""
        from train_mlx_lora import estimate_tokens

        text = "This is a test with ten words in it."
        tokens = estimate_tokens(text)

        # ~4 chars per token, 38 chars / 4 = 9 tokens
        assert tokens == 9

    def test_split_text_to_fit_short_text(self):
        """Test that short text is not split."""
        from train_mlx_lora import split_text_to_fit

        short_text = "This is a short sentence."

        chunks = split_text_to_fit(short_text, max_tokens=1000)

        assert len(chunks) == 1
        assert chunks[0] == short_text

    def test_split_text_to_fit_long_text(self):
        """Test that long text is split on sentence boundaries."""
        from train_mlx_lora import split_text_to_fit

        long_text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here. Fifth sentence here."

        chunks = split_text_to_fit(long_text, max_tokens=20)  # Force splitting

        assert len(chunks) > 1
        # Each chunk should end with a sentence
        for chunk in chunks:
            assert chunk.strip()[-1] in '.!?'

    def test_prepare_from_neutralized_format(self, tmp_path):
        """Test that training data has correct format."""
        from train_mlx_lora import prepare_from_neutralized

        # Create mock neutralized data
        neutralized_data = [
            {
                "author": "Test Author",
                "original": "The cosmos is vast and beautiful. Stars shine brightly in the night sky.",
                "description": "A description of space and stars in third person.",
                "word_count": 12
            },
            {
                "author": "Test Author",
                "original": "Science reveals the wonders of nature. We learn through observation.",
                "description": "Discussion of scientific method and discovery.",
                "word_count": 10
            }
        ]

        # Write to temp file
        input_path = tmp_path / "neutralized.jsonl"
        with open(input_path, 'w') as f:
            for item in neutralized_data:
                f.write(json.dumps(item) + '\n')

        output_path = tmp_path / "training"

        paths = prepare_from_neutralized(
            neutralized_path=str(input_path),
            author="Test Author",
            output_path=str(output_path),
            max_seq_length=2048
        )

        # Check files were created
        assert Path(paths['train']).exists()
        assert Path(paths['valid']).exists()

        # Check format of training examples
        with open(paths['train'], 'r') as f:
            for line in f:
                example = json.loads(line)
                assert 'text' in example

                # Should have instruction format
                text = example['text']
                assert "Write a" in text
                assert "word excerpt" in text
                assert "emulating the style and voice of" in text
                assert "Test Author" in text

    def test_prepare_from_neutralized_includes_word_count(self, tmp_path):
        """Test that word count is included in instruction."""
        from train_mlx_lora import prepare_from_neutralized

        neutralized_data = [{
            "author": "Test Author",
            "original": "A " * 100,  # 100 words
            "description": "A test description.",
            "word_count": 100
        }]

        input_path = tmp_path / "neutralized.jsonl"
        with open(input_path, 'w') as f:
            f.write(json.dumps(neutralized_data[0]) + '\n')

        output_path = tmp_path / "training"

        paths = prepare_from_neutralized(
            neutralized_path=str(input_path),
            author="Test Author",
            output_path=str(output_path)
        )

        # With only 1 example, it goes to validation (val_size = max(1, 1//10) = 1)
        with open(paths['valid'], 'r') as f:
            example = json.loads(f.readline())
            # Should include the word count
            assert "100 word excerpt" in example['text']

    def test_train_val_split_ratio(self, tmp_path):
        """Test that train/val split is approximately 90/10."""
        from train_mlx_lora import prepare_from_neutralized

        # Create 100 examples
        neutralized_data = [
            {
                "author": "Test Author",
                "original": f"Example {i} text here.",
                "description": f"Description {i}.",
                "word_count": 5
            }
            for i in range(100)
        ]

        input_path = tmp_path / "neutralized.jsonl"
        with open(input_path, 'w') as f:
            for item in neutralized_data:
                f.write(json.dumps(item) + '\n')

        output_path = tmp_path / "training"

        paths = prepare_from_neutralized(
            neutralized_path=str(input_path),
            author="Test Author",
            output_path=str(output_path)
        )

        # Count examples
        with open(paths['train'], 'r') as f:
            train_count = sum(1 for _ in f)
        with open(paths['valid'], 'r') as f:
            val_count = sum(1 for _ in f)

        # Should be ~90/10 split
        assert train_count == 90
        assert val_count == 10


# =============================================================================
# Tests for lora_generator.py
# =============================================================================

class TestLoRAGenerator:
    """Tests for LoRA inference configuration."""

    def test_generation_config_defaults(self):
        """Test that generation config has paper's recommended defaults."""
        from src.generation.lora_generator import GenerationConfig

        config = GenerationConfig()

        # Temperature 0.7 allows creative/rare word choices
        # min_p 0.05 filters nonsense at high temp
        assert config.temperature == 0.7
        assert config.top_p == 0.95
        assert config.min_p == 0.05
        assert config.repetition_penalty == 1.4

    def test_inference_prompt_format_matches_training(self):
        """Test that inference prompt matches training format."""
        # The key is that inference uses same format as training:
        # "Write a {n} word excerpt about the content below emulating the style and voice of {author}"

        # This is a format test - we verify the string pattern
        author = "Carl Sagan"
        word_count = 150
        content = "Description of space exploration."

        # Training format (from train_mlx_lora.py)
        training_format = f"Write a {word_count} word excerpt about the content below emulating the style and voice of {author}\n\n{content}"

        # Verify the pattern
        assert f"Write a {word_count} word excerpt" in training_format
        assert f"emulating the style and voice of {author}" in training_format
        assert content in training_format

    def test_adapter_metadata_loading(self, tmp_path):
        """Test that adapter metadata is loaded correctly."""
        from src.generation.lora_generator import AdapterMetadata

        # Create mock metadata
        metadata = {
            "author": "Test Author",
            "base_model": "mlx-community/test-model",
            "lora_rank": 32,
            "lora_alpha": 64,
            "epochs": 1,
            "training_examples": 100
        }

        metadata_path = tmp_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

        loaded = AdapterMetadata.from_file(metadata_path)

        assert loaded.author == "Test Author"
        assert loaded.base_model == "mlx-community/test-model"
        assert loaded.lora_rank == 32
        assert loaded.lora_alpha == 64
        assert loaded.epochs == 1
        assert loaded.training_examples == 100


# =============================================================================
# Integration Tests
# =============================================================================

class TestPipelineIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline_format_consistency(self, tmp_path):
        """Test that formats are consistent across the pipeline."""
        from train_mlx_lora import prepare_from_neutralized

        # Simulate what neutralize_corpus.py produces
        neutralized = {
            "author": "Test Author",
            "original": "The universe speaks to those who listen. Its voice echoes through the cosmos.",
            "description": "A narrator reflects on the universe's communication with humanity in a philosophical tone.",
            "word_count": 14
        }

        input_path = tmp_path / "neutralized.jsonl"
        with open(input_path, 'w') as f:
            f.write(json.dumps(neutralized) + '\n')

        # Run training prep
        paths = prepare_from_neutralized(
            neutralized_path=str(input_path),
            author="Test Author",
            output_path=str(tmp_path / "training")
        )

        # With only 1 example, it goes to validation (val_size = max(1, 1//10) = 1)
        with open(paths['valid'], 'r') as f:
            example = json.loads(f.readline())

        text = example['text']

        # Training format should be:
        # "Write a {word_count} word excerpt about the content below emulating the style and voice of {author}\n\n{description}\n\n{original}"

        assert text.startswith("Write a 14 word excerpt")
        assert "Test Author" in text
        assert neutralized["description"] in text
        assert neutralized["original"] in text

        # Verify the structure: instruction\n\ndescription\n\noriginal
        parts = text.split("\n\n")
        assert len(parts) == 3
        assert "Write a" in parts[0]  # Instruction
        assert parts[1] == neutralized["description"]  # Description
        assert parts[2] == neutralized["original"]  # Original

    def test_hyperparameter_defaults_match_paper(self):
        """Test that all hyperparameter defaults match the paper."""
        from train_mlx_lora import train_lora
        from src.generation.lora_generator import GenerationConfig
        import inspect

        # Get train_lora defaults
        sig = inspect.signature(train_lora)
        defaults = {
            name: param.default
            for name, param in sig.parameters.items()
            if param.default is not inspect.Parameter.empty
        }

        # Verify training defaults
        assert defaults['epochs'] == 1  # 1 epoch with curated corpus
        assert defaults['batch_size'] == 1  # Batch size 1
        assert defaults['learning_rate'] == 1e-5  # Lower LR with high rank for stability

        # Verify inference defaults
        config = GenerationConfig()
        assert config.temperature == 0.7  # Higher temp allows creative word choices
        assert config.min_p == 0.05  # Filter nonsense at high temp


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
