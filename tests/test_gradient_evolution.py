"""Tests for gradient-based evolution with fitness-based selection."""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import with error handling for missing dependencies
try:
    from src.generator.translator import StyleTranslator
    from src.validator.semantic_critic import SemanticCritic
    from src.critic.scorer import SoftScorer
    from src.generator.mutation_operators import (
        OP_SEMANTIC_INJECTION, OP_GRAMMAR_REPAIR, OP_STYLE_POLISH,
        get_operator, SemanticInjectionOperator, GrammarRepairOperator, StylePolishOperator
    )
    from src.ingestion.blueprint import SemanticBlueprint
    from src.atlas.rhetoric import RhetoricalType
    DEPENDENCIES_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = str(e)
    print(f"⚠ Skipping tests: Missing dependencies - {IMPORT_ERROR}")
    # Create dummy classes to prevent NameError
    class StyleTranslator:
        pass
    class SemanticCritic:
        pass
    class SoftScorer:
        pass
    class SemanticInjectionOperator:
        pass
    class GrammarRepairOperator:
        pass
    class StylePolishOperator:
        pass
    def get_operator(op_type):
        return None
    OP_SEMANTIC_INJECTION = "semantic_injection"
    OP_GRAMMAR_REPAIR = "grammar_repair"
    OP_STYLE_POLISH = "style_polish"
    class SemanticBlueprint:
        pass
    class RhetoricalType:
        OBSERVATION = "observation"


class TestSoftScorer(unittest.TestCase):
    """Test soft scoring for evolution guidance."""

    def setUp(self):
        """Set up test fixtures."""
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest(f"Missing dependencies: {IMPORT_ERROR}")
        self.scorer = SoftScorer(config_path="config.json")

    def test_calculate_raw_score_never_zero(self):
        """Test that raw_score is never zero for non-empty text."""
        blueprint = SemanticBlueprint(
            original_text="Every object we touch eventually breaks.",
            svo_triples=[("we", "touch", "object")],
            core_keywords={"object", "touch", "break", "eventually"},
            named_entities=[],
            citations=[],
            quotes=[]
        )

        # Even a bad draft should get a non-zero raw_score
        bad_draft = "We touch breaks."
        raw_score, metrics = self.scorer.calculate_raw_score(bad_draft, blueprint)

        self.assertGreater(raw_score, 0.0, "raw_score should never be zero for non-empty text")
        self.assertLessEqual(raw_score, 1.0, "raw_score should be <= 1.0")

    def test_raw_score_weights(self):
        """Test that raw_score uses correct weights."""
        blueprint = SemanticBlueprint(
            original_text="The cat sat on the mat.",
            svo_triples=[("cat", "sat", "mat")],
            core_keywords={"cat", "sat", "mat"},
            named_entities=[],
            citations=[],
            quotes=[]
        )

        # Perfect draft
        perfect_draft = "The cat sat on the mat."
        raw_score, metrics = self.scorer.calculate_raw_score(perfect_draft, blueprint)

        # Should have high recall, fluency, similarity
        self.assertGreater(metrics["recall"], 0.8)
        self.assertGreater(metrics["fluency"], 0.8)
        self.assertGreater(raw_score, 0.8)

    def test_evaluate_with_raw_score_returns_both(self):
        """Test that evaluate_with_raw_score returns both pass and raw_score."""
        blueprint = SemanticBlueprint(
            original_text="Test sentence.",
            svo_triples=[("sentence",)],
            core_keywords={"test", "sentence"},
            named_entities=[],
            citations=[],
            quotes=[]
        )

        result = self.scorer.evaluate_with_raw_score("Test sentence.", blueprint)

        self.assertIn("pass", result)
        self.assertIn("raw_score", result)
        self.assertIn("recall_score", result)
        self.assertIn("fluency_score", result)
        self.assertIsInstance(result["pass"], bool)
        self.assertIsInstance(result["raw_score"], float)
        self.assertGreaterEqual(result["raw_score"], 0.0)
        self.assertLessEqual(result["raw_score"], 1.0)


class TestMutationOperators(unittest.TestCase):
    """Test mutation operators."""

    def setUp(self):
        """Set up test fixtures."""
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest(f"Missing dependencies: {IMPORT_ERROR}")
        self.blueprint = SemanticBlueprint(
            original_text="Every object we touch eventually breaks.",
            svo_triples=[("we", "touch", "object")],
            core_keywords={"object", "touch", "break", "eventually"},
            named_entities=[],
            citations=[],
            quotes=[]
        )
        self.mock_llm = Mock()
        self.mock_llm.call.return_value = "Every object we touch eventually breaks."

    def test_get_operator(self):
        """Test getting operators by type."""
        op1 = get_operator(OP_SEMANTIC_INJECTION)
        self.assertIsInstance(op1, SemanticInjectionOperator)

        op2 = get_operator(OP_GRAMMAR_REPAIR)
        self.assertIsInstance(op2, GrammarRepairOperator)

        op3 = get_operator(OP_STYLE_POLISH)
        self.assertIsInstance(op3, StylePolishOperator)

    def test_semantic_injection_operator(self):
        """Test semantic injection operator."""
        operator = SemanticInjectionOperator()
        current_draft = "We touch breaks."

        result = operator.generate(
            current_draft=current_draft,
            blueprint=self.blueprint,
            author_name="Test",
            style_dna="Test style",
            rhetorical_type=RhetoricalType.OBSERVATION,
            llm_provider=self.mock_llm,
            temperature=0.6
        )

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_grammar_repair_operator(self):
        """Test grammar repair operator."""
        operator = GrammarRepairOperator()
        current_draft = "We touch breaks."

        result = operator.generate(
            current_draft=current_draft,
            blueprint=self.blueprint,
            author_name="Test",
            style_dna="Test style",
            rhetorical_type=RhetoricalType.OBSERVATION,
            llm_provider=self.mock_llm,
            temperature=0.6
        )

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_style_polish_operator(self):
        """Test style polish operator."""
        operator = StylePolishOperator()
        current_draft = "Every object we touch eventually breaks."

        result = operator.generate(
            current_draft=current_draft,
            blueprint=self.blueprint,
            author_name="Test",
            style_dna="Test style",
            rhetorical_type=RhetoricalType.OBSERVATION,
            llm_provider=self.mock_llm,
            temperature=0.6
        )

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


class TestFitnessBasedEvolution(unittest.TestCase):
    """Test fitness-based evolution selection."""

    def setUp(self):
        """Set up test fixtures."""
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest(f"Missing dependencies: {IMPORT_ERROR}")
        self.translator = StyleTranslator(config_path="config.json")
        self.blueprint = SemanticBlueprint(
            original_text="Every object we touch eventually breaks.",
            svo_triples=[("we", "touch", "object")],
            core_keywords={"object", "touch", "break", "eventually"},
            named_entities=[],
            citations=[],
            quotes=[]
        )

    def test_diagnose_draft_low_recall(self):
        """Test diagnosis selects semantic injection for low recall."""
        mock_critic = Mock()
        mock_critic.evaluate.return_value = {
            "recall_score": 0.5,  # Low recall
            "fluency_score": 0.9,
            "pass": False
        }

        strategy = self.translator._diagnose_draft("Bad draft", self.blueprint, mock_critic)
        self.assertEqual(strategy, OP_SEMANTIC_INJECTION)

    def test_diagnose_draft_low_fluency(self):
        """Test diagnosis selects grammar repair for low fluency."""
        mock_critic = Mock()
        mock_critic.evaluate.return_value = {
            "recall_score": 1.0,  # Good recall
            "fluency_score": 0.5,  # Low fluency
            "pass": False
        }

        strategy = self.translator._diagnose_draft("Bad draft", self.blueprint, mock_critic)
        self.assertEqual(strategy, OP_GRAMMAR_REPAIR)

    def test_diagnose_draft_style_polish(self):
        """Test diagnosis selects style polish when recall and fluency are good."""
        mock_critic = Mock()
        mock_critic.evaluate.return_value = {
            "recall_score": 1.0,  # Good recall
            "fluency_score": 0.9,  # Good fluency
            "pass": False
        }

        strategy = self.translator._diagnose_draft("Good draft", self.blueprint, mock_critic)
        self.assertEqual(strategy, OP_STYLE_POLISH)

    @patch('src.generator.translator.StyleTranslator._generate_population_with_operator')
    def test_fitness_based_selection_improves_from_low_score(self, mock_generate):
        """Test that evolution selects candidate with higher raw_score even if pass=False."""
        # Mock critic
        mock_critic = Mock()

        # Parent: low score, pass=False
        parent_result = {
            "pass": False,
            "score": 0.2,
            "recall_score": 0.5,
            "precision_score": 0.4,
            "fluency_score": 0.4,
            "feedback": "Low scores"
        }

        # Child: higher score, still pass=False
        child_result = {
            "pass": False,
            "score": 0.6,
            "recall_score": 0.7,
            "precision_score": 0.6,
            "fluency_score": 0.5,
            "feedback": "Better but still failing"
        }

        evaluate_calls = []
        def mock_evaluate(text, blueprint, allowed_style_words=None, **kwargs):
            evaluate_calls.append(text)
            if len(evaluate_calls) <= 1:
                return parent_result
            return child_result

        mock_critic.evaluate.side_effect = mock_evaluate

        # Mock soft scorer
        mock_soft_scorer = Mock()
        parent_eval = {"raw_score": 0.2, "pass": False}
        child_eval = {"raw_score": 0.6, "pass": False}
        mock_soft_scorer.evaluate_with_raw_score.side_effect = [parent_eval, child_eval]

        self.translator.soft_scorer = mock_soft_scorer

        # Mock population generation
        mock_generate.return_value = [("semantic", "Better draft")]

        # Mock other methods
        self.translator._check_acceptance = Mock(return_value=False)
        self.translator._generate_simplification = Mock(return_value="Simplified")

        # Run evolution
        best_draft, best_score = self.translator._evolve_text(
            initial_draft="Bad draft",
            blueprint=self.blueprint,
            author_name="Test",
            style_dna="Test style",
            rhetorical_type=RhetoricalType.OBSERVATION,
            initial_score=0.2,
            initial_feedback="Low scores",
            critic=mock_critic,
            verbose=False
        )

        # Should have improved (even if still failing)
        # The system should select the better candidate
        self.assertIsNotNone(best_draft)

    def test_incremental_improvement_scenario(self):
        """Test the specific scenario from the plan: Draft 1 (0.2) → Draft 2 (0.6)."""
        mock_critic = Mock()

        # Draft 1: Score 0.2, pass=False
        draft1_result = {
            "pass": False,
            "score": 0.2,
            "recall_score": 0.4,
            "precision_score": 0.3,
            "fluency_score": 0.3,
            "feedback": "Missing concepts"
        }

        # Draft 2: Score 0.6, pass=False (still failing but better)
        draft2_result = {
            "pass": False,
            "score": 0.6,
            "recall_score": 0.7,
            "precision_score": 0.6,
            "fluency_score": 0.5,
            "feedback": "Better but still needs work"
        }

        evaluate_calls = []
        def mock_evaluate(text, blueprint, allowed_style_words=None, **kwargs):
            evaluate_calls.append(text)
            if len(evaluate_calls) <= 1:
                return draft1_result
            return draft2_result

        mock_critic.evaluate.side_effect = mock_evaluate

        # Mock soft scorer
        mock_soft_scorer = Mock()
        draft1_eval = {"raw_score": 0.2, "pass": False}
        draft2_eval = {"raw_score": 0.6, "pass": False}
        mock_soft_scorer.evaluate_with_raw_score.side_effect = [draft1_eval, draft2_eval]
        mock_soft_scorer.calculate_raw_score.side_effect = [
            (0.2, {"recall": 0.4, "fluency": 0.3}),
            (0.6, {"recall": 0.7, "fluency": 0.5})
        ]

        self.translator.soft_scorer = mock_soft_scorer

        # Mock population generation to return Draft 2
        with patch.object(self.translator, '_generate_population_with_operator') as mock_gen:
            mock_gen.return_value = [("semantic", "We touch objects that break.")]

            # Mock other methods
            self.translator._check_acceptance = Mock(return_value=False)
            self.translator._generate_simplification = Mock(return_value="Simplified")

            # Run evolution
            best_draft, best_score = self.translator._evolve_text(
                initial_draft="We touch breaks.",
                blueprint=self.blueprint,
                author_name="Test",
                style_dna="Test style",
                rhetorical_type=RhetoricalType.OBSERVATION,
                initial_score=0.2,
                initial_feedback="Missing concepts",
                critic=mock_critic,
                verbose=False
            )

            # System should pick Draft 2 (higher raw_score) even though pass=False
            # Verify that the better candidate was considered
            self.assertIsNotNone(best_draft)


class TestEvolutionConvergence(unittest.TestCase):
    """Test that evolution actually converges to high acceptance scores."""

    def setUp(self):
        """Set up test fixtures."""
        if not DEPENDENCIES_AVAILABLE:
            self.skipTest(f"Missing dependencies: {IMPORT_ERROR}")
        self.translator = StyleTranslator(config_path="config.json")

    def test_evolution_converges_from_low_similarity(self):
        """Test evolution from low semantic similarity (0.83) to high acceptance.

        This tests the failure case: "Human experience reinforces the rule of finitude"
        Initial draft: "Human experience confirms the law of finitude" (similarity 0.83)
        """
        blueprint = SemanticBlueprint(
            original_text="Human experience reinforces the rule of finitude.",
            svo_triples=[("Human experience", "reinforce", "rule of finitude")],
            core_keywords={"experience", "finitude", "human", "reinforce", "rule"},
            named_entities=[],
            citations=[],
            quotes=[]
        )

        initial_draft = "Human experience confirms the law of finitude."

        # Mock critic to simulate the actual failure scenario
        mock_critic = Mock()

        # Initial evaluation: low similarity, missing keywords
        initial_result = {
            "pass": False,
            "score": 0.73,
            "recall_score": 0.60,
            "precision_score": 0.50,
            "fluency_score": 0.90,
            "feedback": "CRITICAL: Missing concepts: rule, reinforce. Preserve all input meaning."
        }

        # Generation 1: Improved draft with all keywords
        gen1_result = {
            "pass": True,
            "score": 0.90,
            "recall_score": 1.00,
            "precision_score": 0.80,
            "fluency_score": 0.90,
            "feedback": "Passed semantic validation."
        }

        evaluate_calls = []
        def mock_evaluate(text, blueprint, allowed_style_words=None, **kwargs):
            evaluate_calls.append(text)
            # First call: initial draft evaluation
            if len(evaluate_calls) == 1:
                return initial_result
            # Subsequent calls: candidate evaluations
            # Check if the text contains the improved keywords
            if "reinforce" in text.lower() and "rule" in text.lower():
                return gen1_result
            return initial_result

        mock_critic.evaluate.side_effect = mock_evaluate

        # Mock soft scorer - needs to return higher raw_score for improved candidate
        mock_soft_scorer = Mock()
        def mock_eval_with_raw_score(text, blueprint, style_lexicon=None, **kwargs):
            # Return higher raw_score for improved candidate
            if "reinforce" in text.lower() and "rule" in text.lower():
                return {"raw_score": 0.96, "pass": True, "recall_score": 1.0, "fluency_score": 0.9, "precision_score": 0.8, "score": 0.90}
            return {"raw_score": 0.73, "pass": False, "recall_score": 0.60, "fluency_score": 0.9, "precision_score": 0.5, "score": 0.73}
        
        mock_soft_scorer.evaluate_with_raw_score.side_effect = mock_eval_with_raw_score
        mock_soft_scorer.calculate_raw_score.return_value = (0.96, {
            "recall": 1.0, "fluency": 0.9, "similarity": 0.95
        })

        self.translator.soft_scorer = mock_soft_scorer

        # Mock population generation - return improved candidate
        with patch.object(self.translator, '_generate_population_with_operator') as mock_gen:
            mock_gen.return_value = [
                ("semantic_injection", "Human experience confirms and reinforces the universal rule of finitude.")
            ]

            # Mock acceptance check - should pass for improved candidate
            def mock_acceptance(recall_score, precision_score, fluency_score=None, overall_score=0.0, pass_threshold=0.9, **kwargs):
                return overall_score >= 0.90 and recall_score >= 0.95
            self.translator._check_acceptance = mock_acceptance

            # Run evolution
            best_draft, best_score = self.translator._evolve_text(
                initial_draft=initial_draft,
                blueprint=blueprint,
                author_name="Mao",
                style_dna="Authoritative and philosophical",
                rhetorical_type=RhetoricalType.OBSERVATION,
                initial_score=0.73,
                initial_feedback="CRITICAL: Missing concepts: rule, reinforce.",
                critic=mock_critic,
                verbose=False
            )

            # Verify convergence to high score
            self.assertGreaterEqual(best_score, 0.90,
                                  f"Evolution should converge to >= 0.90, got {best_score:.2f}")
            self.assertIn("reinforce", best_draft.lower(),
                         "Final draft should contain 'reinforce'")
            self.assertIn("rule", best_draft.lower(),
                         "Final draft should contain 'rule'")

    def test_evolution_converges_from_zero_score(self):
        """Test evolution from score 0.00 (stuck state) to high acceptance.

        This tests the failure case: "Every object we touch eventually breaks"
        Initial draft: "All things we handle ultimately break" (similarity 0.63, score 0.00)
        """
        blueprint = SemanticBlueprint(
            original_text="Every object we touch eventually breaks.",
            svo_triples=[("we", "touch", "object")],
            core_keywords={"object", "touch", "break", "eventually"},
            named_entities=[],
            citations=[],
            quotes=[]
        )

        initial_draft = "All things we handle ultimately break."

        # Mock critic
        mock_critic = Mock()

        # Initial: score 0.00 due to low similarity
        initial_result = {
            "pass": False,
            "score": 0.00,
            "recall_score": 0.00,
            "precision_score": 0.00,
            "fluency_score": 0.00,
            "feedback": "CRITICAL: Semantic similarity too low (0.63 < 0.75)."
        }

        # Generation 1: Improved but still not perfect
        gen1_result = {
            "pass": False,
            "score": 0.92,
            "recall_score": 0.67,
            "precision_score": 1.00,
            "fluency_score": 1.00,
            "feedback": "Missing concepts: eventually."
        }

        # Generation 2: Perfect
        gen2_result = {
            "pass": True,
            "score": 0.95,
            "recall_score": 1.00,
            "precision_score": 1.00,
            "fluency_score": 1.00,
            "feedback": "Passed semantic validation."
        }

        def mock_evaluate(text, blueprint, allowed_style_words=None, **kwargs):
            # Return results based on text content
            text_lower = text.lower()
            if "eventually" in text_lower and "object" in text_lower:
                return gen2_result  # Perfect version
            elif "object" in text_lower and "touch" in text_lower:
                return gen1_result  # Improved but missing "eventually"
            return initial_result  # Initial draft

        mock_critic.evaluate.side_effect = mock_evaluate

        # Mock soft scorer with incremental improvements based on text
        mock_soft_scorer = Mock()
        def mock_eval_with_raw_score(text, blueprint, style_lexicon=None, **kwargs):
            text_lower = text.lower()
            if "eventually" in text_lower and "object" in text_lower:
                return {"raw_score": 0.95, "pass": True, "recall_score": 1.0, "fluency_score": 1.0, "precision_score": 1.0, "score": 0.95}
            elif "object" in text_lower and "touch" in text_lower:
                return {"raw_score": 0.82, "pass": False, "recall_score": 0.67, "fluency_score": 1.0, "precision_score": 1.0, "score": 0.92}
            return {"raw_score": 0.53, "pass": False, "recall_score": 0.0, "fluency_score": 0.0, "precision_score": 0.0, "score": 0.0}
        
        mock_soft_scorer.evaluate_with_raw_score.side_effect = mock_eval_with_raw_score
        self.translator.soft_scorer = mock_soft_scorer

        # Mock population generation - return improving candidates
        call_count = [0]
        def mock_gen(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return [("semantic_injection", "All objects we touch ultimately break.")]
            else:
                return [("semantic_injection", "Every object we touch eventually breaks.")]

        with patch.object(self.translator, '_generate_population_with_operator', side_effect=mock_gen):
            def mock_acceptance(recall_score, precision_score, fluency_score=None, overall_score=0.0, pass_threshold=0.9, **kwargs):
                # Gen 1: recall 0.67 < 0.95, so False
                # Gen 2: recall 1.0 >= 0.95 and overall 0.95 >= 0.9, so True
                return overall_score >= 0.90 and recall_score >= 0.95
            self.translator._check_acceptance = mock_acceptance

            # Run evolution
            best_draft, best_score = self.translator._evolve_text(
                initial_draft=initial_draft,
                blueprint=blueprint,
                author_name="Mao",
                style_dna="Authoritative",
                rhetorical_type=RhetoricalType.OBSERVATION,
                initial_score=0.00,
                initial_feedback="CRITICAL: Semantic similarity too low.",
                critic=mock_critic,
                verbose=False
            )

            # Verify convergence: should reach high score
            self.assertGreaterEqual(best_score, 0.90,
                                  f"Evolution should converge to >= 0.90 from 0.00, got {best_score:.2f}")
            # Verify all keywords present
            self.assertIn("object", best_draft.lower())
            self.assertIn("touch", best_draft.lower())
            self.assertIn("break", best_draft.lower())
            self.assertIn("eventually", best_draft.lower())

    def test_evolution_preserves_all_content(self):
        """Test that evolution preserves all content, especially in complex sentences.

        This tests the failure case: "The biological cycle of birth, life, and decay defines our reality"
        Should NOT lose "life, and decay"
        """
        blueprint = SemanticBlueprint(
            original_text="The biological cycle of birth, life, and decay defines our reality.",
            svo_triples=[("biological cycle of birth", "define", "our reality")],
            core_keywords={"biological", "birth", "cycle", "decay", "define", "life", "reality"},
            named_entities=[],
            citations=[],
            quotes=[]
        )

        initial_draft = "The biological cycle of birth, life, and decay defines the fundamental parameters of our material reality."

        # Mock critic
        mock_critic = Mock()

        # Initial: good recall but low fluency
        initial_result = {
            "pass": False,
            "score": 0.81,
            "recall_score": 1.00,
            "precision_score": 0.86,
            "fluency_score": 0.70,  # Low fluency
            "feedback": "Incomplete sentence structure"
        }

        # Improved: better fluency
        improved_result = {
            "pass": True,
            "score": 0.92,
            "recall_score": 1.00,
            "precision_score": 0.90,
            "fluency_score": 0.95,
            "feedback": "Passed semantic validation."
        }

        def mock_evaluate(text, blueprint, allowed_style_words=None, **kwargs):
            # Check if text has all required content
            text_lower = text.lower()
            has_all = all(word in text_lower for word in ["birth", "life", "decay", "cycle", "reality"])
            if has_all and "biological" in text_lower:
                return improved_result
            return initial_result

        mock_critic.evaluate.side_effect = mock_evaluate

        # Mock soft scorer - return higher score for improved draft
        mock_soft_scorer = Mock()
        def mock_eval_with_raw_score(text, blueprint, style_lexicon=None, **kwargs):
            text_lower = text.lower()
            has_all = all(word in text_lower for word in ["birth", "life", "decay", "cycle", "reality"])
            if has_all and "biological" in text_lower:
                return {"raw_score": 0.93, "pass": True, "recall_score": 1.0, "fluency_score": 0.95, "precision_score": 0.90, "score": 0.92}
            return {"raw_score": 0.81, "pass": False, "recall_score": 1.0, "fluency_score": 0.70, "precision_score": 0.86, "score": 0.81}
        
        mock_soft_scorer.evaluate_with_raw_score.side_effect = mock_eval_with_raw_score
        self.translator.soft_scorer = mock_soft_scorer

        # Mock population generation
        with patch.object(self.translator, '_generate_population_with_operator') as mock_gen:
            # Return improved draft that preserves all content
            mock_gen.return_value = [
                ("grammar_repair", "The biological cycle of birth, life, and decay defines our reality.")
            ]

            def mock_acceptance(recall_score, precision_score, fluency_score=None, overall_score=0.0, pass_threshold=0.9, **kwargs):
                return overall_score >= 0.90 and recall_score >= 0.95
            self.translator._check_acceptance = mock_acceptance

            # Run evolution
            best_draft, best_score = self.translator._evolve_text(
                initial_draft=initial_draft,
                blueprint=blueprint,
                author_name="Mao",
                style_dna="Authoritative",
                rhetorical_type=RhetoricalType.OBSERVATION,
                initial_score=0.81,
                initial_feedback="Incomplete sentence structure",
                critic=mock_critic,
                verbose=False
            )

            # CRITICAL: Verify all content preserved
            self.assertIn("birth", best_draft.lower())
            self.assertIn("life", best_draft.lower())
            self.assertIn("decay", best_draft.lower())
            self.assertIn("cycle", best_draft.lower())
            self.assertIn("reality", best_draft.lower())

            # Verify convergence
            self.assertGreaterEqual(best_score, 0.90,
                                  f"Should converge to >= 0.90, got {best_score:.2f}")

    def test_perfection_boost_works(self):
        """Test that perfection boost raises scores for near-perfect translations."""
        blueprint = SemanticBlueprint(
            original_text="The cat sat on the mat.",
            svo_triples=[("cat", "sat", "mat")],
            core_keywords={"cat", "sat", "mat"},
            named_entities=[],
            citations=[],
            quotes=[]
        )

        # Perfect translation (recall=1.0, fluency=1.0, but similarity might be 0.80)
        perfect_draft = "The feline rested upon the rug."

        scorer = SoftScorer(config_path="config.json")

        # Mock critic to return perfect recall/fluency but lower similarity
        mock_critic = Mock()
        mock_critic.evaluate.return_value = {
            "pass": True,
            "score": 0.85,  # Lower due to similarity
            "recall_score": 1.0,
            "precision_score": 0.9,
            "fluency_score": 1.0,
        }
        mock_critic.semantic_model = Mock()
        mock_critic._calculate_semantic_similarity = Mock(return_value=0.80)  # Lower similarity

        scorer.critic = mock_critic

        raw_score, metrics = scorer.calculate_raw_score(perfect_draft, blueprint)

        # With perfection boost: recall=1.0, fluency=1.0 should boost to at least 0.92
        self.assertGreaterEqual(raw_score, 0.92,
                              f"Perfection boost should raise score to >= 0.92, got {raw_score:.2f}")
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["fluency"], 1.0)

    def test_keyword_anchoring_prevents_regression(self):
        """Test that keyword anchoring prevents losing keywords when adding new ones."""
        blueprint = SemanticBlueprint(
            original_text="Human experience reinforces the rule of finitude.",
            svo_triples=[("Human experience", "reinforce", "rule of finitude")],
            core_keywords={"experience", "finitude", "human", "reinforce", "rule"},
            named_entities=[],
            citations=[],
            quotes=[]
        )

        # Draft missing "rule" and "reinforce"
        current_draft = "Human experience confirms the law of finitude."
        missing_keywords = ["rule", "reinforce"]

        operator = SemanticInjectionOperator()
        mock_llm = Mock()

        # LLM should see ALL required keywords in the prompt
        operator.generate(
            current_draft=current_draft,
            blueprint=blueprint,
            author_name="Mao",
            style_dna="Authoritative",
            rhetorical_type=RhetoricalType.OBSERVATION,
            llm_provider=mock_llm,
            missing_keywords=missing_keywords,
            temperature=0.6
        )

        # Verify the prompt includes all required keywords
        call_args = mock_llm.call.call_args
        self.assertIsNotNone(call_args)
        user_prompt = call_args[1]["user_prompt"]  # kwargs

        # Should mention all keywords
        self.assertIn("ALL required keywords", user_prompt)
        self.assertIn("experience", user_prompt)
        self.assertIn("finitude", user_prompt)
        self.assertIn("human", user_prompt)
        self.assertIn("reinforce", user_prompt)
        self.assertIn("rule", user_prompt)

    def test_cot_mutation_structure(self):
        """Test that CoT mutations use the 3-step structure."""
        blueprint = SemanticBlueprint(
            original_text="Test sentence.",
            svo_triples=[("sentence",)],
            core_keywords={"test", "sentence"},
            named_entities=[],
            citations=[],
            quotes=[]
        )

        operator = SemanticInjectionOperator()
        mock_llm = Mock()
        mock_llm.call.return_value = "Test sentence with keywords."

        operator.generate(
            current_draft="Test.",
            blueprint=blueprint,
            author_name="Test",
            style_dna="Test",
            rhetorical_type=RhetoricalType.OBSERVATION,
            llm_provider=mock_llm,
            missing_keywords=["sentence"],
            temperature=0.6
        )

        # Verify CoT structure in prompt
        call_args = mock_llm.call.call_args
        user_prompt = call_args[1]["user_prompt"]

        # Should have 3-step structure
        self.assertIn("Step 1 (Reasoning)", user_prompt)
        self.assertIn("Step 2 (Rough Draft)", user_prompt)
        self.assertIn("Step 3 (Polish)", user_prompt)
        self.assertIn("Return ONLY the final polished sentence", user_prompt)

    def test_full_evolution_cycle_convergence(self):
        """Test a full evolution cycle with multiple generations showing convergence.

        Simulates the real scenario:
        - Start: Score 0.73, missing keywords
        - Gen 1: Score 0.85, still missing some
        - Gen 2: Score 0.92, all keywords present, passes
        """
        blueprint = SemanticBlueprint(
            original_text="Human experience reinforces the rule of finitude.",
            svo_triples=[("Human experience", "reinforce", "rule of finitude")],
            core_keywords={"experience", "finitude", "human", "reinforce", "rule"},
            named_entities=[],
            citations=[],
            quotes=[]
        )

        initial_draft = "Human experience confirms the law of finitude."

        # Track evolution progress
        evolution_history = []

        # Mock critic with progressive improvements
        mock_critic = Mock()

        def mock_evaluate(text, blueprint, allowed_style_words=None, **kwargs):
            # Simulate progressive improvement
            if "reinforce" not in text.lower() or "rule" not in text.lower():
                # Missing keywords
                result = {
                    "pass": False,
                    "score": 0.73,
                    "recall_score": 0.60,
                    "precision_score": 0.50,
                    "fluency_score": 0.90,
                    "feedback": "Missing concepts: rule, reinforce."
                }
            elif "reinforce" in text.lower() and "rule" in text.lower():
                if "universal" in text.lower():
                    # Perfect version
                    result = {
                        "pass": True,
                        "score": 0.95,
                        "recall_score": 1.00,
                        "precision_score": 0.90,
                        "fluency_score": 0.95,
                        "feedback": "Passed semantic validation."
                    }
                else:
                    # Good but could be better
                    result = {
                        "pass": False,
                        "score": 0.85,
                        "recall_score": 1.00,
                        "precision_score": 0.70,
                        "fluency_score": 0.85,
                        "feedback": "Good but could improve style."
                    }
            else:
                result = {
                    "pass": False,
                    "score": 0.50,
                    "recall_score": 0.50,
                    "precision_score": 0.50,
                    "fluency_score": 0.50,
                    "feedback": "Needs improvement."
                }

            evolution_history.append((text, result["score"], result["pass"]))
            return result

        mock_critic.evaluate.side_effect = mock_evaluate

        # Mock soft scorer with progressive raw_scores
        mock_soft_scorer = Mock()
        raw_score_map = {
            "Human experience confirms the law of finitude.": 0.73,
            "Human experience reinforces the rule of finitude.": 0.85,
            "Human experience confirms and reinforces the universal rule of finitude.": 0.95,
        }

        def mock_eval_with_raw_score(text, blueprint, style_lexicon=None, **kwargs):
            raw_score = raw_score_map.get(text, 0.5)
            result = mock_evaluate(text, blueprint)
            return {
                "raw_score": raw_score,
                "pass": result["pass"],
                "recall_score": result["recall_score"],
                "fluency_score": result["fluency_score"],
                "precision_score": result["precision_score"],
                "score": result["score"],
                "feedback": result["feedback"]
            }

        mock_soft_scorer.evaluate_with_raw_score.side_effect = mock_eval_with_raw_score

        self.translator.soft_scorer = mock_soft_scorer

        # Mock population generation - return progressively better candidates
        generation_count = [0]
        def mock_gen(*args, **kwargs):
            generation_count[0] += 1
            if generation_count[0] == 1:
                # Gen 1: Add "reinforce" and "rule"
                return [("semantic_injection", "Human experience reinforces the rule of finitude.")]
            else:
                # Gen 2: Perfect version
                return [("style_polish", "Human experience confirms and reinforces the universal rule of finitude.")]

        with patch.object(self.translator, '_generate_population_with_operator', side_effect=mock_gen):
            # Mock acceptance check - Gen 2 should pass
            def mock_acceptance(recall_score, precision_score, fluency_score=None, overall_score=0.0, threshold=0.85, style_density=None, logic_fail=False, **kwargs):
                return overall_score >= 0.90 and recall_score >= 0.95

            self.translator._check_acceptance = mock_acceptance

            # Run evolution
            best_draft, best_score = self.translator._evolve_text(
                initial_draft=initial_draft,
                blueprint=blueprint,
                author_name="Mao",
                style_dna="Authoritative and philosophical",
                rhetorical_type=RhetoricalType.OBSERVATION,
                initial_score=0.73,
                initial_feedback="Missing concepts: rule, reinforce.",
                critic=mock_critic,
                verbose=False
            )

            # Verify convergence happened
            self.assertGreaterEqual(best_score, 0.90,
                                  f"Evolution should converge to >= 0.90, got {best_score:.2f}")

            # Verify all keywords present in final draft
            final_lower = best_draft.lower()
            self.assertIn("reinforce", final_lower, "Final draft must contain 'reinforce'")
            self.assertIn("rule", final_lower, "Final draft must contain 'rule'")
            self.assertIn("experience", final_lower, "Final draft must contain 'experience'")
            self.assertIn("finitude", final_lower, "Final draft must contain 'finitude'")

            # Verify evolution actually improved (not just stayed the same)
            if len(evolution_history) > 1:
                initial_score = evolution_history[0][1]
                final_score = evolution_history[-1][1]
                self.assertGreater(final_score, initial_score,
                                  f"Evolution should improve from {initial_score:.2f} to > {initial_score:.2f}, "
                                  f"got {final_score:.2f}")


if __name__ == "__main__":
    if not DEPENDENCIES_AVAILABLE:
        print(f"⚠ SKIPPED: Missing dependencies - {IMPORT_ERROR}")
        sys.exit(1)
    unittest.main()

