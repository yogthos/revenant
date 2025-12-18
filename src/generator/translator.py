"""Style translator for Pipeline 2.0.

This module translates semantic blueprints into styled text using
few-shot examples from a rhetorically-indexed style atlas.
"""

import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from src.ingestion.blueprint import SemanticBlueprint
from src.atlas.rhetoric import RhetoricalType
from src.generator.llm_provider import LLMProvider
from src.generator.llm_interface import clean_generated_text
from src.critic.judge import LLMJudge
from src.critic.scorer import SoftScorer
from src.validator.semantic_critic import SemanticCritic
from src.generator.mutation_operators import (
    get_operator, OP_SEMANTIC_INJECTION, OP_GRAMMAR_REPAIR, OP_STYLE_POLISH, OP_DYNAMIC_STYLE, OP_STRUCTURAL_CLONE
)
from src.analyzer.structuralizer import Structuralizer


def _load_prompt_template(template_name: str) -> str:
    """Load a prompt template from the prompts directory.

    Args:
        template_name: Name of the template file (e.g., 'translator_system.md')

    Returns:
        Template content as string.
    """
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    template_path = prompts_dir / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return template_path.read_text().strip()


def _load_positional_instructions() -> Dict[str, str]:
    """Load positional instructions from prompt file.

    Returns:
        Dictionary mapping position names to instruction text.
    """
    content = _load_prompt_template("translator_positional_instructions.md")
    instructions = {}
    current_section = None
    current_text = []

    for line in content.split('\n'):
        if line.startswith('## '):
            if current_section:
                instructions[current_section] = '\n'.join(current_text).strip()
            current_section = line[3:].strip()
            current_text = []
        elif current_section:
            current_text.append(line)

    if current_section:
        instructions[current_section] = '\n'.join(current_text).strip()

    return instructions


class StyleTranslator:
    """Translates semantic blueprints into styled text using few-shot examples."""

    def __init__(self, config_path: str = "config.json"):
        """Initialize the translator.

        Args:
            config_path: Path to configuration file.
        """
        self.llm_provider = LLMProvider(config_path=config_path)
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.translator_config = self.config.get("translator", {})
        # Load positional instructions
        self.positional_instructions = _load_positional_instructions()
        # Cache spaCy model for blueprint completeness checks
        self._nlp_cache = None
        # Initialize soft scorer for fitness-based evolution
        self.soft_scorer = SoftScorer(config_path=config_path)
        # Initialize structuralizer for JIT structural templating
        self.structuralizer = Structuralizer(config_path=config_path)

    def _get_nlp(self):
        """Get or load spaCy model for noun extraction."""
        if self._nlp_cache is None:
            try:
                import spacy
                self._nlp_cache = spacy.load("en_core_web_sm")
            except (OSError, ImportError):
                # If spaCy not available, return None (check will be skipped)
                self._nlp_cache = False
        return self._nlp_cache if self._nlp_cache else None

    def _is_blueprint_incomplete(self, blueprint: SemanticBlueprint) -> bool:
        """Check if blueprint is semantically incomplete (missing critical nouns).

        Uses spaCy to extract and compare nouns from original_text vs blueprint.
        Uses lemmatization for matching (e.g., "Objects" matches "object").

        Args:
            blueprint: Semantic blueprint to check.

        Returns:
            True if blueprint is incomplete (missing > 50% of original nouns or empty SVO for long text).
        """
        nlp = self._get_nlp()
        if not nlp:
            # If spaCy not available, be conservative: assume complete
            return False

        if not blueprint.original_text or not blueprint.original_text.strip():
            return False

        # Extract nouns from original_text (lemmatized, non-stop words)
        original_doc = nlp(blueprint.original_text)
        original_nouns = set()
        for token in original_doc:
            if token.pos_ == "NOUN" and not token.is_stop:
                original_nouns.add(token.lemma_.lower())

        # If original has no nouns, can't determine completeness
        if not original_nouns:
            # Check if original has > 5 words but blueprint SVO is empty
            original_word_count = len(blueprint.original_text.split())
            if original_word_count > 5 and not blueprint.svo_triples:
                return True
            return False

        # Extract nouns from blueprint (subjects + objects, lemmatized)
        blueprint_nouns = set()

        # Extract from subjects
        for subject in blueprint.get_subjects():
            if subject:
                subj_doc = nlp(subject.lower())
                for token in subj_doc:
                    if token.pos_ == "NOUN" and not token.is_stop:
                        blueprint_nouns.add(token.lemma_.lower())

        # Extract from objects
        for obj in blueprint.get_objects():
            if obj:
                obj_doc = nlp(obj.lower())
                for token in obj_doc:
                    if token.pos_ == "NOUN" and not token.is_stop:
                        blueprint_nouns.add(token.lemma_.lower())

        # Check: if blueprint has zero matching nouns, it's incomplete
        matching_nouns = original_nouns.intersection(blueprint_nouns)
        if len(matching_nouns) == 0 and len(original_nouns) > 0:
            return True

        # Check: if blueprint has < 50% of original nouns, it's incomplete
        if len(original_nouns) > 0:
            match_ratio = len(matching_nouns) / len(original_nouns)
            if match_ratio < 0.5:
                return True

        # Also check: if original has > 5 words but blueprint SVO is empty
        original_word_count = len(blueprint.original_text.split())
        if original_word_count > 5 and not blueprint.svo_triples:
            return True

        return False

    def _extract_multiple_skeletons(
        self,
        examples: List[str],
        blueprint: SemanticBlueprint
    ) -> List[Tuple[str, str]]:
        """Extract skeletons from multiple examples with complexity filtering and adaptation.

        Args:
            examples: List of example sentences from ChromaDB.
            blueprint: Semantic blueprint with original text.

        Returns:
            List of (skeleton, source_example) tuples (top 3 surviving after filtering/adaptation).
        """
        if not examples:
            return []

        input_len = len(blueprint.original_text.split())
        compatible_skeletons = []

        for example in examples:
            try:
                # Pass input text for skeleton pruning (if single sentence, truncate to first sentence)
                skeleton = self.structuralizer.extract_skeleton(example, input_text=blueprint.original_text)
                if not skeleton:
                    continue

                skeleton_slots = self.structuralizer.count_skeleton_slots(skeleton)

                # Complexity gate: reject if mold is vastly too big or too small
                if 0.5 * input_len <= skeleton_slots <= 3.0 * input_len:
                    compatible_skeletons.append((skeleton, example))
                elif skeleton_slots > input_len * 2:
                    # Too long: try to compress it
                    adapted_skeleton = self.structuralizer.adapt_skeleton(skeleton, input_len)
                    adapted_slots = self.structuralizer.count_skeleton_slots(adapted_skeleton)
                    # Check if adaptation helped
                    if 0.5 * input_len <= adapted_slots <= 3.0 * input_len:
                        compatible_skeletons.append((adapted_skeleton, example))
                elif skeleton_slots < input_len * 0.5:
                    # Too short: try to expand it
                    adapted_skeleton = self.structuralizer.adapt_skeleton(skeleton, input_len)
                    adapted_slots = self.structuralizer.count_skeleton_slots(adapted_skeleton)
                    # Check if adaptation helped
                    if 0.5 * input_len <= adapted_slots <= 3.0 * input_len:
                        compatible_skeletons.append((adapted_skeleton, example))
            except Exception:
                # Skip if skeleton extraction/adaptation fails
                continue

        # Return top 3 surviving skeletons (or all if fewer than 3)
        return compatible_skeletons[:3]

    def _calculate_skeleton_adherence(self, candidate: str, skeleton: str) -> float:
        """Calculate skeleton adherence score using anchor word overlap.

        Extracts function words (anchor words) from skeleton and checks
        what percentage appear in candidate in roughly the same order.

        Args:
            candidate: Generated candidate text.
            skeleton: Skeleton template with placeholders.

        Returns:
            Adherence score 0.0-1.0 (matched_anchors / total_anchors).
        """
        if not candidate or not skeleton:
            return 0.0

        # Function words (anchor words) that should be preserved
        # These are structural words that appear in the skeleton
        function_words = {
            'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by',
            'and', 'or', 'but', 'if', 'then', 'when', 'where', 'while', 'as',
            'from', 'into', 'onto', 'upon', 'over', 'under', 'above', 'below',
            'through', 'during', 'before', 'after', 'since', 'until', 'about',
            'against', 'between', 'among', 'within', 'without', 'across',
            'that', 'this', 'these', 'those', 'which', 'who', 'whom', 'whose',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'must', 'can', 'not', 'no', 'nor', 'so', 'than', 'too', 'very', 'more',
            'most', 'some', 'any', 'all', 'each', 'every', 'both', 'either', 'neither'
        }

        # Extract anchor words from skeleton (non-placeholder words)
        # First, remove placeholders [NP], [VP], [ADJ] from skeleton
        skeleton_clean = re.sub(r'\[NP\]|\[VP\]|\[ADJ\]', '', skeleton, flags=re.IGNORECASE)
        skeleton_tokens = re.findall(r'\b\w+\b', skeleton_clean.lower())
        anchor_words = []
        anchor_positions = []

        for i, token in enumerate(skeleton_tokens):
            # Include function words
            if token in function_words:
                anchor_words.append(token)
                anchor_positions.append(i)

        if not anchor_words:
            # No anchor words found, return neutral score
            return 0.5

        # Extract tokens from candidate
        candidate_tokens = re.findall(r'\b\w+\b', candidate.lower())

        # Count matched anchors (with position tolerance ±2)
        matched_count = 0
        candidate_idx = 0

        for anchor_idx, anchor_word in enumerate(anchor_words):
            # Search for anchor word in candidate within position tolerance
            skeleton_pos = anchor_positions[anchor_idx]
            search_start = max(0, candidate_idx - 2)
            search_end = min(len(candidate_tokens), candidate_idx + 10)

            for i in range(search_start, search_end):
                if candidate_tokens[i] == anchor_word:
                    matched_count += 1
                    candidate_idx = i + 1  # Move forward
                    break

        adherence_score = matched_count / len(anchor_words) if anchor_words else 0.0
        return adherence_score

    def _calculate_style_density(self, candidate: str, style_lexicon: Optional[List[str]]) -> float:
        """Calculate style density score (ratio of style lexicon words in candidate).

        Args:
            candidate: Generated candidate text.
            style_lexicon: Optional list of style words to check for.

        Returns:
            Style density score 0.0-1.0 (count(output_words in lexicon) / total_words).
        """
        if not candidate or not style_lexicon:
            return 0.0

        # Tokenize candidate
        candidate_tokens = re.findall(r'\b\w+\b', candidate.lower())
        if not candidate_tokens:
            return 0.0

        # Create set of style lexicon words (lowercase)
        style_words_set = {word.lower().strip() for word in style_lexicon}

        # Count style words in candidate
        style_word_count = sum(1 for token in candidate_tokens if token in style_words_set)

        style_density = style_word_count / len(candidate_tokens)
        return style_density

    def _evaluate_template_candidate(
        self,
        candidate: str,
        blueprint: SemanticBlueprint,
        skeleton: str,
        style_dna: Optional[Dict[str, any]] = None
    ) -> Dict[str, any]:
        """Evaluate template candidate using composite scoring.

        Calculates semantic score, structural adherence score, and style density score.
        Applies hard gates (reject if semantic < 0.7 or adherence < 0.7).

        Args:
            candidate: Generated candidate text.
            blueprint: Original semantic blueprint.
            skeleton: Skeleton template used for generation.
            style_dna: Optional style DNA dictionary with lexicon.

        Returns:
            Dictionary with scores and pass status:
            {
                "semantic_score": float,
                "adherence_score": float,
                "style_density": float,
                "composite_score": float,
                "passed_gates": bool
            }
        """
        # GHOSTBUSTER: Reject candidates containing skeleton placeholder brackets
        # These are generation failures (raw skeletons), not valid candidates
        # Note: Citation references like [^155] are allowed - only reject [NP], [VP], [ADJ] placeholders
        skeleton_placeholder_pattern = r'\[(?:NP|VP|ADJ)\]'
        if re.search(skeleton_placeholder_pattern, candidate, re.IGNORECASE):
            return {
                "semantic_score": 0.0,
                "adherence_score": 0.0,
                "style_density": 0.0,
                "composite_score": 0.0,
                "passed_gates": False,
                "rejection_reason": "Candidate contains skeleton placeholder brackets ([NP], [VP], [ADJ]) - generation failure"
            }

        from src.validator.semantic_critic import SemanticCritic

        # Load weights from config
        weights = self.config.get("weights", {})
        semantic_weight = weights.get("semantic", 0.4)
        adherence_weight = weights.get("structure_adherence", 0.3)
        style_weight = weights.get("style_density", 0.3)

        # Load thresholds from config
        template_config = self.translator_config.get("template_evolution", {})
        semantic_threshold = template_config.get("semantic_threshold", 0.7)
        adherence_threshold = template_config.get("adherence_threshold", 0.7)

        # Extract style lexicon
        style_lexicon = None
        if style_dna and isinstance(style_dna, dict):
            style_lexicon = style_dna.get("lexicon")

        # 1. Semantic Score (40% weight)
        critic = SemanticCritic(config_path=self.config_path)
        semantic_result = self.soft_scorer.evaluate_with_raw_score(
            candidate, blueprint, style_lexicon=style_lexicon, skeleton=skeleton
        )
        semantic_score = semantic_result.get("raw_score", semantic_result.get("score", 0.0))

        # 2. Structural Adherence Score (30% weight)
        adherence_score = self._calculate_skeleton_adherence(candidate, skeleton)

        # 3. Style Density Score (30% weight)
        style_density = self._calculate_style_density(candidate, style_lexicon)

        # 4. Expansion Ratio Gate (Dynamic): Reject excessive expansion
        input_words = blueprint.original_text.split()
        candidate_words = candidate.split()
        input_len = len(input_words)
        expansion_ratio = len(candidate_words) / max(1, input_len)

        # Dynamic expansion thresholds: Allow more expansion for short inputs
        # Short sentences need more room to breathe when transformed to complex styles (e.g., Maoist dialectic)
        if input_len < 10:
            max_ratio = 6.0  # Allow 6x expansion for very short inputs (e.g., 7 words -> 42 words)
        elif input_len < 20:
            max_ratio = 4.0  # Allow 4x expansion for short inputs
        else:
            max_ratio = 2.5  # Allow 2.5x expansion for longer inputs

        # Hard rejection: If ratio > max_ratio, reject immediately
        # Exception: Allow if style_density > 0.4 (very high style usage indicates intentional expansion)
        if expansion_ratio > max_ratio and style_density <= 0.4:
            return {
                "semantic_score": semantic_score,
                "adherence_score": adherence_score,
                "style_density": style_density,
                "composite_score": 0.0,
                "passed_gates": False,
                "rejection_reason": f"Candidate rejected due to excessive expansion (Ratio: {expansion_ratio:.2f} > {max_ratio:.1f} for {input_len}-word input)."
            }

        # Hard gates: reject if semantic or adherence below threshold
        passed_gates = (semantic_score >= semantic_threshold and
                       adherence_score >= adherence_threshold)

        # Calculate composite score
        if not passed_gates:
            composite_score = 0.0
        else:
            composite_score = (
                semantic_score * semantic_weight +
                adherence_score * adherence_weight +
                style_density * style_weight
            )

        return {
            "semantic_score": semantic_score,
            "adherence_score": adherence_score,
            "style_density": style_density,
            "composite_score": composite_score,
            "passed_gates": passed_gates
        }

    def translate(
        self,
        blueprint: SemanticBlueprint,
        author_name: str,
        style_dna: str,
        rhetorical_type: RhetoricalType,
        examples: List[str],  # 3 examples from atlas
        verbose: bool = False
    ) -> str:
        """Translate blueprint into styled text.

        Args:
            blueprint: Semantic blueprint to translate.
            author_name: Target author name.
            style_dna: Style DNA description.
            rhetorical_type: Rhetorical mode.
            examples: Few-shot examples from atlas.

        Returns:
            Generated text in target style.
        """
        if not examples:
            # Fallback if no examples provided
            examples = ["Example text in the target style."]

        # Extract style_dna_dict from examples if available
        style_dna_dict = None
        try:
            from src.analyzer.style_extractor import StyleExtractor
            style_extractor = StyleExtractor(config_path=self.config_path)
            if examples:
                style_dna_dict = style_extractor.extract_style_dna(examples)
        except Exception:
            style_dna_dict = None

        # Phase 1: Smart Skeleton Selection (Pre-Filter)
        template_config = self.translator_config.get("template_evolution", {})
        num_examples = template_config.get("num_examples", 5)
        candidates_per_template = template_config.get("candidates_per_template", 2)

        # Retrieve top N examples (request more to have options after filtering)
        if len(examples) < num_examples:
            # Use all available examples
            candidate_examples = examples
        else:
            # Use top N examples
            candidate_examples = examples[:num_examples]

        if verbose:
            print(f"  Parallel Template Evolution: Processing {len(candidate_examples)} examples")

        # Extract multiple skeletons with complexity filtering
        compatible_skeletons = self._extract_multiple_skeletons(candidate_examples, blueprint)

        if verbose:
            print(f"  Skeletons extracted: {len(compatible_skeletons)} compatible (after complexity filtering)")

        if not compatible_skeletons:
            if verbose:
                print(f"  No compatible skeletons found, falling back to standard generation")
        else:
            # Phase 2: Style-Infused Generation
            all_candidates = []
            structural_operator = get_operator(OP_STRUCTURAL_CLONE)

            # Extract style lexicon from style_dna_dict
            style_lexicon = None
            if style_dna_dict and isinstance(style_dna_dict, dict):
                style_lexicon = style_dna_dict.get("lexicon")

            for skeleton, source_example in compatible_skeletons:
                if verbose:
                    print(f"  Generating {candidates_per_template} candidates for skeleton: {skeleton[:60]}...")

                # Generate multiple candidates per skeleton with temperature variation
                temperatures = [0.7, 0.75, 0.8][:candidates_per_template]
                for temp_idx, temp in enumerate(temperatures):
                    try:
                        candidate = structural_operator.generate(
                            current_draft="",
                            blueprint=blueprint,
                            author_name=author_name,
                            style_dna=style_dna,
                            rhetorical_type=rhetorical_type,
                            llm_provider=self.llm_provider,
                            temperature=temp,
                            max_tokens=self.translator_config.get("max_tokens", 300),
                            skeleton=skeleton,
                            style_lexicon=style_lexicon
                        )

                        if candidate and candidate.strip():
                            # ESCAPE HATCH: Check for logic mismatch signal
                            if "SKIPPING: LOGIC_MISMATCH" in candidate.upper() or "LOGIC_MISMATCH" in candidate.upper():
                                if verbose:
                                    print(f"    ⚠ Logic mismatch detected for skeleton, skipping this template")
                                # Skip this candidate and continue to next skeleton
                                continue

                            # EXPANSION GATE: Reject candidates where word_count > 3 * input_word_count
                            input_word_count = len(blueprint.original_text.split())
                            candidate_word_count = len(candidate.split())
                            if candidate_word_count > 3 * input_word_count:
                                if verbose:
                                    ratio = candidate_word_count / max(1, input_word_count)
                                    print(f"    ✗ Candidate rejected: excessive expansion (Ratio: {ratio:.2f}, {candidate_word_count} words vs {input_word_count} input words)")
                                continue

                            all_candidates.append({
                                "text": candidate,
                                "skeleton": skeleton,
                                "source_example": source_example
                            })
                    except Exception as e:
                        if verbose:
                            print(f"    ✗ Candidate generation failed: {e}")
                        continue

            if verbose:
                print(f"  Total candidates generated: {len(all_candidates)}")

            # Phase 3: Composite Evaluation
            evaluated_candidates = []
            for candidate_data in all_candidates:
                try:
                    evaluation = self._evaluate_template_candidate(
                        candidate=candidate_data["text"],
                        blueprint=blueprint,
                        skeleton=candidate_data["skeleton"],
                        style_dna=style_dna_dict
                    )

                    # Check if this is a rescue candidate (high adherence but low semantic)
                    needs_repair = False
                    if evaluation.get("adherence_score", 0.0) > 0.9 and evaluation.get("semantic_score", 0.0) < 0.6:
                        needs_repair = True
                        if verbose:
                            print(f"    ⚠ Rescue candidate detected (Adherence: {evaluation.get('adherence_score', 0.0):.2f}, Semantic: {evaluation.get('semantic_score', 0.0):.2f}) - marking for repair")

                    evaluated_candidates.append({
                        "text": candidate_data["text"],
                        "skeleton": candidate_data["skeleton"],
                        "source_example": candidate_data["source_example"],
                        "evaluation": evaluation,
                        "needs_repair": needs_repair
                    })

                    if verbose:
                        eval_data = evaluation
                        print(f"  Candidate: {candidate_data['text'][:60]}...")
                        print(f"    Semantic: {eval_data['semantic_score']:.2f}, "
                              f"Adherence: {eval_data['adherence_score']:.2f}, "
                              f"Style: {eval_data['style_density']:.2f}, "
                              f"Composite: {eval_data['composite_score']:.2f}, "
                              f"Passed: {eval_data['passed_gates']}")
                except Exception as e:
                    if verbose:
                        print(f"  ✗ Evaluation failed for candidate: {e}")
                    continue

            # Phase 4: Selection
            # Filter candidates that passed hard gates OR are rescue candidates
            passed_candidates = [c for c in evaluated_candidates if c["evaluation"]["passed_gates"]]
            rescue_candidates = [c for c in evaluated_candidates if c.get("needs_repair", False) and not c["evaluation"]["passed_gates"]]

            if passed_candidates:
                # Sort by composite score (descending)
                passed_candidates.sort(key=lambda x: x["evaluation"]["composite_score"], reverse=True)
                best_candidate = passed_candidates[0]

                if verbose:
                    print(f"  ✓ Best candidate selected (composite score: {best_candidate['evaluation']['composite_score']:.2f})")
                    print(f"    Top 3 candidates:")
                    for i, cand in enumerate(passed_candidates[:3]):
                        print(f"      {i+1}. Score: {cand['evaluation']['composite_score']:.2f} - {cand['text'][:60]}...")

                # Restore citations and quotes
                best_text = self._restore_citations_and_quotes(best_candidate["text"], blueprint)
                return best_text
            elif rescue_candidates:
                # Select best rescue candidate (highest adherence score)
                rescue_candidates.sort(key=lambda x: x["evaluation"].get("adherence_score", 0.0), reverse=True)
                best_rescue = rescue_candidates[0]

                if verbose:
                    print(f"  ⚠ No passed candidates, but found {len(rescue_candidates)} rescue candidate(s)")
                    print(f"    Selected rescue candidate (Adherence: {best_rescue['evaluation'].get('adherence_score', 0.0):.2f}, Semantic: {best_rescue['evaluation'].get('semantic_score', 0.0):.2f})")
                    print(f"    Text: {best_rescue['text'][:80]}...")
                    print(f"    → Passing to evolution loop with semantic injection priority")

                # Restore citations and quotes
                rescue_text = self._restore_citations_and_quotes(best_rescue["text"], blueprint)

                # Pass to evolution loop with semantic injection priority
                # Evaluate the rescue candidate to get initial score
                critic = SemanticCritic(config_path=self.config_path)
                initial_result = critic.evaluate(rescue_text, blueprint, skeleton=best_rescue["skeleton"])
                initial_score = initial_result.get("score", 0.0)
                initial_feedback = initial_result.get("feedback", "")

                # Evolve with semantic injection priority
                evolved_text, evolved_score = self._evolve_text(
                    initial_draft=rescue_text,
                    blueprint=blueprint,
                    author_name=author_name,
                    style_dna=style_dna,
                    rhetorical_type=rhetorical_type,
                    initial_score=initial_score,
                    initial_feedback=initial_feedback,
                    critic=critic,
                    verbose=verbose,
                    style_dna_dict=style_dna_dict,
                    examples=examples,
                    force_semantic_injection=True  # Flag to prioritize semantic injection
                )

                return evolved_text
            else:
                if verbose:
                    print(f"  ✗ All candidates failed hard gates, falling back to standard generation")

        # Step 2: Standard generation (fallback if structural cloning fails or not applicable)
        # Style-Infused Fallback: Ensure we use style DNA even in fallback
        prompt = self._build_prompt(blueprint, author_name, style_dna, rhetorical_type, examples)

        # Extract style lexicon for prompt injection
        style_lexicon_text = ""
        if style_dna_dict and isinstance(style_dna_dict, dict):
            style_lexicon = style_dna_dict.get("lexicon", [])
            if style_lexicon:
                lexicon_preview = ", ".join(style_lexicon[:15])  # Show first 15 words
                style_lexicon_text = f"\n\n**STYLE VOCABULARY (Use these words):** {lexicon_preview}"

        # Build examples text for style injection
        examples_text = ""
        if examples:
            examples_preview = "\n".join([f"- \"{ex}\"" for ex in examples[:3]])
            examples_text = f"\n\n**STYLE EXAMPLES (Match this voice):**\n{examples_preview}"

        # Inject style guidance into prompt
        if style_lexicon_text or examples_text:
            style_injection = f"""
=== STYLE REQUIREMENT (CRITICAL - DO NOT BE BORING) ===
You are writing in the style of these examples. Do NOT write generic corporate English.{examples_text}{style_lexicon_text}

**CRITICAL:** Transform the sentence to match the author's distinctive voice. Use the vocabulary provided. Be distinct, not generic.
=====================================================
"""
            prompt = style_injection + prompt

        system_prompt_template = _load_prompt_template("translator_system.md")
        system_prompt = system_prompt_template.format(
            author_name=author_name,
            style_dna=style_dna
        )

        try:
            # Phase 2: Two-Pass Pipeline
            # Pass 1: Draft generation (high temperature for meaning preservation)
            draft_temperature = self.translator_config.get("draft_temperature", 0.75)
            draft = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=draft_temperature,
                max_tokens=self.translator_config.get("max_tokens", 300)
            )
            draft = clean_generated_text(draft)
            draft = draft.strip()

            # Pass 2: Polish for natural English (low temperature for refinement)
            polished = self._polish_draft(draft, blueprint, author_name, style_dna)

            # Restore citations and quotes if missing
            polished = self._restore_citations_and_quotes(polished, blueprint)
            return polished
        except Exception as e:
            # Fallback on error
            return self.translate_literal(blueprint, author_name, style_dna)

    def _build_prompt(
        self,
        blueprint: SemanticBlueprint,
        author_name: str,
        style_dna: str,
        rhetorical_type: RhetoricalType,
        examples: List[str]
    ) -> str:
        """Build few-shot prompt with contextual anchoring.

        Args:
            blueprint: Semantic blueprint (with positional metadata).
            author_name: Target author name.
            style_dna: Style DNA description.
            rhetorical_type: Rhetorical mode.
            examples: Few-shot examples.

        Returns:
            Formatted prompt string with position instructions and context.
        """
        # Check if blueprint is semantically complete
        if self._is_blueprint_incomplete(blueprint):
            # Use original-text-only prompt (no blueprint structure)
            return self._build_original_text_only_prompt(blueprint, author_name, style_dna, rhetorical_type, examples)

        examples_text = "\n".join([f"Example {i+1}: \"{ex}\"" for i, ex in enumerate(examples)])

        # Get position-specific instruction
        pos_instruction = self.positional_instructions.get(
            blueprint.position,
            self.positional_instructions.get("BODY", "")
        )

        # Build context block (only for BODY/CLOSER positions)
        context_block = ""
        if blueprint.position in ["BODY", "CLOSER"] and blueprint.previous_context:
            context_block = f"""
=== PREVIOUS CONTEXT (The sentence you just wrote) ===
"{blueprint.previous_context}"
(Your rewriting MUST logically follow this sentence.)
======================================================
"""

        # FAILSAFE: If blueprint is empty, use original text directly
        if not blueprint.svo_triples and not blueprint.core_keywords:
            # Build citations and quotes sections even for empty blueprint
            citations_text = ""
            if blueprint.citations:
                citation_list = [cit[0] for cit in blueprint.citations]
                citations_text = f"- Citations to preserve: {', '.join(citation_list)}"

            quotes_text = ""
            if blueprint.quotes:
                quote_list = [quote[0] for quote in blueprint.quotes]
                quotes_text = f"- Direct quotes to preserve exactly: {', '.join(quote_list)}"

            preservation_section = ""
            if citations_text or quotes_text:
                preservation_section = f"""
=== CRITICAL PRESERVATION REQUIREMENTS (NON-NEGOTIABLE) ===
{citations_text}
{quotes_text}

CRITICAL RULES:
- ALL citations MUST use EXACTLY the [^number] format shown above (e.g., [^155])
- DO NOT use (Author, Year), (Smith 42), or any other citation formats - ONLY [^number] format is allowed
- ALL citations [^number] MUST be included in your output (you may place them at the end of the sentence if needed)
- ALL direct quotes MUST be preserved EXACTLY word-for-word as shown above
- DO NOT modify, paraphrase, or change any quoted text
- DO NOT remove or relocate citations"""

            template = _load_prompt_template("translator_user_empty_blueprint_template.md")
            return template.format(
                rhetorical_type=rhetorical_type.value,
                context_block=context_block,
                examples_text=examples_text,
                original_text=blueprint.original_text,
                preservation_section=preservation_section,
                pos_instruction=pos_instruction
            )

        # Normal blueprint path
        subjects = blueprint.get_subjects()
        verbs = blueprint.get_verbs()
        objects = blueprint.get_objects()
        entities = blueprint.named_entities
        keywords = sorted(blueprint.core_keywords)

        entities_text = ', '.join([f"{ent[0]} ({ent[1]})" for ent in entities]) if entities else "None"

        # Build citations and quotes sections
        citations_text = ""
        if blueprint.citations:
            citation_list = [cit[0] for cit in blueprint.citations]
            citations_text = f"- Citations to preserve: {', '.join(citation_list)}"

        quotes_text = ""
        if blueprint.quotes:
            quote_list = [quote[0] for quote in blueprint.quotes]
            quotes_text = f"- Direct quotes to preserve exactly: {', '.join(quote_list)}"

        preservation_section = ""
        if citations_text or quotes_text:
            preservation_section = f"""
=== CRITICAL PRESERVATION REQUIREMENTS (NON-NEGOTIABLE) ===
{citations_text}
{quotes_text}

CRITICAL RULES:
- ALL citations MUST use EXACTLY the [^number] format shown above (e.g., [^155])
- DO NOT use (Author, Year), (Smith 42), or any other citation formats - ONLY [^number] format is allowed
- ALL citations [^number] MUST be included in your output (you may place them at the end of the sentence if needed)
- ALL direct quotes MUST be preserved EXACTLY word-for-word as shown above
- DO NOT modify, paraphrase, or change any quoted text
- DO NOT remove or relocate citations"""

        template = _load_prompt_template("translator_user_template.md")
        return template.format(
            rhetorical_type=rhetorical_type.value,
            context_block=context_block,
            examples_text=examples_text,
            original_text=blueprint.original_text,
            subjects=', '.join(subjects) if subjects else 'None',
            verbs=', '.join(verbs) if verbs else 'None',
            objects=', '.join(objects) if objects else 'None',
            entities=entities_text,
            keywords=', '.join(keywords) if keywords else 'None',
            preservation_section=preservation_section,
            pos_instruction=pos_instruction
        )

    def _build_original_text_only_prompt(
        self,
        blueprint: SemanticBlueprint,
        author_name: str,
        style_dna: str,
        rhetorical_type: RhetoricalType,
        examples: List[str]
    ) -> str:
        """Build prompt using only original text (no blueprint structure).

        Used when blueprint is semantically incomplete to avoid generating
        broken sentences from incomplete blueprints.

        Args:
            blueprint: Semantic blueprint (with original_text).
            author_name: Target author name.
            style_dna: Style DNA description.
            rhetorical_type: Rhetorical mode.
            examples: Few-shot examples.

        Returns:
            Formatted prompt string using original text only.
        """
        examples_text = "\n".join([f"Example {i+1}: \"{ex}\"" for i, ex in enumerate(examples)])

        # Get position-specific instruction
        pos_instruction = self.positional_instructions.get(
            blueprint.position,
            self.positional_instructions.get("BODY", "")
        )

        # Build context block (only for BODY/CLOSER positions)
        context_block = ""
        if blueprint.position in ["BODY", "CLOSER"] and blueprint.previous_context:
            context_block = f"""
=== PREVIOUS CONTEXT (The sentence you just wrote) ===
"{blueprint.previous_context}"
(Your rewriting MUST logically follow this sentence.)
======================================================
"""

        # Build citations and quotes sections
        citations_text = ""
        if blueprint.citations:
            citation_list = [cit[0] for cit in blueprint.citations]
            citations_text = f"- Citations to preserve: {', '.join(citation_list)}"

        quotes_text = ""
        if blueprint.quotes:
            quote_list = [quote[0] for quote in blueprint.quotes]
            quotes_text = f"- Direct quotes to preserve exactly: {', '.join(quote_list)}"

        preservation_section = ""
        if citations_text or quotes_text:
            preservation_section = f"""
=== CRITICAL PRESERVATION REQUIREMENTS (NON-NEGOTIABLE) ===
{citations_text}
{quotes_text}

CRITICAL RULES:
- ALL citations MUST use EXACTLY the [^number] format shown above (e.g., [^155])
- DO NOT use (Author, Year), (Smith 42), or any other citation formats - ONLY [^number] format is allowed
- ALL citations [^number] MUST be included in your output (you may place them at the end of the sentence if needed)
- ALL direct quotes MUST be preserved EXACTLY word-for-word as shown above
- DO NOT modify, paraphrase, or change any quoted text
- DO NOT remove or relocate citations"""

        template = _load_prompt_template("translator_user_original_only_template.md")
        return template.format(
            rhetorical_type=rhetorical_type.value,
            context_block=context_block,
            examples_text=examples_text,
            original_text=blueprint.original_text,
            preservation_section=preservation_section,
            pos_instruction=pos_instruction
        )

    def _polish_draft(
        self,
        draft: str,
        blueprint: SemanticBlueprint,
        author_name: str,
        style_dna: str
    ) -> str:
        """Polish a draft sentence for natural English flow.

        This is the second pass in the two-pass pipeline. Takes a draft that
        preserves meaning and refines it for natural English, fixing passive
        voice and stilted phrasing.

        Args:
            draft: Draft sentence to polish.
            blueprint: Original semantic blueprint (for reference).
            author_name: Target author name.
            style_dna: Style DNA description.

        Returns:
            Polished sentence in natural English.
        """
        if not draft or not draft.strip():
            return draft

        polish_template = _load_prompt_template("translator_polish.md")
        polish_prompt = polish_template.format(
            draft_text=draft,
            original_text=blueprint.original_text
        )

        system_prompt_template = _load_prompt_template("translator_system.md")
        system_prompt = system_prompt_template.format(
            author_name=author_name,
            style_dna=style_dna
        )

        try:
            polished = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=polish_prompt,
                temperature=self.translator_config.get("polish_temperature", 0.25),
                max_tokens=self.translator_config.get("max_tokens", 300)
            )
            polished = clean_generated_text(polished)
            polished = polished.strip()

            # If polish fails or returns empty, return original draft
            if not polished:
                return draft

            return polished
        except Exception as e:
            # If polish fails, return original draft
            return draft

    def translate_literal(
        self,
        blueprint: SemanticBlueprint,
        author_name: str,
        style_dna: str,
        rhetorical_type: Optional[RhetoricalType] = None,
        examples: Optional[List[str]] = None
    ) -> str:
        """Style-preserving fallback (loose style transfer when blueprint constraints fail).

        Args:
            blueprint: Semantic blueprint.
            author_name: Target author name.
            style_dna: Style DNA description.
            rhetorical_type: Optional rhetorical mode (for style-preserving fallback).
            examples: Optional few-shot examples (for style-preserving fallback).

        Returns:
            Style-transferred text (never returns original text verbatim).
        """
        # If blueprint is incomplete, use style-preserving fallback
        if self._is_blueprint_incomplete(blueprint):
            return self._translate_style_fallback(blueprint, author_name, style_dna, rhetorical_type, examples)

        # FAILSAFE: If blueprint is empty, use style-preserving fallback
        if not blueprint.svo_triples and not blueprint.core_keywords:
            return self._translate_style_fallback(blueprint, author_name, style_dna, rhetorical_type, examples)

        # Normal blueprint path - use literal translation template
        subjects = blueprint.get_subjects()
        verbs = blueprint.get_verbs()
        objects = blueprint.get_objects()

        # Build citations and quotes sections for literal translation too
        citations_text = ""
        if blueprint.citations:
            citation_list = [cit[0] for cit in blueprint.citations]
            citations_text = f"\nCitations to include: {', '.join(citation_list)}"

        quotes_text = ""
        if blueprint.quotes:
            quote_list = [quote[0] for quote in blueprint.quotes]
            quotes_text = f"\nQuotes to preserve exactly: {', '.join(quote_list)}"

        prompt_template = _load_prompt_template("translator_literal_user.md")
        # CRITICAL: Include original_text so LLM can see full content even if blueprint is incomplete
        prompt = prompt_template.format(
            original_text=blueprint.original_text,
            subjects=', '.join(subjects) if subjects else 'None',
            verbs=', '.join(verbs) if verbs else 'None',
            objects=', '.join(objects) if objects else 'None',
            citations_text=citations_text,
            quotes_text=quotes_text
        )

        system_prompt_template = _load_prompt_template("translator_literal_system.md")
        system_prompt = system_prompt_template.format(author_name=author_name)

        try:
            generated = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=self.translator_config.get("literal_temperature", 0.3),
                max_tokens=self.translator_config.get("literal_max_tokens", 200)
            )
            generated = clean_generated_text(generated)
            generated = generated.strip()
            # Restore citations and quotes if missing
            generated = self._restore_citations_and_quotes(generated, blueprint)
            return generated
        except Exception:
            # If normal path fails, use style-preserving fallback
            return self._translate_style_fallback(blueprint, author_name, style_dna, rhetorical_type, examples)

    def _translate_style_fallback(
        self,
        blueprint: SemanticBlueprint,
        author_name: str,
        style_dna: str,
        rhetorical_type: Optional[RhetoricalType] = None,
        examples: Optional[List[str]] = None
    ) -> str:
        """Style-preserving fallback: do style transfer without structural constraints.

        This is the "Hail Mary" attempt when blueprint is incomplete or normal translation fails.
        It still does style transfer, just without strict blueprint constraints.

        Args:
            blueprint: Semantic blueprint (with original_text).
            author_name: Target author name.
            style_dna: Style DNA description.
            rhetorical_type: Optional rhetorical mode.
            examples: Optional few-shot examples.

        Returns:
            Style-transferred text.
        """
        # Use OBSERVATION as default if not provided
        if rhetorical_type is None:
            rhetorical_type = RhetoricalType.OBSERVATION

        # Use empty examples if not provided
        if examples is None:
            examples = []

        # Build prompt using original-text-only template
        prompt = self._build_original_text_only_prompt(
            blueprint=blueprint,
            author_name=author_name,
            style_dna=style_dna,
            rhetorical_type=rhetorical_type,
            examples=examples
        )

        # Load system prompt
        system_prompt_template = _load_prompt_template("translator_system.md")
        system_prompt = system_prompt_template.format(
            author_name=author_name,
            style_dna=style_dna
        )

        try:
            generated = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=self.translator_config.get("literal_temperature", 0.5),  # Slightly higher for style
                max_tokens=self.translator_config.get("literal_max_tokens", 200)
            )
            generated = clean_generated_text(generated)
            generated = generated.strip()
            # Restore citations and quotes if missing
            generated = self._restore_citations_and_quotes(generated, blueprint)
            return generated
        except Exception:
            # Ultimate fallback: still try to do style transfer with minimal prompt
            # This should never happen, but if it does, at least attempt transformation
            minimal_prompt = f"""Rewrite this text in the style of {author_name}: "{blueprint.original_text}"

Style: {style_dna}

Do NOT copy the text verbatim. Transform it into the target style while preserving meaning."""
            try:
                generated = self.llm_provider.call(
                    system_prompt=f"You are {author_name}.",
                    user_prompt=minimal_prompt,
                    temperature=0.5,
                    max_tokens=200
                )
                generated = clean_generated_text(generated)
                generated = generated.strip()
                return generated
            except Exception:
                # Only return original if ALL attempts fail
                return blueprint.original_text

    def _check_acceptance(
        self,
        recall_score: float,
        precision_score: float,
        fluency_score: Optional[float] = None,
        overall_score: float = 0.0,
        pass_threshold: float = 0.9,
        original_text: str = "",
        generated_text: str = ""
    ) -> bool:
        """Check if draft should be accepted using Fluency Forgiveness logic.

        HARD GATE: Length heuristic - reject if input > 6 words and output < 4 words.
        HARD GATE: Fluency must be above minimum threshold (if provided).
        RULE 1: Recall is King. If we miss keywords, we fail.
        RULE 2: Precision is Flexible. If Recall is perfect, we can accept lower precision.
        Fallback: High overall score.

        Args:
            recall_score: Recall score (0-1)
            precision_score: Precision score (0-1)
            fluency_score: Optional fluency score (0-1). If None, fluency check is skipped.
            overall_score: Weighted overall score (0-1)
            pass_threshold: Default pass threshold
            original_text: Original input text (for length check)
            generated_text: Generated text (for length check)

        Returns:
            True if draft should be accepted, False otherwise
        """
        # HARD GATE: Length heuristic - reject if input > 6 words and output < 4 words
        # This catches "We touch breaks" type garbage output
        if original_text and generated_text:
            input_word_count = len(original_text.split())
            output_word_count = len(generated_text.split())
            if input_word_count > 6 and output_word_count < 4:
                return False  # Too short to be a valid translation

        # HARD GATE: Fluency must be above minimum (if fluency_score is provided)
        # Note: Fluency check removed from SemanticCritic in two-gate simplification
        # This check is kept for backward compatibility but defaults to passing if not provided
        if fluency_score is not None and fluency_score < 0.7:
            return False  # HARD REJECT regardless of other scores

        # RULE 1: Recall is King. If we miss keywords, we fail.
        if recall_score < 1.0:
            # Must have all keywords - use strict threshold
            return overall_score >= pass_threshold

        # RULE 2: Precision is Flexible.
        # If Recall is perfect, we can accept lower precision (fluency glue).
        if precision_score >= 0.80:
            return True

        # Fallback: High overall score
        return overall_score >= pass_threshold

    def _get_blueprint_text(self, blueprint: SemanticBlueprint) -> str:
        """Get text representation of blueprint for refinement prompt.

        Returns a concise summary of blueprint content.

        Args:
            blueprint: Semantic blueprint to extract text from.

        Returns:
            String representation of blueprint content.
        """
        if not blueprint.svo_triples and not blueprint.core_keywords:
            return blueprint.original_text

        subjects = blueprint.get_subjects()
        verbs = blueprint.get_verbs()
        objects = blueprint.get_objects()

        parts = []
        if subjects:
            parts.append(f"Subjects: {', '.join(subjects)}")
        if verbs:
            parts.append(f"Actions: {', '.join(verbs)}")
        if objects:
            parts.append(f"Objects: {', '.join(objects)}")

        return " | ".join(parts) if parts else blueprint.original_text

    def _generate_simplification(
        self,
        best_draft: str,
        blueprint: SemanticBlueprint,
        author_name: str,
        style_dna: str,
        rhetorical_type: RhetoricalType,
        critic: 'SemanticCritic',
        verbose: bool = False
    ) -> str:
        """Generate a simplified version when stuck at low scores.

        This is a "Hail Mary" attempt that strips the sentence down to basics
        when the evolution loop has stagnated at a low score.

        Args:
            best_draft: Current best draft (may be ignored in favor of blueprint)
            blueprint: Original semantic blueprint
            author_name: Target author name
            style_dna: Style DNA description
            rhetorical_type: Rhetorical mode
            critic: SemanticCritic instance for validation
            verbose: Enable verbose logging

        Returns:
            Simplified text generated from blueprint, or best_draft if simplification fails validation
        """
        # CRITICAL: Check if blueprint is incomplete
        # If incomplete, use original_text as source (not broken blueprint)
        if self._is_blueprint_incomplete(blueprint):
            # Use original text directly as the source
            source_text = blueprint.original_text
            blueprint_text = "N/A (Using original text due to incomplete blueprint)"
        else:
            # Use blueprint text as before
            blueprint_text = self._get_blueprint_text(blueprint)
            source_text = blueprint.original_text

        repair_prompt_template = _load_prompt_template("translator_repair.md")
        repair_prompt = repair_prompt_template.format(
            original_text=source_text,
            blueprint_text=blueprint_text
        )

        system_prompt_template = _load_prompt_template("translator_simplification_system.md")
        system_prompt = system_prompt_template.format(
            author_name=author_name,
            style_dna=style_dna
        )

        try:
            simplified = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=repair_prompt,
                temperature=0.2,  # Low temperature for repair
                max_tokens=self.translator_config.get("max_tokens", 200)
            )
            simplified = clean_generated_text(simplified)
            simplified = simplified.strip()
            # Restore citations and quotes if missing
            simplified = self._restore_citations_and_quotes(simplified, blueprint)

            # Validate simplification output - don't return broken fragments
            result = critic.evaluate(simplified, blueprint)
            # Note: fluency_score may not be present (removed in two-gate simplification)
            # Use recall_score as proxy for validation
            if result.get("recall_score", 0.0) < 0.7:
                if verbose:
                    print("  Simplification produced low-fluency output, reverting to best draft")
                return best_draft

            return simplified
        except Exception as e:
            # Fallback: return best draft if simplification fails
            if verbose:
                print(f"  Simplification failed with exception: {e}, reverting to best draft")
            return best_draft

    def _evolve_text(
        self,
        initial_draft: str,
        blueprint: SemanticBlueprint,
        author_name: str,
        style_dna: str,
        rhetorical_type: RhetoricalType,
        initial_score: float,
        initial_feedback: str,
        critic: 'SemanticCritic',
        verbose: bool = False,
        style_dna_dict: Optional[Dict[str, any]] = None,
        examples: Optional[List[str]] = None,
        force_semantic_injection: bool = False
    ) -> Tuple[str, float]:
        """Evolve text using population-based beam search with tournament selection.

        Uses a 3-pronged evolution strategy:
        1. Semantic Repair: Focuses on adding missing keywords (recall)
        2. Fluency Polish: Focuses on grammar and flow (fluency)
        3. Style Enhancement: Focuses on matching target voice (style)

        Each generation generates 3 candidates, evaluates them all, and selects the best
        using tournament selection with anti-regression (rejects candidates with lower recall)
        and elitism (only replaces parent if child is strictly better).

        Args:
            initial_draft: First generated draft
            blueprint: Original semantic blueprint
            author_name: Target author name
            style_dna: Style DNA description
            rhetorical_type: Rhetorical mode
            initial_score: Score from initial evaluation
            initial_feedback: Feedback from initial evaluation
            critic: SemanticCritic instance for evaluation
            verbose: Enable verbose logging

        Returns:
            Tuple of (best_draft, best_score)
        """
        # Initialize best draft and score
        best_draft = initial_draft
        best_score = initial_score
        best_feedback = initial_feedback

        # Initialize LLM Judge for ranking
        judge = LLMJudge(config_path=self.config_path)

        # Convergence tracking: track judge's selections
        last_judge_winner = None
        convergence_counter = 0
        convergence_threshold = 2  # Same winner for 2 rounds = converged

        # Load refinement config
        refinement_config = self.config.get("refinement", {})
        max_generations = refinement_config.get("max_generations", 3)
        pass_threshold = refinement_config.get("pass_threshold", 0.9)

        # Smart Patience parameters
        patience_counter = 0
        patience_threshold = refinement_config.get("patience_threshold", 3)
        patience_min_score = refinement_config.get("patience_min_score", 0.80)

        # Stagnation Breaker parameters (separate from patience)
        stagnation_counter = 0
        stagnation_threshold = 3

        # Dynamic temperature parameters
        current_temp = refinement_config.get("initial_temperature",
                                             refinement_config.get("refinement_temperature", 0.3))
        temperature_increment = refinement_config.get("temperature_increment", 0.2)
        max_temperature = refinement_config.get("max_temperature", 0.9)

        if verbose:
            print(f"  Evolution: Starting with score {best_score:.2f}")
            print(f"    Max generations: {max_generations}, Pass threshold: {pass_threshold}")
            print(f"    Patience: {patience_threshold} (min score: {patience_min_score})")
            print(f"    Initial temperature: {current_temp:.2f}")
            print(f"    Convergence threshold: {convergence_threshold} rounds")

        # Get blueprint text representation
        blueprint_text = self._get_blueprint_text(blueprint)

        # Extract style lexicon from style_dna_dict if available
        style_lexicon = None
        style_structure = None
        style_tone = None
        rag_example = None
        if style_dna_dict:
            style_lexicon = style_dna_dict.get("lexicon", [])
            style_structure = style_dna_dict.get("structure")
            style_tone = style_dna_dict.get("tone")
        # Get top RAG example for stylistic repair
        if examples and len(examples) > 0:
            rag_example = examples[0]

        # Get initial raw_score for fitness-based evolution (with style lexicon for density bonus)
        initial_eval = self.soft_scorer.evaluate_with_raw_score(best_draft, blueprint, style_lexicon=style_lexicon)
        best_raw_score = initial_eval.get("raw_score", best_score)

        # Evolution loop
        for gen in range(max_generations):
            # Check if we've reached acceptance criteria (using Fluency Forgiveness)
            # Evaluate current best draft to get recall/precision (with style whitelist)
            best_result = critic.evaluate(best_draft, blueprint, allowed_style_words=style_lexicon)
            if self._check_acceptance(
                recall_score=best_result["recall_score"],
                precision_score=best_result["precision_score"],
                fluency_score=best_result.get("fluency_score"),  # Optional - may not be present
                overall_score=best_score,
                pass_threshold=pass_threshold,
                original_text=blueprint.original_text,
                generated_text=best_draft
            ):
                if verbose:
                    print(f"  Evolution: Draft accepted (recall: {best_result['recall_score']:.2f}, precision: {best_result['precision_score']:.2f}, score: {best_score:.2f})")
                break

            if verbose:
                print(f"  Evolution Generation {gen + 1}/{max_generations}")
                print(f"    Current Score: {best_score:.2f}, Raw Score: {best_raw_score:.2f}")
                print(f"    Temperature: {current_temp:.2f}")

            try:
                # Step A: Diagnosis - Analyze current draft to select mutation strategy
                # If force_semantic_injection is True, prioritize semantic injection for rescue candidates
                if force_semantic_injection and gen == 0:
                    operator_type = OP_SEMANTIC_INJECTION
                    if verbose:
                        print(f"    Forcing semantic injection for rescue candidate repair")
                else:
                    operator_type = self._diagnose_draft(best_draft, blueprint, critic)
                # Use dynamic style if style lexicon is available and we're doing style polish
                if style_lexicon and operator_type == OP_STYLE_POLISH:
                    operator_type = OP_DYNAMIC_STYLE  # Use OP_DYNAMIC_STYLE when style DNA is available
                if verbose:
                    print(f"    Diagnosis: Selected operator '{operator_type}'")

                # Step B: Population - Generate 3 candidates using selected strategy
                candidates = self._generate_population_with_operator(
                    parent_draft=best_draft,
                    blueprint=blueprint,
                    author_name=author_name,
                    style_dna=style_dna,
                    rhetorical_type=rhetorical_type,
                    operator_type=operator_type,
                    temperature=current_temp,
                    num_candidates=3,
                    verbose=verbose,
                    style_lexicon=style_lexicon,
                    style_structure=style_structure,
                    style_tone=style_tone,
                    rag_example=rag_example
                )

                # Step C: Scoring - Get raw_score for all candidates
                scored_candidates = []
                for strategy, candidate_text in candidates:
                    if not candidate_text or not candidate_text.strip():
                        continue

                    # Get both critic evaluation and raw_score (with style whitelist and density bonus)
                    candidate_result = critic.evaluate(candidate_text, blueprint, allowed_style_words=style_lexicon)
                    candidate_eval = self.soft_scorer.evaluate_with_raw_score(candidate_text, blueprint, style_lexicon=style_lexicon)
                    candidate_raw_score = candidate_eval.get("raw_score", candidate_result.get("score", 0.0))

                    scored_candidates.append({
                        "strategy": strategy,
                        "text": candidate_text,
                        "score": candidate_result.get("score", 0.0),
                        "raw_score": candidate_raw_score,
                        "pass": candidate_result.get("pass", False),
                        "result": candidate_result,
                        "recall": candidate_result.get("recall_score", 0.0)
                    })

                # Step D: Selection - Pick candidate with highest raw_score if it improves over parent
                # CRITICAL: Accept improvement even if pass=False (fitness-based selection)
                best_candidate = None
                candidate_score = best_score
                candidate_raw_score = best_raw_score
                winning_strategy = None
                candidate_result = best_result

                if scored_candidates:
                    # Find candidate with highest raw_score
                    best_scored = max(scored_candidates, key=lambda c: c["raw_score"])

                    # Evolution Logic: Accept if raw_score improves (even if pass=False)
                    # CRITICAL: Also accept if score improves significantly even if raw_score is same
                    # This handles cases where critic gives good scores (0.84, 0.80) but pass=False
                    score_improvement = best_scored["score"] > best_score
                    raw_score_improvement = best_scored["raw_score"] > best_raw_score

                    if raw_score_improvement or (score_improvement and best_scored["score"] > 0.7):
                        best_candidate = best_scored["text"]
                        candidate_score = best_scored["score"]
                        candidate_raw_score = best_scored["raw_score"]
                        winning_strategy = best_scored["strategy"]
                        candidate_result = best_scored["result"]

                        if verbose:
                            reason = "raw_score" if raw_score_improvement else "score"
                            print(f"    ✓ Fitness Improvement: {winning_strategy} "
                                  f"({reason}: {candidate_raw_score if raw_score_improvement else candidate_score:.2f} > "
                                  f"{best_raw_score if raw_score_improvement else best_score:.2f}, "
                                  f"pass={best_scored['pass']}, score={best_scored['score']:.2f})")
                    else:
                        if verbose:
                            print(f"    ↻ No fitness improvement "
                                  f"(best raw_score: {best_scored['raw_score']:.2f} <= {best_raw_score:.2f}, "
                                  f"best score: {best_scored['score']:.2f} <= {best_score:.2f})")

                # Check if we have a winner (fitness improvement found)
                if best_candidate is not None:
                    # Improvement found: reset temperature, patience, and stagnation counter
                    current_temp = refinement_config.get("initial_temperature",
                                                         refinement_config.get("refinement_temperature", 0.3))
                    patience_counter = 0
                    stagnation_counter = 0

                    # Convergence check: same winner 2 rounds in a row?
                    if best_candidate == last_judge_winner:
                        convergence_counter += 1
                        if verbose:
                            print(f"    Convergence: Same candidate selected {convergence_counter}/{convergence_threshold} rounds")
                    else:
                        convergence_counter = 0
                        last_judge_winner = best_candidate

                    best_draft = best_candidate
                    best_score = candidate_score
                    best_raw_score = candidate_raw_score  # Update raw_score for next iteration
                    best_feedback = candidate_result["feedback"]
                    best_result = candidate_result
                    if verbose:
                        print(f"    ✓ Fitness Winner: {winning_strategy} "
                              f"(raw_score={best_raw_score:.2f}, score={best_score:.2f}, "
                              f"pass={candidate_result.get('pass', False)}, temp reset to {current_temp:.2f})")

                    # Convergence: Same winner for 2 rounds → stop
                    if convergence_counter >= convergence_threshold:
                        if verbose:
                            print(f"  Evolution: CONVERGED - Judge selected same candidate for {convergence_threshold} rounds. Stopping.")
                        break

                    # Check if improved draft meets acceptance criteria
                    if self._check_acceptance(
                        recall_score=candidate_result["recall_score"],
                        precision_score=candidate_result["precision_score"],
                        fluency_score=candidate_result.get("fluency_score"),  # Optional - may not be present
                        overall_score=candidate_score,
                        pass_threshold=pass_threshold,
                        original_text=blueprint.original_text,
                        generated_text=best_candidate
                    ):
                        if verbose:
                            print(f"  Evolution: Draft accepted after improvement "
                                  f"(recall: {candidate_result['recall_score']:.2f}, "
                                  f"precision: {candidate_result['precision_score']:.2f}, "
                                  f"score: {candidate_score:.2f})")
                        break
                else:
                    # No improvement (elitism kept parent): increment patience, stagnation, and increase temperature
                    patience_counter += 1
                    stagnation_counter += 1
                    current_temp = min(current_temp + temperature_increment, max_temperature)
                    if verbose:
                        print(f"    ↻ Stuck at {best_score:.2f}, increasing temperature to {current_temp:.2f} (patience: {patience_counter}/{patience_threshold}, stagnation: {stagnation_counter}/{stagnation_threshold})")

                    # Stagnation Breaker: triggers regardless of score after 3 non-improvements
                    if stagnation_counter >= stagnation_threshold:
                        if verbose:
                            print(f"  DEBUG: Stagnation detected (3 gens at {best_score:.2f}).")

                        # Check for logic failure before accepting stagnation exit
                        logic_fail = False
                        try:
                            stagnation_result = critic.evaluate(best_draft, blueprint, allowed_style_words=style_lexicon)
                            logic_fail = stagnation_result.get("logic_fail", False)
                        except Exception:
                            pass

                        if best_score >= 0.85 and not logic_fail:
                            if verbose:
                                print("  DEBUG: Score is acceptable. Early exit.")
                            break
                        elif logic_fail:
                            if verbose:
                                print(f"  DEBUG: Logic failure detected (score {best_score:.2f}), continuing evolution.")
                        else:
                            # Never trade a good score (>= 0.5) for a potentially worse simplification
                            # Only attempt simplification if best_score is truly bad (< 0.5)
                            if best_score < 0.5:
                                if verbose:
                                    print("  DEBUG: Score is very low (< 0.5). Attempting 'Simplification Pivot'...")
                                # Try one last radical simplification before giving up
                                final_attempt = self._generate_simplification(best_draft, blueprint, author_name, style_dna, rhetorical_type, critic, verbose)
                                return (final_attempt, best_score)
                            else:
                                # Return the best draft we have (B-grade is better than risking F-grade)
                                if verbose:
                                    print(f"  DEBUG: Returning best draft (score: {best_score:.2f}). Not risking simplification.")
                                return (best_draft, best_score)

                    # Smart Patience: early exit if stuck at good enough score
                    if patience_counter >= patience_threshold and best_score >= patience_min_score:
                        if verbose:
                            print(f"  Evolution converged at {best_score:.2f}. Early exit triggered (patience: {patience_counter})")
                        break

            except Exception as e:
                if verbose:
                    print(f"    ✗ Evolution Failed: Exception during refinement: {e}")
                    import traceback
                    traceback.print_exc()
                # Don't continue - let it fall through to next generation
                # This ensures we don't silently fail
                continue

        if verbose:
            print(f"  Evolution: Final score {best_score:.2f} (improvement: {best_score - initial_score:+.2f})")

        # Soft Pass Logic: Only accept if style is present OR score is very high
        # Ban boring sentences - force evolution to continue if no style
        # LOGIC VETO: Never soft pass if logic failure detected
        style_lexicon = None
        if style_dna_dict and isinstance(style_dna_dict, dict):
            style_lexicon = style_dna_dict.get("lexicon")

        style_density = 0.0
        if style_lexicon and best_draft:
            style_density = self._calculate_style_density(best_draft, style_lexicon)

        # Check for logic failure in the current best result
        logic_fail = False
        try:
            current_result = critic.evaluate(best_draft, blueprint, allowed_style_words=style_lexicon)
            logic_fail = current_result.get("logic_fail", False)
        except Exception:
            pass  # If evaluation fails, continue without logic check

        # Conditional Soft Pass:
        # - If style present (density > 0.1): Accept if score >= 0.85 AND no logic failure
        # - If boring (density == 0): Only accept if score >= 0.95 AND no logic failure
        # - NEVER accept if logic_fail is True (even with high score)
        if best_score >= 0.85 and not logic_fail:
            if style_density > 0.1:
                # Style is present, accept early
                if verbose and best_score < pass_threshold:
                    print(f"  Evolution: Soft pass accepted (score {best_score:.2f} >= 0.85, style_density {style_density:.2f} > 0.1)")
                return (best_draft, best_score)
            elif best_score >= 0.95:
                # Very high score even without style, accept
                if verbose:
                    print(f"  Evolution: Soft pass accepted (score {best_score:.2f} >= 0.95, style_density {style_density:.2f})")
                return (best_draft, best_score)
            else:
                # Boring sentence with mediocre score - reject soft pass, keep evolving
                if verbose:
                    print(f"  Evolution: Soft pass rejected (score {best_score:.2f} but style_density {style_density:.2f} <= 0.1, requiring >= 0.95 for boring sentences)")
        elif logic_fail:
            # Logic failure detected - reject soft pass regardless of score
            if verbose:
                print(f"  Evolution: Soft pass rejected (logic failure detected, score {best_score:.2f})")

        return (best_draft, best_score)


    def _calculate_composite_score(self, recall: float, precision: float, fluency: float) -> float:
        """Calculate weighted composite score prioritizing recall.

        Formula: (recall * 2 + fluency + precision) / 4

        Args:
            recall: Recall score (0-1).
            precision: Precision score (0-1).
            fluency: Fluency score (0-1).

        Returns:
            Composite score (0-1).
        """
        return (recall * 2.0 + fluency + precision) / 4.0

    def _diagnose_draft(
        self,
        draft: str,
        blueprint: SemanticBlueprint,
        critic: 'SemanticCritic'
    ) -> str:
        """Diagnose the draft to select mutation strategy.

        Args:
            draft: Current draft to diagnose.
            blueprint: Original semantic blueprint.
            critic: SemanticCritic instance.

        Returns:
            Mutation operator type (OP_SEMANTIC_INJECTION, OP_GRAMMAR_REPAIR, or OP_STYLE_POLISH).
        """
        result = critic.evaluate(draft, blueprint)
        recall = result.get("recall_score", 0.0)
        fluency = result.get("fluency_score", 1.0)  # Default to 1.0 if not present (fluency checks removed)

        # Diagnosis logic:
        # - If recall < 1.0: Missing keywords → Semantic Injection
        # - Elif fluency < 0.8: Grammar issues → Grammar Repair (only if fluency_score available)
        # - Else: Style needs enhancement → Style Polish
        if recall < 1.0:
            return OP_SEMANTIC_INJECTION
        elif fluency is not None and fluency < 0.8:
            return OP_GRAMMAR_REPAIR
        else:
            return OP_STYLE_POLISH

    def _generate_population_with_operator(
        self,
        parent_draft: str,
        blueprint: SemanticBlueprint,
        author_name: str,
        style_dna: str,
        rhetorical_type: RhetoricalType,
        operator_type: str,
        temperature: float = 0.6,
        num_candidates: int = 3,
        verbose: bool = False,
        style_lexicon: Optional[List[str]] = None,
        style_structure: Optional[str] = None,
        style_tone: Optional[str] = None,
        rag_example: Optional[str] = None
    ) -> List[Tuple[str, str]]:
        """Generate population using a specific mutation operator.

        Args:
            parent_draft: Current best draft.
            blueprint: Original semantic blueprint.
            author_name: Target author name.
            style_dna: Style DNA description.
            rhetorical_type: Rhetorical mode.
            operator_type: Mutation operator type.
            temperature: Temperature for generation.
            num_candidates: Number of candidates to generate.
            verbose: Enable verbose logging.

        Returns:
            List of (strategy_name, candidate_text) tuples.
        """
        operator = get_operator(operator_type)
        candidates = []

        for i in range(num_candidates):
            try:
                if verbose:
                    print(f"    Generating {operator_type} candidate {i+1}/{num_candidates}...")

                # Pass style DNA to operators that support it
                generate_kwargs = {
                    "current_draft": parent_draft,
                    "blueprint": blueprint,
                    "author_name": author_name,
                    "style_dna": style_dna,
                    "rhetorical_type": rhetorical_type,
                    "llm_provider": self.llm_provider,
                    "temperature": temperature,
                    "max_tokens": self.translator_config.get("max_tokens", 300)
                }
                # Add style DNA parameters if operator supports them
                if hasattr(operator, 'generate') and operator_type in [OP_STYLE_POLISH, OP_DYNAMIC_STYLE]:
                    generate_kwargs["style_lexicon"] = style_lexicon
                    generate_kwargs["style_structure"] = style_structure
                    if operator_type == OP_DYNAMIC_STYLE:
                        generate_kwargs["style_tone"] = style_tone

                # Pass style_lexicon and rag_example to semantic injection for stylistic repair
                if operator_type == OP_SEMANTIC_INJECTION:
                    generate_kwargs["style_lexicon"] = style_lexicon
                    generate_kwargs["rag_example"] = rag_example

                candidate = operator.generate(**generate_kwargs)

                # Restore citations and quotes
                candidate = self._restore_citations_and_quotes(candidate, blueprint)

                if candidate and candidate.strip():
                    candidates.append((operator_type, candidate))
                else:
                    if verbose:
                        print(f"    ✗ {operator_type} candidate {i+1} failed (empty)")
            except Exception as e:
                if verbose:
                    print(f"    ✗ {operator_type} candidate {i+1} failed: {e}")

        return candidates


    def _restore_citations_and_quotes(self, generated: str, blueprint: SemanticBlueprint) -> str:
        """Ensure all citations and quotes from blueprint are present in generated text.

        Citations can be appended to the end of the sentence if missing.
        Quotes must be present exactly - if missing or modified, this indicates
        a critical failure that should be caught by the critic.

        Also removes any non-standard citation formats (e.g., (Author, Year), (Smith 42)).

        CRITICAL: Only restores citations that actually exist in the original input text.
        This prevents phantom citations from being added.

        Args:
            generated: Generated text from LLM.
            blueprint: Original blueprint with citations and quotes.

        Returns:
            Generated text with citations restored (if missing) and non-standard formats removed.
        """
        if not generated:
            return generated

        # Remove all non-standard citation formats BEFORE checking for valid citations
        # Remove (Author, Year, p. #) format
        generated = re.sub(r'\([A-Z][a-z]+,?\s+\d{4}(?:,\s*p\.\s*#?)?\)', '', generated)
        # Remove (Author, Year, p. #) template pattern
        generated = re.sub(r'\(Author,?\s+Year,?\s+p\.\s*#\)', '', generated, flags=re.IGNORECASE)
        # Remove (Smith 42) format
        generated = re.sub(r'\([A-Z][a-z]+\s+\d+\)', '', generated)

        # Extract valid citations from generated text (only [^number] format)
        citation_pattern = r'\[\^\d+\]'
        generated_citations = set(re.findall(citation_pattern, generated))

        # CRITICAL FIX: Verify citations actually exist in original input text
        # Extract citations from original text to ensure we only restore real ones
        original_citations = set(re.findall(citation_pattern, blueprint.original_text))

        # Remove phantom citations from generated text (citations not in original input)
        phantom_citations = generated_citations - original_citations
        if phantom_citations:
            # Remove each phantom citation from generated text
            for phantom in phantom_citations:
                # Remove the citation, handling spacing
                generated = re.sub(re.escape(phantom) + r'\s*', '', generated)
                generated = re.sub(r'\s+' + re.escape(phantom), '', generated)
            # Re-extract citations after removal
            generated_citations = set(re.findall(citation_pattern, generated))

        # Only consider citations that are both in the blueprint AND in the original text
        # This prevents phantom citations from being restored
        valid_blueprint_citations = set([cit[0] for cit in blueprint.citations
                                         if cit[0] in original_citations])

        # Check which valid citations from blueprint are missing from generated text
        missing_citations = valid_blueprint_citations - generated_citations

        # Append missing citations to end of sentence
        if missing_citations:
            # Remove any trailing punctuation that might interfere
            generated = generated.rstrip('.!?')
            # Append citations
            citations_to_add = ' '.join(sorted(missing_citations))
            generated = f"{generated} {citations_to_add}"

        # Note: We don't restore quotes here because they must be exact word-for-word.
        # If quotes are missing or modified, the critic will catch it and fail validation.
        # This is intentional - quotes cannot be automatically restored.

        return generated

