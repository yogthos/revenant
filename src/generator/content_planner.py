"""Content Planner for distributing neutral content into sentence slots.

This module handles the "Fact Distribution" step of the Assembly Line architecture,
mapping neutral content facts into specific sentence slots based on structure map.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from src.generator.llm_provider import LLMProvider
from src.utils.nlp_manager import NLPManager


class ContentPlanner:
    """Distributes neutral content facts into sentence slots for assembly-line construction."""

    def __init__(self, config_path: str = "config.json"):
        """Initialize the content planner.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.llm_provider = LLMProvider(config_path=config_path)
        # Lazy-load spaCy model for semantic drift detection
        self._nlp = None

    def plan_content(self, neutral_text: str, structure_map: List[Dict], author_name: str, source_text: Optional[str] = None) -> List[str]:
        """Distribute neutral content facts into sentence slots.

        Args:
            neutral_text: Neutral summary text containing facts to distribute
            structure_map: List of slot dicts with target_len and type
            author_name: Author name (for context, not used in planning)
            source_text: Optional original source text for semantic drift detection

        Returns:
            List of content strings, one per slot, aligned with structure_map
        """
        if not neutral_text or not neutral_text.strip():
            return [""] * len(structure_map)

        if not structure_map:
            return []

        # Sanity check: Log if structure map matches input density
        if source_text:
            try:
                from src.utils.text_processing import count_logical_beats
                input_beats = count_logical_beats(source_text)
                if len(structure_map) == input_beats:
                    # Perfect alignment - content should map 1:1
                    pass  # This is ideal, no action needed
            except Exception:
                pass  # Ignore errors in sanity check

        # Build slot descriptions
        slot_descriptions = []
        for i, slot in enumerate(structure_map, 1):
            target_len = slot.get('target_len', 20)
            slot_type = slot.get('type', 'moderate')
            slot_descriptions.append(f"Slot {i}: {target_len} words ({slot_type})")

        slot_descriptions_text = "\n".join(slot_descriptions)

        # Load prompt template
        prompts_dir = Path(__file__).parent.parent.parent / "prompts"
        try:
            template_path = prompts_dir / "content_planner_user.md"
            if template_path.exists():
                template = template_path.read_text().strip()
            else:
                # Fallback template
                template = self._get_fallback_template()
        except Exception:
            template = self._get_fallback_template()

        # Format prompt (handle both template formats)
        try:
            user_prompt = template.format(
                neutral_text=neutral_text,
                slot_descriptions=slot_descriptions_text,
                num_slots=len(structure_map)
            )
        except KeyError:
            # Template might have different placeholders, use fallback
            user_prompt = f"""# Task: Content Distribution

## Neutral Content:
{neutral_text}

## Structure Slots:
{slot_descriptions_text}

## Instructions:
Distribute the content above into {len(structure_map)} slots. Each slot has a target word count.
Output only the content for each slot, one per line. Do not include slot numbers or labels.
"""

        # Call LLM
        try:
            system_prompt = (
                "You are a content planner. Distribute facts into sentence slots based on target word counts.\n"
                "CRITICAL INSTRUCTION: You are a structural analyst, not a creative writer. "
                "Your content plan must map STRICTLY to the events present in the Source Text. "
                "Do NOT invent sequel events, future timelines, abstract theoretical conclusions, "
                "or generic 'machinery' descriptions to meet a length target. "
                "If the source text ends, your plan must end.\n"
                "**LOGIC RULE:** If the text introduces a new object or concept (e.g., 'smartphone', 'watch', 'hammer'), "
                "you MUST explicitly name it in Slot 1 or Slot 2. Do not refer to 'it', 'the device', 'the object', "
                "or 'the tool' until you have first named the specific object. Definitions and introductions must come before references."
            )
            response = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_type="editor",
                require_json=False,
                temperature=0.3,  # Low temperature for factual distribution
                max_tokens=1000
            )

            # Parse response: one content string per line
            content_slots = []
            for line in response.strip().split('\n'):
                line = line.strip()
                # Remove slot numbers/labels if present
                if ':' in line:
                    # Try to extract content after colon
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        line = parts[1].strip()
                # Preserve EMPTY markers (case-insensitive check)
                if line.upper() == "EMPTY":
                    content_slots.append("EMPTY")
                elif line:
                    content_slots.append(line)

            # Ensure we have the right number of slots
            while len(content_slots) < len(structure_map):
                content_slots.append("")
            if len(content_slots) > len(structure_map):
                content_slots = content_slots[:len(structure_map)]

            # Apply semantic drift guard if source_text is provided
            if source_text:
                content_slots = self._prune_hallucinated_slots(content_slots, source_text)

            return content_slots

        except Exception as e:
            print(f"Warning: Content planning failed: {e}")
            # Fallback: distribute evenly
            return self._fallback_distribution(neutral_text, len(structure_map))

    def _get_fallback_template(self) -> str:
        """Get fallback template if file doesn't exist."""
        return """# Task: Content Distribution

## Neutral Content:
{neutral_text}

## Structure Slots:
{slot_descriptions}

## Instructions:
Distribute the content above into {num_slots} slots. Each slot has a target word count.
Output only the content for each slot, one per line. Do not include slot numbers or labels.
"""

    def _fallback_distribution(self, neutral_text: str, num_slots: int) -> List[str]:
        """Fallback: distribute content evenly across slots."""
        if num_slots == 0:
            return []

        # Simple split by sentences
        sentences = neutral_text.split('. ')
        if len(sentences) < num_slots:
            # Not enough sentences, pad with empty
            result = sentences + [''] * (num_slots - len(sentences))
        else:
            # Distribute sentences across slots
            per_slot = len(sentences) // num_slots
            result = []
            for i in range(num_slots):
                start = i * per_slot
                end = start + per_slot if i < num_slots - 1 else len(sentences)
                slot_content = '. '.join(sentences[start:end])
                if slot_content and not slot_content.endswith('.'):
                    slot_content += '.'
                result.append(slot_content)

        return result[:num_slots]

    def _get_nlp(self):
        """Get or load spaCy model for semantic similarity checking.

        Returns:
            spaCy language model, or None if unavailable
        """
        if self._nlp is None:
            try:
                self._nlp = NLPManager.get_nlp()
            except Exception:
                # If spaCy not available, return None (check will be skipped)
                self._nlp = False
        return self._nlp if self._nlp is not False else None

    def extract_propositions(self, text: str) -> Dict[str, any]:
        """
        Breaks text into atomic logical beats and classifies rhetorical structure.
        Crucially, it preserves the LOGICAL FLOW (cause-effect, contrast) between propositions.

        Args:
            text: Input text to analyze

        Returns:
            Dict with:
            - "rhetorical_type": "Contrast" | "List" | "Definition" | "Cause-Effect" | "Narrative" | "General"
            - "propositions": List of proposition strings with logical connectors preserved
        """
        if not text or not text.strip():
            return {
                "rhetorical_type": "General",
                "propositions": []
            }

        system_prompt = (
            "You are a logic analyzer. Break the text into atomic propositions and identify the primary rhetorical structure.\n"
            "CRITICAL RULES:\n"
            "1. **PRESERVE LOGIC:** Do not just list facts. If A causes B, say 'Because A, B'. If A contrasts B, say 'A, however B'.\n"
            "2. **PRESERVE NOUNS:** Keep all proper nouns (Marx, Stalin), technical terms (Dialectics), and concrete lists (Lithium, Cobalt) exactly as written.\n"
            "3. **PRESERVE DEFINITIONS:** If the text defines a term ('X is Y'), that definition must be a proposition.\n"
            "4. **ATOMICITY:** Each proposition should be one distinct thought/claim.\n"
            "5. **PRESERVE SETUP PHRASES:** Keep introductory phrases like 'Most of us are conditioned to...', 'Consider...', 'Think of...' as part of propositions."
        )

        user_prompt = f"""
Analyze this text: "{text}"

1. List the atomic propositions in logical order.
   - Format: [Logical Connector] [Proposition Content]
   - Example: "The phone appears static." -> "However, it is actually a dynamic process." -> "This creates a contradiction."
   - Preserve logical connectors (However, Because, But, etc.) that show relationships between propositions.

2. Classify the structure as ONE of:
   - 'Contrast' (It is not X, but Y)
   - 'List' (It consists of A, B, and C)
   - 'Definition' (X is defined as Y)
   - 'Cause-Effect' (Because X, Y happens)
   - 'Narrative' (First X, then Y)
   - 'General' (Complex or mixed)

Output JSON: {{ "propositions": ["prop1", "prop2"...], "rhetorical_type": "type" }}
"""

        try:
            response = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_type="editor",
                require_json=True,
                temperature=0.3,  # Low temperature for factual extraction
                max_tokens=1000
            )

            # Parse JSON response
            if isinstance(response, str):
                # Try to extract JSON from response if it's wrapped in markdown or text
                import re
                json_match = re.search(r'\{[^{}]*"propositions"[^{}]*\}', response, re.DOTALL)
                if json_match:
                    response = json_match.group(0)
                data = json.loads(response)
            else:
                data = response

            # Validate structure
            if not isinstance(data, dict):
                raise ValueError("Response is not a dictionary")

            propositions = data.get("propositions", [])
            rhetorical_type = data.get("rhetorical_type", "General")

            # Validate rhetorical type
            valid_types = ["Contrast", "List", "Definition", "Cause-Effect", "Narrative", "General"]
            if rhetorical_type not in valid_types:
                rhetorical_type = "General"

            # Ensure propositions is a list
            if not isinstance(propositions, list):
                propositions = []

            return {
                "rhetorical_type": rhetorical_type,
                "propositions": propositions
            }

        except (json.JSONDecodeError, ValueError, KeyError, Exception) as e:
            # Fallback: Treat sentences as propositions
            print(f"Warning: Proposition extraction failed ({e}), using sentence fallback")
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            # Remove empty sentences
            sentences = [s for s in sentences if s]
            return {
                "rhetorical_type": "General",
                "propositions": sentences if sentences else [text]
            }

    def _prune_hallucinated_slots(self, plan: List[str], source_text: str) -> List[str]:
        """
        Removes final slots that have drifted semantically from the source text.
        Uses spaCy similarity to detect 'filler' topics (e.g. generic machinery).
        Implements iterative pruning: checks slots from the end until finding a valid one.

        Args:
            plan: List of content strings, one per slot
            source_text: Original source text to compare against

        Returns:
            Pruned plan with hallucinated final slots removed
        """
        if not plan or not source_text:
            return plan

        # Get spaCy model
        nlp = self._get_nlp()
        if not nlp:
            # If spaCy not available, return original plan
            return plan

        # Process source once
        try:
            source_doc = nlp(source_text)
        except Exception:
            # If processing fails, return original plan
            return plan

        # Iterative pruning: check slots from the end until finding a valid one
        # Hallucinations usually happen in a sequence at the end
        removed_count = 0
        original_plan = plan.copy()

        while plan:
            last_slot_content = plan[-1]

            # Skip if slot is empty or marked as EMPTY
            if not last_slot_content or last_slot_content.strip().upper() == "EMPTY":
                break

            # Process last slot content
            try:
                topic_doc = nlp(last_slot_content)
            except Exception:
                # If processing fails, assume valid and stop
                break

            # Calculate similarity
            # Note: spaCy similarity is 0.0-1.0.
            # Generic filler usually scores < 0.3 against specific narratives.
            # 'Soviet Ruins' vs 'Soviet Hunger' ~= 0.8
            # 'Soviet Ruins' vs 'Data Streams' ~= 0.15
            try:
                similarity = source_doc.similarity(topic_doc)
            except Exception:
                # If similarity calculation fails, assume valid and stop
                break

            # Threshold: 0.25 is a conservative cutoff for "completely unrelated"
            # Allows loose connections, kills unrelated content
            if similarity < 0.25:
                removed_count += 1
                plan.pop()  # Remove the bad slot and continue checking
            else:
                # If the last slot is valid, we assume the previous ones are too
                # (Hallucinations usually happen in a sequence at the end)
                break

        # Log removal if any slots were pruned
        if removed_count > 0:
            print(f"Warning: Semantic Drift Guard removed {removed_count} hallucinated slot(s) from the end.")

        return plan

