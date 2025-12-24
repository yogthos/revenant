"""Paragraph Atlas loader for statistical archetype generation.

This module loads JSON artifacts (archetypes.json, transition_matrix.json)
and handles Markov chain selection for paragraph generation.
"""

import json
import random
from pathlib import Path
from typing import Dict, Optional, List
import chromadb
from chromadb.config import Settings


class ParagraphAtlas:
    """Loads and manages paragraph archetype data for statistical generation."""

    def __init__(self, atlas_dir: str, author: str):
        """Initialize the paragraph atlas.

        Args:
            atlas_dir: Base directory for paragraph atlas (e.g., "atlas_cache/paragraph_atlas")
            author: Author name (e.g., "Mao" or "mao")
        """
        self.atlas_dir = Path(atlas_dir)
        self.author = author
        # Use lowercase author name for directory lookup
        author_lower = author.lower()
        self.author_dir = self.atlas_dir / author_lower

        # Load archetypes
        archetypes_path = self.author_dir / "archetypes.json"
        if not archetypes_path.exists():
            error_msg = (
                f"\n{'='*70}\n"
                f"ERROR: Paragraph Atlas not found for author '{self.author}'\n"
                f"{'='*70}\n"
                f"Missing file: {archetypes_path}\n\n"
                f"To fix this, build the paragraph atlas for '{self.author}':\n\n"
                f"  Option 1 (Recommended - sets up everything):\n"
                f"    python3 scripts/init_author.py --author \"{self.author}\" --style-file styles/sample_{self.author.lower()}.txt\n\n"
                f"  Option 2 (Just build the atlas):\n"
                f"    python3 scripts/build_paragraph_atlas.py styles/sample_{self.author.lower()}.txt --author \"{self.author}\"\n\n"
                f"  If you get 'No valid paragraphs found', try:\n"
                f"    python3 scripts/build_paragraph_atlas.py styles/sample_{self.author.lower()}.txt --author \"{self.author}\" --relaxed\n"
                f"{'='*70}\n"
            )
            raise FileNotFoundError(error_msg)

        with open(archetypes_path, 'r') as f:
            archetypes_data = json.load(f)

        # Filter out metadata
        self.archetypes = {
            int(k): v for k, v in archetypes_data.items()
            if k != "_metadata" and isinstance(k, str) and k.isdigit()
        }

        # Load transition matrix
        transition_path = self.author_dir / "transition_matrix.json"
        if not transition_path.exists():
            error_msg = (
                f"\n{'='*70}\n"
                f"ERROR: Transition matrix not found for author '{self.author}'\n"
                f"{'='*70}\n"
                f"Missing file: {transition_path}\n\n"
                f"This file should be created when building the paragraph atlas.\n"
                f"To fix this, rebuild the paragraph atlas:\n\n"
                f"    python3 scripts/build_paragraph_atlas.py styles/sample_{self.author.lower()}.txt --author \"{self.author}\"\n\n"
                f"Or use the turnkey script:\n"
                f"    python3 scripts/init_author.py --author \"{self.author}\" --style-file styles/sample_{self.author.lower()}.txt\n"
                f"{'='*70}\n"
            )
            raise FileNotFoundError(error_msg)

        with open(transition_path, 'r') as f:
            transition_data = json.load(f)

        self.transition_matrix = transition_data.get("matrix", {})

        # Connect to ChromaDB for example retrieval
        chroma_dir = self.author_dir / "chroma"
        try:
            self.client = chromadb.PersistentClient(path=str(chroma_dir))
            collection_name = f"paragraph_archetypes_{author.lower()}"
            self.collection = self.client.get_or_create_collection(collection_name)
        except Exception as e:
            print(f"Warning: Could not connect to ChromaDB: {e}")
            self.client = None
            self.collection = None

    def select_next_archetype(self, current_id: Optional[int] = None) -> int:
        """Select next archetype ID based on weighted probabilities from transition matrix.

        Args:
            current_id: Current archetype ID (None for first paragraph)

        Returns:
            Next archetype ID
        """
        # If no current ID, use default archetype or random
        if current_id is None:
            # Try to get default from config, otherwise use archetype 0
            return 0

        # Get transition probabilities for current archetype
        transitions = self.transition_matrix.get(str(current_id), {})

        if not transitions:
            # No transitions found, return default or random archetype
            return 0

        # Convert to list of (target_id, probability) tuples
        targets = [(int(k), v) for k, v in transitions.items()]

        # Weighted random selection
        target_ids = [t[0] for t in targets]
        weights = [t[1] for t in targets]

        selected = random.choices(target_ids, weights=weights, k=1)[0]
        return selected

    def get_archetype_description(self, archetype_id: int) -> Dict:
        """Return stats for the archetype for prompt building.

        Args:
            archetype_id: Archetype ID

        Returns:
            Dictionary with stats (avg_sents, avg_len, burstiness, style, example)
        """
        archetype = self.archetypes.get(archetype_id)
        if not archetype:
            raise ValueError(f"Archetype {archetype_id} not found")

        return {
            "id": archetype.get("id"),
            "avg_sents": archetype.get("avg_sents"),
            "avg_len": archetype.get("avg_len"),
            "burstiness": archetype.get("burstiness"),
            "style": archetype.get("style"),
            "example": archetype.get("example", "")  # Truncated snippet
        }

    def find_style_matched_archetype(self, target_sentence_count: int, tolerance: int = 3) -> Optional[int]:
        """
        Finds the 'nearest neighbor' archetype in terms of sentence count to borrow style metadata.
        Used when grafting author style onto a synthetic structure.

        Args:
            target_sentence_count: Desired sentence count (from synthetic structure)
            tolerance: Maximum acceptable difference in sentence count (default: 3)

        Returns:
            Archetype ID of the best match, or None if no match within tolerance
        """
        best_match_id = None
        best_divergence = float('inf')

        # Iterate through all available archetypes for this author
        for arch_id, arch_data in self.archetypes.items():
            # Get sentence count from archetype data
            arch_sents = arch_data.get('avg_sents', 0)

            divergence = abs(arch_sents - target_sentence_count)

            # We want the closest match within tolerance
            if divergence < best_divergence:
                best_divergence = divergence
                best_match_id = arch_id

        # Only return if it's a reasonable match (within tolerance)
        # Otherwise we risk mapping a 5-sentence style to a 20-sentence input
        if best_divergence <= tolerance:
            return best_match_id

        return None

    def get_author_avg_sentence_length(self) -> float:
        """
        Calculate the average sentence length (words per sentence) across all archetypes.
        Used to determine target density for elastic content mapping.

        Returns:
            Average words per sentence, or 25.0 as default if no archetypes available
        """
        if not self.archetypes:
            return 25.0  # Default to moderate density

        total_avg_len = 0.0
        count = 0

        for arch_id, arch_data in self.archetypes.items():
            avg_len = arch_data.get('avg_len', 0)
            if avg_len > 0:
                total_avg_len += avg_len
                count += 1

        if count > 0:
            return total_avg_len / count

        return 25.0  # Fallback default

    def get_example_paragraph(self, archetype_id: int) -> Optional[str]:
        """Retrieve full example paragraph from ChromaDB collection.

        Args:
            archetype_id: Archetype ID to retrieve example for

        Returns:
            Full paragraph text, or None if not found or ChromaDB unavailable
        """
        if not self.collection:
            return None

        try:
            # Use .get() for metadata filtering (not .query() which requires embeddings)
            # Note: archetype_id is stored as integer in metadata, not string
            results = self.collection.get(
                where={"archetype_id": archetype_id},
                limit=1
            )

            if results and results.get("documents") and len(results["documents"]) > 0:
                return results["documents"][0]

            return None
        except Exception as e:
            print(f"Warning: Could not retrieve example from ChromaDB: {e}")
            return None

    def get_structure_map(self, archetype_id: int) -> List[Dict]:
        """Extract exact sentence structure from exemplar paragraph.

        Returns a blueprint of sentence lengths and types for assembly-line construction.

        Args:
            archetype_id: Archetype ID to get structure map for

        Returns:
            List of dicts with structure information:
            [{'target_len': 12, 'type': 'simple', 'position': 0}, ...]
        """
        # Get exemplar paragraph
        exemplar = self.get_example_paragraph(archetype_id)
        if not exemplar:
            # Fallback: use truncated example from archetype description
            archetype = self.archetypes.get(archetype_id)
            if archetype:
                exemplar = archetype.get('example', '')

        if not exemplar:
            # Last resort: return structure based on averages
            archetype = self.archetypes.get(archetype_id)
            if archetype:
                avg_len = archetype.get('avg_len', 20)
                avg_sents = round(archetype.get('avg_sents', 3))
                # Create uniform structure map
                return [
                    {'target_len': round(avg_len), 'type': 'moderate', 'position': i}
                    for i in range(avg_sents)
                ]
            return []

        # Parse exemplar with spaCy
        try:
            import spacy
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                from spacy.cli import download
                download("en_core_web_sm")
                nlp = spacy.load("en_core_web_sm")

            doc = nlp(exemplar)
            sentences = list(doc.sents)

            if not sentences:
                return []

            structure_map = []
            for i, sent in enumerate(sentences):
                # Count words (tokens, excluding punctuation-only tokens)
                words = [token for token in sent if not token.is_punct]
                word_count = len(words)

                # Classify sentence type
                if word_count < 15:
                    sent_type = 'simple'
                elif word_count <= 25:
                    sent_type = 'moderate'
                else:
                    sent_type = 'complex'

                structure_map.append({
                    'target_len': word_count,
                    'type': sent_type,
                    'position': i
                })

            return structure_map

        except Exception as e:
            print(f"Warning: Could not parse exemplar for structure map: {e}")
            # Fallback to averages
            archetype = self.archetypes.get(archetype_id)
            if archetype:
                avg_len = archetype.get('avg_len', 20)
                avg_sents = round(archetype.get('avg_sents', 3))
                return [
                    {'target_len': round(avg_len), 'type': 'moderate', 'position': i}
                    for i in range(avg_sents)
                ]
            return []

    def find_rhetorical_match(self, rhetorical_type: str, n_candidates: int = 10, llm_provider: Optional[any] = None) -> Optional[Dict]:
        """
        Finds a paragraph in the author's corpus that structurally matches the input's logic.
        Uses JIT (Just-In-Time) matching: selects random candidates and uses heuristic or LLM to find best match.

        Args:
            rhetorical_type: The rhetorical structure type to match ("Contrast", "List", "Definition", etc.)
            n_candidates: Number of random candidates to sample (default: 10)
            llm_provider: Optional LLM provider for intelligent matching. If None, uses heuristic only.

        Returns:
            Dict with 'text' and metadata, or None if no match found
        """
        if not self.collection:
            return None

        try:
            # Get random candidates from ChromaDB collection
            # We'll get all documents and sample randomly
            all_results = self.collection.get(limit=1000)  # Get up to 1000 for sampling
            documents = all_results.get("documents", [])
            metadatas = all_results.get("metadatas", [])

            if not documents:
                return None

            # Sample random candidates
            n_candidates = min(n_candidates, len(documents))
            if n_candidates == 0:
                return None

            indices = random.sample(range(len(documents)), n_candidates)
            candidates = []
            for idx in indices:
                candidates.append({
                    "text": documents[idx],
                    "metadata": metadatas[idx] if idx < len(metadatas) else {}
                })

            # Try LLM-based matching if provider is available
            if llm_provider:
                try:
                    system_prompt = "You are a structural analyst. Match the requested Rhetorical Type to the best available Text Structure."

                    candidates_str = "\n".join([
                        f"{i}. {c['text'][:200]}{'...' if len(c['text']) > 200 else ''}"
                        for i, c in enumerate(candidates)
                    ])

                    user_prompt = f"""
Target Rhetorical Type: {rhetorical_type}

Available Author Paragraphs:
{candidates_str}

Which paragraph above best supports a "{rhetorical_type}" structure?
(e.g., if 'List', find a paragraph with commas/semicolons. If 'Contrast', find one with 'but/however').

Return ONLY the index number (0-{n_candidates-1}). If none fit well, return -1.
"""

                    response = llm_provider.call(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        model_type="editor",
                        require_json=False,
                        temperature=0.3,
                        max_tokens=50
                    )

                    # Try to extract index from response
                    import re
                    index_match = re.search(r'-?\d+', str(response))
                    if index_match:
                        best_idx = int(index_match.group(0))
                        if 0 <= best_idx < len(candidates):
                            return candidates[best_idx]
                except Exception as e:
                    # Fall back to heuristic if LLM fails
                    print(f"Warning: LLM matching failed ({e}), using heuristic")

            # Heuristic keyword matching fallback
            best_idx = -1
            best_score = 0

            for i, candidate in enumerate(candidates):
                text_lower = candidate["text"].lower()
                score = 0

                if rhetorical_type == "Contrast":
                    # Look for contrast indicators
                    if "but" in text_lower or "however" in text_lower or "although" in text_lower:
                        score += 3
                    if "not" in text_lower and ("but" in text_lower or "yet" in text_lower):
                        score += 2
                elif rhetorical_type == "List":
                    # Look for list indicators (high comma count, semicolons)
                    comma_count = text_lower.count(',')
                    semicolon_count = text_lower.count(';')
                    if comma_count > 3:
                        score += 2
                    if semicolon_count > 0:
                        score += 1
                    if "and" in text_lower or "or" in text_lower:
                        score += 1
                elif rhetorical_type == "Definition":
                    # Look for definition indicators
                    if " is " in text_lower or " means " in text_lower or " defined as " in text_lower:
                        score += 3
                    if "refers to" in text_lower or "denotes" in text_lower:
                        score += 2
                elif rhetorical_type == "Cause-Effect":
                    # Look for cause-effect indicators
                    if "because" in text_lower or "since" in text_lower or "as a result" in text_lower:
                        score += 3
                    if "therefore" in text_lower or "thus" in text_lower or "consequently" in text_lower:
                        score += 2
                elif rhetorical_type == "Narrative":
                    # Look for narrative indicators (temporal words)
                    if "first" in text_lower or "then" in text_lower or "next" in text_lower:
                        score += 2
                    if "after" in text_lower or "before" in text_lower or "when" in text_lower:
                        score += 1

                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx >= 0 and best_score > 0:
                return candidates[best_idx]

            return None

        except Exception as e:
            print(f"Warning: Could not find rhetorical match: {e}")
            return None

    def get_centroid_archetype(self) -> Optional[Dict]:
        """
        Returns a 'safe' archetype (median sentence count/length) for fallback.
        This represents the author's typical/representative style.

        Returns:
            Dict with archetype description, or None if no archetypes available
        """
        if not self.archetypes:
            return None

        # Get all archetypes with their sentence counts
        archetype_list = []
        for arch_id, arch_data in self.archetypes.items():
            avg_sents = arch_data.get('avg_sents', 0)
            if avg_sents > 0:
                archetype_list.append({
                    'id': arch_id,
                    'avg_sents': avg_sents,
                    'data': arch_data
                })

        if not archetype_list:
            return None

        # Sort by sentence count and find median
        archetype_list.sort(key=lambda x: x['avg_sents'])
        mid_index = len(archetype_list) // 2
        centroid = archetype_list[mid_index]

        # Return in same format as get_archetype_description
        return {
            "id": centroid['id'],
            "avg_sents": centroid['avg_sents'],
            "avg_len": centroid['data'].get('avg_len', 20),
            "burstiness": centroid['data'].get('burstiness', 'Low'),
            "style": centroid['data'].get('style', ''),
            "example": centroid['data'].get('example', '')
        }

    def get_style_matched_examples(self, n: int = 3, style_characteristics: Optional[Dict] = None) -> List[Dict]:
        """
        Retrieves random examples from the corpus to use as few-shot style targets.

        Args:
            n: Number of examples to retrieve
            style_characteristics: Optional dict with style filters (for future enhancement)

        Returns:
            List of dicts with 'text' and 'metadata'
        """
        if not self.collection:
            return []

        try:
            # Get all documents from ChromaDB collection
            all_results = self.collection.get(limit=1000)  # Get up to 1000 for sampling
            documents = all_results.get("documents", [])
            metadatas = all_results.get("metadatas", [])

            if not documents:
                return []

            # Build candidates list (similar to find_rhetorical_match)
            candidates = []
            for idx, doc in enumerate(documents):
                meta = metadatas[idx] if idx < len(metadatas) else {}
                candidates.append({
                    "text": doc,
                    "metadata": meta
                })

            # Safety check
            if not candidates:
                return []

            # Get n random samples
            sample_size = min(n, len(candidates))
            if sample_size == 0:
                return []

            return random.sample(candidates, sample_size)

        except Exception as e:
            print(f"Error fetching style examples: {e}")
            return []

    def get_rhetorical_references(self, rhetorical_type: str, mood: Optional[str] = None, n: int = 1) -> List[str]:
        """
        Retrieves sentences from the corpus that match the requested rhetorical type and mood.
        Used as 'Soft Templates' for the generator.

        Args:
            rhetorical_type: The rhetorical type ("contrast", "list", "definition", "general")
            mood: Optional grammatical mood ("narrative", "imperative", "definition", "general")
            n: Number of references to return (default: 1)

        Returns:
            List of reference sentences (strings)
        """
        # Expanded Library with Moods
        library = {
            "contrast": {
                "general": [
                    "It is not the consciousness of men that determines their being, but, on the contrary, their social being that determines their consciousness.",
                    "The metaphysical or vulgar evolutionist world outlook sees things as isolated, static and one-sided.",
                    "We must not look at problems one-sidedly, but must look at them all-sidedly."
                ],
                "narrative": [
                    "The old system failed, but the new system succeeded.",
                    "They attacked, but we retreated.",
                    "The revolution began in the cities, but it spread to the countryside."
                ],
                "imperative": [
                    "We must not look at problems one-sidedly, but must look at them all-sidedly.",
                    "Do not see things as isolated, but see them as connected."
                ]
            },
            "list": {
                "imperative": [
                    "The Red Army needs grain, the Red Army needs clothes, the Red Army needs oil.",
                    "We must have faith in the masses and we must have faith in the Party.",
                    "We must fight, we must win, we must change."
                ],
                "narrative": [
                    "There was grain, there was clothing, and there was oil.",
                    "They brought tools, they brought weapons, and they brought supplies.",
                    "The revolution had leaders, it had followers, and it had momentum."
                ],
                "general": [
                    "Qualitative change, quantitative change, and the negation of the negation—these are the laws.",
                    "The Red Army needs grain, the Red Army needs clothes, the Red Army needs oil."
                ]
            },
            "definition": {
                "general": [
                    "What is knowledge? It is the reflection of the objective world.",
                    "A revolution is not a dinner party, or writing an essay, or painting a picture.",
                    "This process, the practice of changing the world, is determined by scientific knowledge."
                ],
                "narrative": [
                    "Dialectics was the method Marx used to analyze reality.",
                    "The term was coined by Joseph Stalin to describe the Marxist approach."
                ]
            },
            "general": {
                "imperative": [
                    "We must act.",
                    "Let us look at the facts.",
                    "We must have faith in the masses and we must have faith in the Party."
                ],
                "narrative": [
                    "He named it Dialectical Materialism.",
                    "The revolution began in 1917.",
                    "Marx wrote the Communist Manifesto."
                ],
                "general": [
                    "The fundamental cause of the development of a thing is not external but internal.",
                    "Marxism-Leninism is the microscope and telescope of our political work.",
                    "Everything divides into two."
                ]
            }
        }

        # Normalize type
        key = rhetorical_type.lower()
        if key not in library:
            key = "general"

        # 1. Try to find specific Rhetoric + Mood match
        candidates = []
        if mood and mood.lower() in library.get(key, {}):
            candidates = library[key][mood.lower()]
        elif key in library:
            # Flatten all moods for this rhetoric if no mood match
            for m in library[key]:
                candidates.extend(library[key][m])

        if not candidates:
            # Fallback: Ignore rhetoric, match Mood (Critical for edge cases)
            if mood == "narrative":
                candidates = [
                    "The revolution emerged from the contradictions of capitalism.",
                    "He wrote the book on dialectical materialism.",
                    "It happened yesterday, and it changed everything."
                ]
            elif mood == "imperative":
                candidates = [
                    "We must fight.",
                    "Let us analyze the situation.",
                    "We need faith in the masses."
                ]
            elif mood == "definition":
                candidates = [
                    "What is dialectics? It is a method of analysis.",
                    "Materialism means that matter is primary."
                ]
            else:
                candidates = [
                    "It is a fact.",
                    "This is true.",
                    "The fundamental cause of the development of a thing is not external but internal."
                ]

        if not candidates:
            # Ultimate fallback
            candidates = ["Everything divides into two."]

        return random.sample(candidates, min(n, len(candidates)))

    def _create_synthetic_archetype(self, input_text: str, target_density: float = 25.0, author_profile: Optional[Dict] = None) -> Dict:
        """
        Creates a temporary archetype based on the input text's structure,
        reshaped to match the target author's sentence density.

        Args:
            input_text: Original input paragraph text
            target_density: Target words per sentence (from author's average)
            author_profile: Optional author style profile containing structural_dna

        Returns:
            Dictionary with structure_map, content_map (grouped sentences), and stats
        """
        try:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(input_text)
        except (ImportError, Exception):
            # Fallback: split by periods
            sentences = [s.strip() for s in input_text.split('.') if len(s.strip()) > 5]

        structure_map = []
        content_map = []  # Store which sentences go into which slot
        total_words = 0

        # DYNAMIC INFLATION CALCULATION
        # Try to get stats from profile, otherwise fallback to density heuristic
        struct_stats = author_profile.get('structural_dna', {}) if author_profile else {}
        avg_len = struct_stats.get('avg_words_per_sentence')

        if avg_len:
            # Dynamic Formula: Calculate inflation based on author's base density
            neutral_baseline = 15.0  # The input "Logical Beat" length
            raw_ratio = avg_len / neutral_baseline

            # Dampening Function: Pull the ratio closer to 1.0 to avoid runaway expansion
            # e.g., Mao (27.0) / Neutral (15.0) = 1.8
            # inflation = 1.0 + (1.8 - 1.0) * 0.5 = 1.4 (40% expansion)
            inflation_factor = 1.0 + (raw_ratio - 1.0) * 0.5

            # Clamp for safety (between 0.8 and 1.5)
            inflation_factor = max(0.8, min(inflation_factor, 1.5))
        else:
            # Fallback Legacy Logic (if no structural_dna in profile)
            if target_density > 25:
                inflation_factor = 1.20  # +20% budget for verbose authors
            elif target_density < 15:
                inflation_factor = 0.90  # -10% budget for concise authors
            else:
                inflation_factor = 1.10  # +10% standard buffer for style overhead

        # Helper function to avoid code duplication
        def create_slot(word_count, grouping):
            """Create a slot entry with inflation applied."""
            # Store raw length BEFORE inflation
            raw_len = word_count  # NEW: Capture the floor value

            # Apply Inflation
            adjusted_target = int(word_count * inflation_factor)

            # --- THE SAFETY CAP ---
            # Current LLMs struggle to maintain coherence beyond ~60 words.
            # Even if the author writes 100-word sentences, we cap at 60 to ensure success.
            # This prevents infinite retry loops when targets exceed model capabilities.
            MAX_SENTENCE_CEILING = 60

            # Ensure reasonable bounds (don't shrink below 3 or grow beyond ceiling)
            final_target = max(3, min(adjusted_target, MAX_SENTENCE_CEILING))

            # Determine slot type based on FINAL target
            if final_target < 10:
                slot_type = "simple"
            elif final_target < 25:
                slot_type = "moderate"
            else:
                slot_type = "complex"

            return {
                'target_len': final_target,
                'raw_len': raw_len,  # NEW: Store the floor value
                'type': slot_type
            }

        current_group = []
        current_word_count = 0
        complex_streak = 0  # NEW: Track streak of complex sentences

        for i, sent in enumerate(sentences):
            if not sent.strip():
                continue

            sent_len = len(sent.split())
            potential_len = current_word_count + sent_len

            # Elastic Grouping Logic with Burstiness Variation:
            # If target_density is high (>25), introduce variation to create natural rhythm
            # This prevents uniform chunk sizes (e.g., all 29w, 31w) for bursty authors
            if target_density > 25:
                # Add random variation to capacity (±20%)
                import random
                variation_factor = random.uniform(0.8, 1.2)
                adjusted_capacity = target_density * 1.5 * variation_factor
            else:
                adjusted_capacity = target_density * 1.5

            # RHYTHM BREAKER: Check if we need to force a break
            # If we have 2 complex slots in a row, don't let the next one get too big.
            # "Moderate" threshold is roughly 60% of target density.
            force_break = False
            if complex_streak >= 2:
                if potential_len > (target_density * 0.6):
                    force_break = True

            # Modified Merge Logic
            if not force_break and (current_word_count == 0 or potential_len < adjusted_capacity):
                # Merge: Add to current group
                current_group.append(sent)
                current_word_count += sent_len
            else:
                # FLUSH GROUP
                if current_group:
                    # Create slot with inflation applied
                    slot = create_slot(current_word_count, current_group)
                    structure_map.append(slot)
                    content_map.append(" ".join(current_group))
                    total_words += current_word_count

                    # Update Streak Logic
                    if slot['type'] == 'complex':
                        complex_streak += 1
                    else:
                        complex_streak = 0  # Reset streak on simple/moderate slots

                # Start new group with current sentence
                current_group = [sent]
                current_word_count = sent_len

        # Flush remaining group
        if current_group:
            slot = create_slot(current_word_count, current_group)
            structure_map.append(slot)
            content_map.append(" ".join(current_group))
            total_words += current_word_count
            # Note: Final slot streak tracking not needed as this is the last slot

        avg_words = total_words / len(structure_map) if structure_map else 0

        return {
            "id": "synthetic_fallback",
            "structure_map": structure_map,
            "content_map": content_map,  # NEW: Grouped content for direct mapping
            "stats": {
                "sentence_count": len(structure_map),
                "avg_words_per_sent": avg_words,
                "avg_len": avg_words,
                "avg_sents": len(structure_map)
            }
        }

