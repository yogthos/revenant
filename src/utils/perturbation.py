"""Input perturbation for inference to match training distribution.

Training data used random perturbations (typos, word drops, synonym swaps) to
force the model to creatively reconstruct text. At inference, we must apply
similar perturbations or the model produces mechanical output.

This matches the training script's perturb_text() function exactly.
"""

import random
from typing import Optional

# Simple synonym map matching training
SYNONYMS = {
    "big": ["large", "huge", "great"],
    "small": ["little", "tiny", "minor"],
    "old": ["ancient", "aged", "elderly"],
    "new": ["fresh", "recent", "modern"],
    "good": ["fine", "nice", "great"],
    "bad": ["poor", "awful", "terrible"],
    "house": ["building", "home", "dwelling"],
    "said": ["stated", "spoke", "remarked"],
    "walked": ["went", "moved", "traveled"],
    "looked": ["gazed", "stared", "peered"],
    "saw": ["noticed", "observed", "spotted"],
    "strange": ["odd", "peculiar", "unusual"],
    "dark": ["dim", "shadowy", "murky"],
    "light": ["bright", "pale", "faint"],
}


def perturb_text(
    text: str,
    perturbation_rate: float = 0.08,
    drop_adjectives: bool = True,
) -> str:
    """Apply random perturbations to text (Poor Man's NEFTune).

    Matches the training script exactly to ensure distribution match.

    Applies ~8% random changes:
    - Synonym swap: Replace word with synonym (40%)
    - Word drop: Remove non-essential words (30%)
    - Typo: Swap adjacent characters (30%)

    Args:
        text: Input text to perturb
        perturbation_rate: Probability of perturbing each word (default 8%)
        drop_adjectives: If True, 30% chance to strip adjectives

    Returns:
        Perturbed text
    """
    words = text.split()
    result = []
    droppable = {'the', 'a', 'an', 'very', 'really', 'just', 'quite'}

    # Common adjectives to drop (forces model to regenerate them in author's style)
    adjectives_to_drop = {
        'great', 'small', 'large', 'old', 'new', 'good', 'bad', 'long', 'short',
        'high', 'low', 'young', 'little', 'big', 'dark', 'light', 'strange',
        'ancient', 'terrible', 'horrible', 'beautiful', 'ugly', 'quiet', 'loud',
        'vast', 'deep', 'wide', 'narrow', 'thick', 'thin', 'heavy', 'empty',
    }

    for word in words:
        word_lower = word.lower().rstrip('.,!?;:"\'-')

        # Adjective dropping (separate from perturbation)
        if drop_adjectives and word_lower in adjectives_to_drop:
            if random.random() < 0.30:
                continue  # Drop the adjective

        if random.random() > perturbation_rate:
            result.append(word)
            continue

        # Choose perturbation type
        choice = random.random()

        if choice < 0.4:
            # Synonym swap (40% of perturbations)
            if word_lower in SYNONYMS:
                synonym = random.choice(SYNONYMS[word_lower])
                # Preserve case
                if word[0].isupper():
                    synonym = synonym.capitalize()
                result.append(synonym + word[len(word_lower):])
            else:
                result.append(word)

        elif choice < 0.7:
            # Word drop (30% of perturbations)
            if word.lower() in droppable:
                pass  # Drop the word
            else:
                result.append(word)

        else:
            # Typo - swap two adjacent chars (30% of perturbations)
            if len(word) > 3:
                i = random.randint(1, len(word) - 2)
                word = word[:i] + word[i+1] + word[i] + word[i+2:]
            result.append(word)

    return ' '.join(result)
