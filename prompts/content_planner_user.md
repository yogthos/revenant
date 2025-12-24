# Task: Content Distribution

## Neutral Content:
{neutral_text}

## Structure Slots:
{slot_descriptions}

## Instructions:
Distribute the content above into {num_slots} slots. Each slot has a target word count.

**CRITICAL CONSTRAINTS:**
1. **Distinct Information Only**: Each slot must contain DISTINCT information. Do NOT repeat the same fact in multiple slots.
2. **No Content Stretching**: If you run out of distinct facts, mark remaining slots as `EMPTY` (one per line).
3. **Content Density**: Prefer fewer, information-rich sentences over many repetitive sentences.
4. **Content Word Restrictions** Avoid using frivolous adjectives like intricate, vast, or profound, particularly ones associated with AI generated content
5. **Definition Priority**: If the text introduces or defines a key concept (e.g., "Consider a smartphone", "A watch is..."), that definition/introduction MUST appear in Slot 1 or Slot 2. Do not place definitions in later slots if they are needed to understand earlier references.
6. **Explicit Naming**: If the text introduces a concrete object (e.g., "smartphone", "watch", "hammer"), explicitly name it in early slots. Do not use generic terms like "device", "object", or "tool" until the specific object has been named.
7. **Preserve Lists**: If the neutral content contains a list (e.g., "a tree, a smartphone, a government"), preserve the complete list in a single slot. Do not split lists across multiple slots or summarize them into generic terms. Keep ALL items in the list.
8. **Preserve Order**: Maintain the logical order from the neutral content. Setup phrases should come before examples, examples before explanations. Do not rearrange the sequence.
9. **Preserve Setup Phrases**: If the content includes setup phrases like "Most of us are conditioned to...", "Consider...", "Think of...", preserve these in the appropriate slots. They provide essential context.

Output only the content for each slot, one per line. Use `EMPTY` for slots without distinct content.
Do not include slot numbers or labels.
