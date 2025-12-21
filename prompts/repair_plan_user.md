# Task: Generate Repair Plan

## Draft Paragraph:
{draft}

## Original Blueprint (Structure Map):
{structure_map_info}

## Feedback:
{feedback}

## Instructions:
The original plan required: {structure_map_info}
The current draft has errors: {feedback}
Create a repair plan to match the original structure map.

Analyze the draft for flow, coherence, and natural transitions. Compare the draft against the blueprint targets. Generate a JSON repair plan with specific instructions for each sentence that needs fixing.

**CRITICAL: Output format must be valid JSON with double quotes, not single quotes.**

Output a JSON array of repair instructions. Each instruction must have:
- **"sent_index"**: Integer, 1-based sentence index (1 = first sentence, 2 = second sentence, etc.)
- **"action"**: String, one of: "shorten", "lengthen", "merge_with_next", "split", "rewrite_connector", "add_transition", "simplify", "expand"
- **"instruction"**: String, specific instruction for the LLM (e.g., "Shorten to 25 words by removing adjectives", "Merge with next sentence using 'however'")
- **"target_len"**: Integer (optional), target word count if action is "shorten" or "lengthen"

Focus on:
- Sentence length compliance (if blueprint targets are specified)
- Smooth transitions between sentences
- Natural flow and coherence
- Maintaining the author's voice ({author_name})
- Fixing any choppy or disjointed sections

**Output ONLY the JSON array, no other text. Use double quotes for all keys and string values.**

Example format:
```json
[
  {"sent_index": 1, "action": "add_transition", "instruction": "Add a transition word at the start to connect with previous context"},
  {"sent_index": 2, "action": "shorten", "target_len": 25, "instruction": "Shorten to 25 words by removing unnecessary adjectives"},
  {"sent_index": 3, "action": "merge_with_next", "instruction": "Merge with sentence 4 using 'however' to create smoother flow"}
]
```
