# Task: Execute Repair Instruction

## Current Sentence:
"{current_sentence}"

## Action:
{action}

## Instruction:
{instruction}

## Previous Context:
{prev_context}

## Next Sentence (if merging):
{next_sentence}

## Instructions:
Apply the repair instruction to the current sentence. Maintain the author's voice ({author_name}).

Actions:
- "merge_with_next": Combine current sentence with the next sentence using the specified connector
- "split": Split the sentence at the specified point
- "rewrite_connector": Rewrite the transition/connector phrase
- "add_transition": Add a transition word or phrase at the start
- "simplify": Simplify the sentence structure
- "expand": Add more detail or elaboration

Output only the fixed sentence(s), no explanations.

