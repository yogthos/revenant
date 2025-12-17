Compare the generated text against these references using the HIERARCHY OF RULES:

{structure_section}{situation_section}{original_section}

GENERATED TEXT (to evaluate):
"{generated_text}"

--- TASK ---
Evaluate the GENERATED TEXT against the HIERARCHY:
1. SEMANTIC SAFETY: Does it preserve the original meaning? (Highest Priority)
2. STRUCTURAL RIGIDITY: Does it match the syntax/length/punctuation of the STRUCTURAL REFERENCE? (Second Priority)
3. VOCABULARY FLAVOR: Does it use the word choices/tone of the SITUATIONAL REFERENCE? (Third Priority)

If it fails, provide ONE single, specific instruction to fix the biggest error.
Do not list multiple conflicting errors. Pick the one that violates the highest priority rule.

Format your feedback as a direct editing instruction with specific metrics when relevant.
Example: "Current text has 25 words; Target has 12. Delete adjectives and split the relative clause."

{preservation_checks}

OUTPUT JSON with "pass", "feedback", "score", and "primary_failure_type" fields.
{preservation_instruction}
