Distill this text into a neutral, logical summary. Preserve causal links. Remove all rhetoric.

**Perspective Constraint:**
You must write this summary from the **{target_perspective}** point of view.
{pronoun_guidance}
Do NOT use 'The text says' or 'The author describes'. Be direct. Preserve the subject's agency and perspective.

{global_context_section}

{author_voice_instruction}

**CRITICAL: Stance Awareness**
- If the text describes an opposing view, make it clear the author is critiquing it
- Do NOT present counter-arguments as facts
- Check the Global Thesis above: if this paragraph mentions a counter-argument, mark it as such
- Example: If the author criticizes Baudelaire, write "The author criticizes Baudelaire's view that..." not "Baudelaire argued that..."

**CRITICAL: Author Voice Preservation**
- When preserving perspective, ensure you are writing AS the author, not ABOUT the author
- Do NOT convert the author's voice into a third-person analysis of the author's views
- If perspective is author_voice_third_person, write as the author would write, not as an analysis
- Example: BAD: "The subject argues that AI is theft." | GOOD: "AI is fundamentally a form of theft."
- Do NOT attribute views to "The subject", "The author", or the author's name when using author_voice_third_person
- State the arguments as facts, not as attributed statements

**CRITICAL SUMMARIZATION RULES:**
1. **Preserve Proper Nouns:** Do not generalize specific names. Keep exact names like "Baudelaire", "Photoshop", "Luddite Trap", "DeepSeek", "Disney", etc. PRESERVE specific proper nouns (people, places) and key technical terms.
2. **Preserve Lists and Enumerations:** If the text contains a list (e.g., "a tree, a smartphone, a government" or "lithium from Chile and cobalt from the Congo"), you MUST preserve ALL items in the list. Do not summarize lists into generic terms. Keep the complete enumeration with all items.
3. **Preserve Setup Phrases:** If the text uses setup phrases like "Most of us are conditioned to...", "Consider...", "Think of...", "Imagine...", "Visualize...", you MUST preserve these phrases. They establish essential context for understanding. Do not remove or generalize them.
4. **Preserve Introductions and Definitions:** If the text uses an analogy or example to introduce a concept (e.g., "Consider a smartphone", "Think of a watch", "Like a mechanical clock"), you MUST include that specific example in the summary. Do NOT generalize it to "an object", "a device", or "it". If the text defines a term, keep the definition intact.
5. **Preserve Concrete Nouns:** Keep specific concrete nouns (e.g., "smartphone", "watch", "hammer", "lithium", "cobalt") exactly as written. Do not replace them with generic terms like "object", "device", "tool", or "material".
6. **Preserve Analogies:** If the text compares X to Y (e.g., "AI art to photography" or "AI to theft machine"), keep the comparison structure intact.
7. **Preserve Key Arguments:** Do not flatten specific arguments into generic statements. Maintain the logical structure of the argument.
8. **Preserve Concrete Examples:** Keep specific historical examples, technical terms, and named concepts.
9. **Preserve Logical Flow:** Maintain the order of setup → list → introduction → explanation → application. If a concept is introduced before being used, preserve that order. Do not rearrange the logical sequence.
10. **No Attribution:** State the arguments directly (e.g., "AI is..." not "The author says AI is...") unless the perspective requires attribution.

Text:
{text}

Output the neutral summary:

