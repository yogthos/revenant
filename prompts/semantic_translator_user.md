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
2. **Preserve Analogies:** If the text compares X to Y (e.g., "AI art to photography" or "AI to theft machine"), keep the comparison structure intact.
3. **Preserve Key Arguments:** Do not flatten specific arguments into generic statements. Maintain the logical structure of the argument.
4. **Preserve Concrete Examples:** Keep specific historical examples, technical terms, and named concepts.
5. **No Attribution:** State the arguments directly (e.g., "AI is..." not "The author says AI is...") unless the perspective requires attribution.

Text:
{text}

Output the neutral summary:

