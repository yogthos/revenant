#!/usr/bin/env python3
"""Test different prompts for style transfer to avoid AI detection flags."""

import os
import re
import requests

# Sample paragraphs from input/imf.md
SAMPLE_SHORT = """It is a story that rarely makes headlines until the foundation cracks, but it is quietly unfolding in the bedrock of our financial system. In October 2025, the International Monetary Fund issued a report that serves as a grim retrospective on the post-pandemic era. Their words were careful, but the message was unmistakable: beneath the surface of apparent market calm, the "ground is shifting". While global financial markets appear placid despite trade and geopolitical uncertainty, this tranquility masks a buildup of systemic pressure."""

SAMPLE_LONG = """The scale of this accumulation is staggering. Banks in the U.S. and U.K. now have approximately $4.5 trillion in exposure to hedge funds, private credit groups, and other non-bank lenders. We are watching credit originate and circulate in a new, less-regulated sphere. In effect, banks are financing institutions that behave like banks but without the same constraints, creating a vast system of shadow credit that is highly leveraged and tightly woven into the core financial system. The IMF warns that the growing size of these NBFIs and their "deepened ties with banks" have heightened sectoral interconnection. This creates a "Unity of Opposites": banks and NBFIs appear to be competitors, yet they are structurally unified."""

SAMPLE_TEXT = SAMPLE_LONG  # Test with longer sample

# Current minimal prompt
PROMPT_CURRENT = """Rewrite in H.P. Lovecraft's style (~{word_count} words):
[STYLE: Long & Flowing | Complex Syntax]
{content}
###"""

# NEW v2 prompt - optimized for anti-AI detection
PROMPT_V2_NEW = """You are H.P. Lovecraft writing about this subject with your characteristic voice.

Your prose is NEVER smooth or mechanical. You:
- Write sentences that sprawl, then abruptly cut short
- Never use "Furthermore," "Moreover," "Additionally" - you jump between ideas
- Filter facts through your subjective lens (dread, wonder, skepticism)
- Sometimes lose yourself in parenthetical tangents
- Use dashes (—) and semicolons (;) instead of smooth connectors

Rewrite this (~{word_count} words), keeping ALL facts and figures:

{content}
###
"""

# Prompt v2: Anti-robotic guidelines (based on suggestions)
PROMPT_V2_ANTI_ROBOTIC = """[TASK]: Rewrite the content below in H.P. Lovecraft's distinctive voice.

[ANTI-ROBOTIC GUIDELINES]:
1. BREAK THE RHYTHM: Vary sentence length wildly. Crash long, winding explanations into short, blunt fragments.
2. KILL TRANSITIONS: BAN "Furthermore," "Moreover," "Additionally," "In conclusion." Jump between ideas abruptly.
3. ADOPT A STANCE: Filter facts through dread, unease, or cosmic horror. Use emotive weight over precision.
4. OBSCURITY OVER CLARITY: Bury the lead. Use parenthetical asides and non-linear structure.

Output ~{word_count} words. Preserve all facts and figures.

[INPUT]:
{content}

[OUTPUT]:
"""

# Prompt v3: Focused on burstiness and voice
PROMPT_V3_BURSTY = """Rewrite as H.P. Lovecraft might have written it.

CRITICAL RULES:
- Sentence length must VARY WILDLY: follow a 40-word sentence with a 5-word one
- NO transitional phrases (Furthermore, Moreover, Additionally, However)
- Express the author's UNEASE and DREAD about these facts
- Allow the prose to MEANDER before arriving at its point
- Use dashes (—) and semicolons (;) instead of smooth connectors

Target: ~{word_count} words. Keep all facts.

TEXT:
{content}

LOVECRAFT'S VERSION:
"""

# Prompt v4: Minimal but with anti-AI instructions
PROMPT_V4_MINIMAL_ANTI = """Rewrite in Lovecraft's voice (~{word_count} words).

AVOID: smooth transitions, uniform sentence length, objective tone.
EMBRACE: abrupt jumps, long/short alternation, subjective dread.

{content}
###"""

# Prompt v5: Character-based approach
PROMPT_V5_CHARACTER = """You are H.P. Lovecraft writing about modern finance with your characteristic dread.

Your prose is NEVER smooth or mechanical. You:
- Write sentences that sprawl, then abruptly cut short
- Never use "Furthermore" or "Moreover" - you jump between horrors
- Infuse cold cosmic dread into mundane financial facts
- Sometimes lose yourself in parenthetical tangents
- Bury revelations within baroque syntax

Rewrite this (~{word_count} words), keeping all facts:

{content}

---
"""

# Prompt v3: Even more aggressive about burstiness
PROMPT_V3_AGGRESSIVE = """You are H.P. Lovecraft. Your prose is deliberately UNEVEN.

MANDATORY RULES:
1. After any sentence over 25 words, your NEXT sentence must be under 8 words
2. Use fragments. "A mask." "The horror." "Nothing more."
3. NEVER use: Furthermore, Moreover, Additionally, However, Consequently
4. Use dashes (—) and semicolons (;) to break rhythm

Rewrite (~{word_count} words). Keep all facts and figures:

{content}
###
"""

# For quick comparison
PROMPTS = {
    "current": PROMPT_CURRENT,
    "v2_new": PROMPT_V2_NEW,
    "v3_aggressive": PROMPT_V3_AGGRESSIVE,
}


def test_with_deepseek(prompt_name: str, prompt_template: str, text: str, word_count: int = 150):
    """Test a prompt using DeepSeek API."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("DEEPSEEK_API_KEY not set")
        return None

    # Handle optional {author} and {style_tag} placeholders
    prompt = prompt_template.format(
        content=text,
        word_count=word_count,
        author="H.P. Lovecraft",
        style_tag="[STYLE: Long & Flowing | Complex Syntax]"
    )

    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 1.1,  # Higher as suggested
            "max_tokens": 500,
        },
        timeout=60,
    )

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


def analyze_output(text: str) -> dict:
    """Analyze output for AI detection markers."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentence_lengths = [len(s.split()) for s in sentences if s.strip()]

    # Banned transitions
    banned = ["furthermore", "moreover", "additionally", "in conclusion",
              "conversely", "notably", "consequently", "subsequently"]
    transitions_found = [w for w in banned if w in text.lower()]

    # Calculate variance in sentence length (burstiness)
    if len(sentence_lengths) > 1:
        avg = sum(sentence_lengths) / len(sentence_lengths)
        variance = sum((x - avg) ** 2 for x in sentence_lengths) / len(sentence_lengths)
        std_dev = variance ** 0.5
    else:
        std_dev = 0

    # Check for dashes and semicolons (human markers)
    dashes = text.count('—') + text.count('--')
    semicolons = text.count(';')

    return {
        "word_count": len(text.split()),
        "sentence_count": len(sentence_lengths),
        "avg_sentence_length": round(avg, 1) if sentence_lengths else 0,
        "sentence_length_std": round(std_dev, 1),
        "min_sentence": min(sentence_lengths) if sentence_lengths else 0,
        "max_sentence": max(sentence_lengths) if sentence_lengths else 0,
        "banned_transitions": transitions_found,
        "dashes": dashes,
        "semicolons": semicolons,
    }


def main():
    print("=" * 70)
    print("PROMPT EXPERIMENT: Testing different prompts for anti-AI detection")
    print("=" * 70)
    print(f"\nSample text ({len(SAMPLE_TEXT.split())} words):")
    print(SAMPLE_TEXT[:200] + "...")
    print()

    results = {}

    for name, template in PROMPTS.items():
        print(f"\n{'='*70}")
        print(f"Testing: {name}")
        print("=" * 70)

        output = test_with_deepseek(name, template, SAMPLE_TEXT, word_count=120)

        if output:
            print(f"\nOutput ({len(output.split())} words):")
            print("-" * 50)
            print(output)
            print("-" * 50)

            analysis = analyze_output(output)
            print(f"\nAnalysis:")
            print(f"  Sentences: {analysis['sentence_count']}")
            print(f"  Avg length: {analysis['avg_sentence_length']} words")
            print(f"  Length StdDev: {analysis['sentence_length_std']} (higher = more bursty)")
            print(f"  Range: {analysis['min_sentence']} - {analysis['max_sentence']} words")
            print(f"  Banned transitions: {analysis['banned_transitions'] or 'None'}")
            print(f"  Dashes: {analysis['dashes']}, Semicolons: {analysis['semicolons']}")

            results[name] = {"output": output, "analysis": analysis}
        else:
            print("Failed to get output")

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"{'Prompt':<20} {'StdDev':<8} {'Range':<12} {'Banned':<8} {'Dashes':<8}")
    print("-" * 60)

    for name, data in results.items():
        a = data["analysis"]
        print(f"{name:<20} {a['sentence_length_std']:<8} {a['min_sentence']}-{a['max_sentence']:<8} {len(a['banned_transitions']):<8} {a['dashes']:<8}")


if __name__ == "__main__":
    main()
