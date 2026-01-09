# Lovecraft LoRA Adapter Guide

This document covers findings and best practices for using the Lovecraft LoRA adapter for style transfer.

## Training Data Characteristics

The adapter was trained on H.P. Lovecraft's corpus with the following characteristics:

| Metric | Value |
|--------|-------|
| Mean expansion ratio | 1.21x |
| Word count instruction accuracy | 1.00 (exact match) |
| Training examples | ~3,700 |

### Expansion Distribution

The training data showed this distribution of input→output length ratios:

- **< 0.8x** (compression): 0.1%
- **0.8-1.0x** (slight compression): 25.3%
- **1.0-1.2x** (similar length): 52.8%
- **1.2-1.5x** (moderate expansion): 13.3%
- **> 1.5x** (strong expansion): 8.5%

**Key insight:** The model primarily learned to maintain similar length or modest expansion. Requesting 2x expansion is outside typical training distribution.

## Prompt Format

The model was trained on a specific prompt format. Inference prompts must match this format exactly to activate the LoRA weights correctly.

### Training Format

```
{persona_frame}

Write approximately {N} words.

[CONSTRAINT]: Do not use: 'Moreover', 'Furthermore'...
[CONSTRAINT]: Do not hedge...

{neutral_input}
###
```

### Critical Elements

1. **Persona frame** - Situational acting direction (not "You are Lovecraft")
2. **Word count instruction** - "Write approximately N words." on its own line
3. **Constraints** - Tiered system, typically 3-5 constraints
4. **Neutral input** - RTT-neutralized text (simple vocabulary, short sentences)
5. **Stop token** - `###` signals end of input

### What NOT to Include

The model was NOT trained on:
- `[EXPAND]` tags or expansion instructions
- Verbose transformation directives
- 10+ constraints per prompt
- Style hints or vocabulary suggestions

## Configuration Recommendations

### config.json Settings

```json
{
  "generation": {
    "target_expansion_ratio": 1.3,
    "lora_adapters": {
      "lora_adapters/lovecraft_14b": {
        "scale": 0.5,
        "temperature": 0.7,
        "top_p": 0.92,
        "min_p": 0.05,
        "repetition_penalty": 1.15,
        "max_tokens": 1024,
        "worldview": "lovecraft_worldview.txt"
      }
    }
  }
}
```

### Scale Parameter

The `scale` parameter controls LoRA influence:

| Scale | Behavior |
|-------|----------|
| 0.3-0.5 | Better instruction following, lighter style |
| 0.7-1.0 | Balanced style and instruction following |
| 1.5-2.5 | Strong style, may override instructions |

**Recommendation:** Start with `scale: 0.5` for expansion tasks, increase to 1.0+ if style is too weak.

### Expansion Ratio

Based on training data distribution:

| Ratio | Reliability |
|-------|-------------|
| 1.0-1.2 | High - within typical training range |
| 1.2-1.5 | Medium - achievable but less consistent |
| 1.5-2.0 | Low - outside typical training, results vary |
| > 2.0 | Very low - not recommended |

**Recommendation:** Use `target_expansion_ratio: 1.3` for reliable results.

## RTT Neutralization

The Round-Trip Translation (RTT) process strips style while preserving meaning:

1. **English → Mandarin** (HSK 5 vocabulary)
2. **Mandarin → English** (SVO structure, simple words)

### RTT Goals

- Strip literary vocabulary ("eldritch" → "strange")
- Simplify complex syntax
- Remove atmospheric flourishes
- Preserve ALL facts and meaning

### RTT Output Characteristics

- Short sentences (under 15 words)
- SVO (Subject-Verb-Object) structure
- Common vocabulary (CEFR B1 level)
- Clinical, monotonous tone

## Troubleshooting

### Output Not Expanding

**Symptoms:** Output length matches input despite higher target ratio.

**Causes:**
1. LoRA scale too high (overriding instructions)
2. Expansion ratio too aggressive for training distribution
3. RTT output too similar to input (not neutralized enough)

**Solutions:**
1. Lower `scale` to 0.5
2. Use `target_expansion_ratio: 1.3`
3. Verify RTT is producing simplified output

### Style Too Weak

**Symptoms:** Output sounds generic, not distinctly Lovecraftian.

**Causes:**
1. LoRA scale too low
2. RTT stripping too much content

**Solutions:**
1. Increase `scale` to 1.0-1.5
2. Consider skipping RTT (`skip_neutralization: true`) for already-neutral input

### Output Contains AI-isms

**Symptoms:** Output has "Moreover", "Furthermore", hedging language.

**Causes:**
1. Base model patterns bleeding through
2. Constraints not being followed

**Solutions:**
1. Increase `scale` slightly
2. Verify constraints are in prompt
3. Enable `reduce_repetition: true` for post-processing

## Persona Frames

The model responds to situational personas, not identity declarations.

### Effective (Situational)

```
You are writing in a diary by candlelight. Your hand is shaking.
You must record what happened, but you are terrified to write it down.
```

### Ineffective (Identity)

```
You are H.P. Lovecraft. Write in his style.
```

Persona frames are configured in `prompts/lovecraft_worldview.txt`.

## Content Type Detection

The system automatically detects content type and selects appropriate persona frames:

- **Narrative** - Stories, events, sequences → Journal/Testimony frames
- **Conceptual** - Explanations, mechanisms → Scholarly/Warning frames

This prevents instruction-content mismatch (e.g., using "explain the concept" for a story).
