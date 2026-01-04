"""Fact checker and repairer for style transfer output.

Extracts factual elements from source text and verifies they are
preserved in the styled output. Can repair outputs that have
mangled or hallucinated facts.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from ..utils.logging import get_logger
from ..utils.prompts import format_prompt

logger = get_logger(__name__)


@dataclass
class ExtractedFact:
    """A fact extracted from text."""
    fact_type: str  # date, time, age, number, measurement, address, name, year
    value: str  # The exact string
    normalized: str  # Normalized for comparison (lowercase, no extra spaces)
    context: str  # Surrounding context for repair


@dataclass
class FactCheckResult:
    """Result of fact checking."""
    source_facts: List[ExtractedFact]
    preserved_facts: List[ExtractedFact]
    missing_facts: List[ExtractedFact]
    preservation_rate: float


class FactExtractor:
    """Extracts factual elements from text."""

    # Number words to digit mappings
    WORD_TO_DIGIT = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
        'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
        'eighteen': '18', 'nineteen': '19', 'twenty': '20', 'thirty': '30',
        'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
        'eighty': '80', 'ninety': '90', 'hundred': '100',
    }

    def extract(self, text: str) -> List[ExtractedFact]:
        """Extract all factual elements from text."""
        facts = []

        # Full dates (March 15, 1923)
        for m in re.finditer(
            r'((?:January|February|March|April|May|June|July|August|September|'
            r'October|November|December)\s+\d{1,2},?\s*\d{4})', text
        ):
            facts.append(ExtractedFact(
                fact_type='date',
                value=m.group(1),
                normalized=m.group(1).lower().replace(',', ''),
                context=self._get_context(text, m.start(), m.end()),
            ))

        # Years (4-digit)
        for m in re.finditer(r'\b(1[89]\d{2}|20[0-2]\d)\b', text):
            # Skip if already captured in a date
            if not any(m.group(1) in f.value for f in facts if f.fact_type == 'date'):
                facts.append(ExtractedFact(
                    fact_type='year',
                    value=m.group(1),
                    normalized=m.group(1),
                    context=self._get_context(text, m.start(), m.end()),
                ))

        # Times (3:47 PM)
        for m in re.finditer(r'(\d{1,2}:\d{2})\s*(AM|PM|am|pm)?', text):
            facts.append(ExtractedFact(
                fact_type='time',
                value=m.group(0).strip(),
                normalized=m.group(1).lower(),
                context=self._get_context(text, m.start(), m.end()),
            ))

        # Ages (42 years old, age 42)
        for m in re.finditer(r'(?:age[d]?\s*)?(\d+)\s*years?\s*old|age[d]?\s+(\d+)', text, re.I):
            age = m.group(1) or m.group(2)
            facts.append(ExtractedFact(
                fact_type='age',
                value=age,
                normalized=age,
                context=self._get_context(text, m.start(), m.end()),
            ))

        # Measurements with units (12.5 degrees Celsius)
        for m in re.finditer(
            r'(\d+\.?\d*)\s*(degrees?\s*(?:Celsius|Fahrenheit|C|F)|'
            r'percent|%|meters?|feet|miles?|kilometers?|kg|pounds?|lbs?)',
            text, re.I
        ):
            facts.append(ExtractedFact(
                fact_type='measurement',
                value=m.group(0),
                normalized=m.group(1),
                context=self._get_context(text, m.start(), m.end()),
            ))

        # Street addresses (47 Oak Street)
        for m in re.finditer(
            r'(\d+)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+'
            r'(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln)',
            text
        ):
            facts.append(ExtractedFact(
                fact_type='address',
                value=m.group(0),
                normalized=m.group(1),  # Just the number for matching
                context=self._get_context(text, m.start(), m.end()),
            ))

        # Proper names (Dr. Elizabeth Warren)
        for m in re.finditer(
            r'(?:Dr\.?|Mr\.?|Mrs\.?|Ms\.?|Professor|Prof\.?)\s+'
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            text
        ):
            facts.append(ExtractedFact(
                fact_type='name',
                value=m.group(0),
                normalized=m.group(1).lower(),
                context=self._get_context(text, m.start(), m.end()),
            ))

        # Place names (Providence, Rhode Island)
        for m in re.finditer(
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),\s*'
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
            text
        ):
            # Check if it looks like a city, state pair
            facts.append(ExtractedFact(
                fact_type='location',
                value=m.group(0),
                normalized=m.group(0).lower(),
                context=self._get_context(text, m.start(), m.end()),
            ))

        return facts

    def _get_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Get surrounding context for a match."""
        ctx_start = max(0, start - window)
        ctx_end = min(len(text), end + window)
        return text[ctx_start:ctx_end]


class FactChecker:
    """Checks if facts from source are preserved in output."""

    def __init__(self):
        self.extractor = FactExtractor()

    def check(self, source: str, output: str) -> FactCheckResult:
        """Check fact preservation from source to output."""
        source_facts = self.extractor.extract(source)
        output_lower = output.lower()

        preserved = []
        missing = []

        for fact in source_facts:
            # Check if fact appears in output
            if self._fact_in_output(fact, output_lower):
                preserved.append(fact)
            else:
                missing.append(fact)
                logger.debug(f"Missing fact: {fact.fact_type} = {fact.value}")

        rate = len(preserved) / len(source_facts) if source_facts else 1.0

        return FactCheckResult(
            source_facts=source_facts,
            preserved_facts=preserved,
            missing_facts=missing,
            preservation_rate=rate,
        )

    def _fact_in_output(self, fact: ExtractedFact, output_lower: str) -> bool:
        """Check if a fact appears in the output."""
        # Direct match
        if fact.normalized in output_lower:
            return True

        # For numbers, also check spelled-out versions
        if fact.fact_type in ('age', 'year', 'measurement', 'time', 'address'):
            # Check if the number value appears
            if fact.fact_type == 'time':
                # Times like 3:47 - check for both formats
                parts = fact.normalized.split(':')
                if len(parts) == 2:
                    # Check for "three forty-seven" type patterns
                    spelled = self._spell_time(parts[0], parts[1])
                    if spelled and spelled in output_lower:
                        return True
            else:
                # Check if the numeric part appears
                numbers = re.findall(r'\d+\.?\d*', fact.value)
                for num in numbers:
                    if num in output_lower:
                        return True

        return False

    def _spell_time(self, hour: str, minute: str) -> Optional[str]:
        """Convert time to spelled form for matching."""
        try:
            h = int(hour)
            m = int(minute)
            # Simple patterns like "three forty-seven"
            hour_words = ['', 'one', 'two', 'three', 'four', 'five', 'six',
                         'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve']
            if h <= 12:
                if m == 0:
                    return hour_words[h]
                elif m < 10:
                    return f"{hour_words[h]} oh {self._num_to_word(m)}"
                else:
                    return f"{hour_words[h]} {self._num_to_word(m)}"
        except (ValueError, IndexError):
            pass
        return None

    def _num_to_word(self, n: int) -> str:
        """Convert number to word."""
        ones = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
                'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen',
                'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen']
        tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty',
                'seventy', 'eighty', 'ninety']

        if n < 20:
            return ones[n]
        elif n < 100:
            return tens[n // 10] + ('-' + ones[n % 10] if n % 10 else '')
        return str(n)


class FactRepairer:
    """Repairs outputs with missing or mangled facts."""

    def __init__(self, llm_provider):
        self.llm = llm_provider
        self.checker = FactChecker()

    def repair(self, source: str, output: str) -> Tuple[str, FactCheckResult]:
        """Repair facts in output based on source."""
        result = self.checker.check(source, output)

        if not result.missing_facts:
            logger.debug("No facts to repair")
            return output, result

        # Format missing facts for prompt
        missing_str = "\n".join(
            f"- {f.fact_type}: {f.value} (context: ...{f.context}...)"
            for f in result.missing_facts
        )

        system_prompt = format_prompt("fact_repair_system")
        user_prompt = format_prompt(
            "fact_repair_input",
            source=source,
            output=output,
            missing_facts=missing_str,
        )

        logger.info(f"Repairing {len(result.missing_facts)} missing facts")

        try:
            repaired = self.llm.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1,
                max_tokens=len(output.split()) * 2,
            )

            if repaired and len(repaired.split()) > 10:
                # Verify repair improved things
                new_result = self.checker.check(source, repaired)
                if new_result.preservation_rate >= result.preservation_rate:
                    logger.info(
                        f"Fact preservation improved: {result.preservation_rate:.0%} -> "
                        f"{new_result.preservation_rate:.0%}"
                    )
                    return repaired, new_result
                else:
                    logger.warning("Repair made things worse, keeping original")

        except Exception as e:
            logger.warning(f"Fact repair failed: {e}")

        return output, result


# Convenience functions
def check_facts(source: str, output: str) -> FactCheckResult:
    """Check fact preservation."""
    checker = FactChecker()
    return checker.check(source, output)


def extract_facts(text: str) -> List[ExtractedFact]:
    """Extract facts from text."""
    extractor = FactExtractor()
    return extractor.extract(text)
