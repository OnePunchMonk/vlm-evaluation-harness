"""Answer extraction from raw model output."""

from __future__ import annotations

import re
from dataclasses import dataclass

from vlm_harness.benchmarks.schema import AnswerExtractionConfig
from vlm_harness.parsing.normalizer import normalize_answer


@dataclass
class ExtractionResult:
    raw: str          # original model output
    extracted: str    # extracted answer before normalization
    normalized: str   # after normalization
    confident: bool   # whether extraction was unambiguous


class AnswerExtractor:
    """Extracts structured answers from free-form model output."""

    def extract(self, text: str, config: AnswerExtractionConfig) -> ExtractionResult:
        strategy = config.strategy
        if strategy == "first_letter":
            extracted, confident = self._first_letter(text)
        elif strategy == "regex":
            extracted, confident = self._regex(text, config.regex_pattern or "")
        elif strategy == "number":
            extracted, confident = self._number(text)
        elif strategy == "yes_no":
            extracted, confident = self._yes_no(text)
        elif strategy == "json":
            extracted, confident = self._json_field(text)
        elif strategy == "exact":
            extracted, confident = text.strip(), True
        else:
            extracted, confident = text.strip(), True

        normalized = normalize_answer(extracted, mode=config.normalize)
        return ExtractionResult(
            raw=text, extracted=extracted, normalized=normalized, confident=confident
        )

    def _first_letter(self, text: str) -> tuple[str, bool]:
        """Extract the first capital letter (A/B/C/D) from the response."""
        # Try "The answer is X" patterns first
        for pattern in [
            r"(?:the\s+)?answer\s+is\s+([A-E])\b",
            r"^([A-E])[.\):\s]",
            r"\b([A-E])\b",
        ]:
            m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if m:
                return m.group(1).upper(), True
        # Last resort: first letter of the response
        stripped = text.strip()
        if stripped and stripped[0].upper() in "ABCDE":
            return stripped[0].upper(), False
        return text.strip()[:1].upper(), False

    def _regex(self, text: str, pattern: str) -> tuple[str, bool]:
        """Extract using a custom regex. Uses the first capture group."""
        if not pattern:
            return text.strip(), False
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if m:
            return (m.group(1) if m.lastindex else m.group(0)).strip(), True
        return text.strip(), False

    def _number(self, text: str) -> tuple[str, bool]:
        """Extract a numeric value (int or float)."""
        # Try to find explicit number patterns
        m = re.search(r"[-+]?\d+(?:\.\d+)?(?:\s*%)?", text)
        if m:
            return m.group(0).strip(), True
        return text.strip(), False

    def _yes_no(self, text: str) -> tuple[str, bool]:
        """Extract yes/no from the response."""
        lower = text.lower().strip()
        for yes_word in ["yes", "true", "correct", "right"]:
            if re.search(rf"\b{yes_word}\b", lower):
                return "yes", True
        for no_word in ["no", "false", "incorrect", "wrong"]:
            if re.search(rf"\b{no_word}\b", lower):
                return "no", True
        return text.strip(), False

    def _json_field(self, text: str) -> tuple[str, bool]:
        """Extract an 'answer' field from a JSON response."""
        import json
        # Try to find JSON block in the output
        m = re.search(r"\{.*?\}", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
                answer = data.get("answer", data.get("result", ""))
                return str(answer), True
            except json.JSONDecodeError:
                pass
        return text.strip(), False
