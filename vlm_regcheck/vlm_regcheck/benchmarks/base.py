"""
Base class for VLM regression benchmarks.

Each benchmark exposes:
  - load(n_samples)  -> list[Sample]
  - score(prediction, sample) -> bool
  - capability: what this benchmark tests
  - sota_score: approximate current SOTA (used to contextualize regression headroom)
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union

from PIL import Image


@dataclass
class Sample:
    id: str
    images: list[Image.Image]          # always a list; single-image tasks have len==1
    prompt: str                         # the question / instruction
    choices: Optional[list[str]]        # None for open-ended tasks
    answer: str                         # correct answer (letter A/B/C/D or free text)
    capability: str                     # e.g. "spatial_reasoning"
    metadata: dict = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    benchmark: str
    capability: str
    accuracy: float
    n_samples: int
    sota_score: float                   # approximate SOTA — headroom reference
    per_sample: list[dict] = field(default_factory=list)

    @property
    def headroom(self) -> float:
        """How much room is left before saturation."""
        return max(0.0, self.sota_score - self.accuracy)


class Benchmark(ABC):
    name: str
    capability: str
    sota_score: float                   # approximate current SOTA (0-1)

    @abstractmethod
    def load(self, n_samples: Optional[int] = None) -> list[Sample]:
        """Load samples from HuggingFace datasets."""
        ...

    @abstractmethod
    def score(self, prediction: str, sample: Sample) -> bool:
        """Return True if prediction is correct for this sample."""
        ...

    def evaluate(
        self,
        model,                          # VLMWrapper instance
        samples: list[Sample],
        verbose: bool = False,
    ) -> BenchmarkResult:
        correct = 0
        per_sample = []
        for s in samples:
            prediction = model.answer(s.images, s.prompt, s.choices)
            is_correct = self.score(prediction, s)
            correct += int(is_correct)
            if verbose:
                print(f"  [{s.id}] pred={prediction!r} gt={s.answer!r} {'✓' if is_correct else '✗'}")
            per_sample.append({
                "id": s.id,
                "prediction": prediction,
                "answer": s.answer,
                "correct": is_correct,
            })
        accuracy = correct / len(samples) if samples else 0.0
        return BenchmarkResult(
            benchmark=self.name,
            capability=self.capability,
            accuracy=accuracy,
            n_samples=len(samples),
            sota_score=self.sota_score,
            per_sample=per_sample,
        )


# ── Shared helpers ────────────────────────────────────────────────────────────

def extract_letter(text: str) -> str:
    """
    Extract answer letter (A/B/C/D/E) from generated text.
    Handles common patterns:
      - "A"  "A."  "(A)"  "Answer: A"  "The answer is A"
      - Multi-sentence responses where the answer letter appears
    """
    text = text.strip()

    # Explicit patterns first
    for pattern in [
        r"\bthe answer is[:\s]+([A-E])\b",
        r"\banswer[:\s]+([A-E])\b",
        r"^\s*\(?([A-E])\)?[\s\.\):]",
        r"\b([A-E])\s*[\.\)]\s",
        r"^([A-E])$",
        r"\b([A-E])\b",
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # Fallback: first capital letter in A-E range
    for ch in text:
        if ch.upper() in "ABCDE":
            return ch.upper()
    return ""


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def build_mc_prompt(question: str, choices: list[str]) -> str:
    """Format a multiple-choice question with lettered options."""
    letters = "ABCDEFGHIJ"
    options = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
    return f"{question}\n\nOptions:\n{options}\n\nAnswer with the letter only."
