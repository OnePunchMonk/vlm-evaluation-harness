"""
MMStar  —  Designed to be non-saturable and leakage-resistant.

Dataset : Lin-Chen/MMStar
Split   : val (1500 samples)
Format  : multiple-choice (A/B/C/D), single image
SOTA    : ~67% (InternVL2-26B, 2024)
Headroom: large — explicitly designed so random/text-only baselines score ~10%

Why designed better than older benchmarks:
  - Questions require actual image understanding (verified manually)
  - No data leakage into pre-training corpora
  - Balanced across 6 core capabilities × 18 detailed axes

Capabilities tested:
  fine-grained perception, instance reasoning, logical reasoning,
  mathematics, science, technology
"""

from __future__ import annotations

import io
from typing import Optional

from datasets import load_dataset
from PIL import Image

from .base import Benchmark, Sample, build_mc_prompt, extract_letter


# MMStar's 6 core capability categories
CATEGORIES = [
    "fine-grained_perception",
    "instance_reasoning",
    "logical_reasoning",
    "math",
    "science_&_technology",
    "attribute_recognition",
]


class MMStar(Benchmark):
    name = "mmstar"
    capability = "multi_capability_vision_language"
    sota_score = 0.67

    def __init__(self, categories: Optional[list[str]] = None):
        self.categories = categories

    def load(self, n_samples: Optional[int] = None) -> list[Sample]:
        ds = load_dataset("Lin-Chen/MMStar", split="val", trust_remote_code=True)

        samples = []
        for row in ds:
            if n_samples and len(samples) >= n_samples:
                break

            img = row.get("image")
            if img is None:
                continue
            if isinstance(img, dict) and "bytes" in img:
                img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
            elif isinstance(img, bytes):
                img = Image.open(io.BytesIO(img)).convert("RGB")
            elif not isinstance(img, Image.Image):
                continue

            category = row.get("category", "")
            if self.categories and category not in self.categories:
                continue

            question = row.get("question", "")
            # MMStar embeds options in the question field in format:
            # "Question? A. opt1 B. opt2 C. opt3 D. opt4"
            # We pass it as-is since the options are already formatted.
            prompt = question + "\n\nAnswer with the letter only (A, B, C, or D)."
            answer = str(row.get("answer", "")).strip().upper()

            samples.append(Sample(
                id=f"mmstar_{row.get('index', len(samples))}",
                images=[img],
                prompt=prompt,
                choices=None,           # already embedded in prompt
                answer=answer,
                capability=self.capability,
                metadata={
                    "category": category,
                    "l2_category": row.get("l2-category", ""),
                },
            ))

        return samples

    def score(self, prediction: str, sample: Sample) -> bool:
        return extract_letter(prediction) == sample.answer.upper()
