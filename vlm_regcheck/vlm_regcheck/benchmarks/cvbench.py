"""
CV-Bench  —  2D and 3D spatial understanding.

Dataset : nyu-visionx/CV-Bench
Split   : test
Format  : multiple-choice (A/B/C/D), single image
SOTA    : ~77% (GPT-4o, 2024)
Headroom: medium-large

Covers two axes:
  2D: count, relation (above/below/left/right/etc.)
  3D: count, relation (depth ordering, spatial relationships)

Why important:
  Spatial reasoning is disproportionately damaged by:
  - Text-heavy SFT (model stops "looking" and starts "guessing")
  - Over-parameterized LoRA on language side
  - RLHF that rewards verbosity over precision
  Provides a complementary signal to MMVP (perception vs. reasoning).
"""

from __future__ import annotations

import io
from typing import Optional

from datasets import load_dataset
from PIL import Image

from .base import Benchmark, Sample, build_mc_prompt, extract_letter


class CVBench(Benchmark):
    name = "cvbench"
    capability = "spatial_reasoning"
    sota_score = 0.77

    TASK_TYPES = ["2D", "3D"]

    def __init__(self, task_types: Optional[list[str]] = None):
        self.task_types = task_types or self.TASK_TYPES

    def load(self, n_samples: Optional[int] = None) -> list[Sample]:
        ds = load_dataset("nyu-visionx/CV-Bench", split="test", trust_remote_code=True)

        samples = []
        for row in ds:
            if n_samples and len(samples) >= n_samples:
                break

            task = row.get("type", "2D")
            if task not in self.task_types:
                continue

            img = row.get("image")
            if img is None:
                continue
            if isinstance(img, dict) and "bytes" in img:
                img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
            elif isinstance(img, bytes):
                img = Image.open(io.BytesIO(img)).convert("RGB")
            elif not isinstance(img, Image.Image):
                continue

            question = row.get("question", "")
            choices = row.get("choices", [])
            if isinstance(choices, str):
                import ast
                try:
                    choices = ast.literal_eval(choices)
                except Exception:
                    choices = [choices]

            prompt = build_mc_prompt(question, choices) if choices else (
                question + "\n\nGive a concise answer."
            )

            answer = str(row.get("answer", "A")).strip().upper()
            if answer not in "ABCDE" and choices:
                for i, c in enumerate(choices):
                    if str(c).upper() == answer.upper():
                        answer = "ABCDE"[i]
                        break

            samples.append(Sample(
                id=f"cvbench_{row.get('idx', len(samples))}",
                images=[img],
                prompt=prompt,
                choices=choices if choices else None,
                answer=answer,
                capability=self.capability,
                metadata={"task_type": task, "source": row.get("source", "")},
            ))

        return samples

    def score(self, prediction: str, sample: Sample) -> bool:
        return extract_letter(prediction) == sample.answer.upper()
