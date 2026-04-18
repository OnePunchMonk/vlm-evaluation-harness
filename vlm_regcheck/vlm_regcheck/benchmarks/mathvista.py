"""
MathVista  —  Mathematical reasoning in visual contexts.

Dataset : AI4Math/MathVista
Split   : testmini (1000 samples, standard eval split)
Format  : multiple-choice AND free-form (mixed)
SOTA    : ~63% (GPT-4o, 2024)
Headroom: large

Capabilities tested:
  arithmetic, algebraic, geometric, statistical and logical
  reasoning — all grounded in charts, figures, and diagrams.

Why this is useful for regression detection:
  Fine-tuning on QA/instruction data often damages mathematical
  reasoning chains. This is the most sensitive benchmark for
  detecting that regression.
"""

from __future__ import annotations

import io
from typing import Optional

from datasets import load_dataset
from PIL import Image

from .base import Benchmark, Sample, build_mc_prompt, extract_letter, normalize


class MathVista(Benchmark):
    name = "mathvista"
    capability = "mathematical_visual_reasoning"
    sota_score = 0.63

    QUESTION_TYPES = ["multi_choice", "free_form"]

    def __init__(self, question_types: Optional[list[str]] = None):
        self.question_types = question_types or self.QUESTION_TYPES

    def load(self, n_samples: Optional[int] = None) -> list[Sample]:
        ds = load_dataset("AI4Math/MathVista", split="testmini", trust_remote_code=True)

        samples = []
        for row in ds:
            if n_samples and len(samples) >= n_samples:
                break

            qtype = row.get("question_type", "free_form")
            if qtype not in self.question_types:
                continue

            img = row.get("decoded_image") or row.get("image")
            if img is None:
                continue
            if isinstance(img, dict) and "bytes" in img:
                img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
            elif isinstance(img, bytes):
                img = Image.open(io.BytesIO(img)).convert("RGB")
            elif not isinstance(img, Image.Image):
                continue

            question = row.get("question", "")
            choices = row.get("choices") or []

            if qtype == "multi_choice" and choices:
                prompt = build_mc_prompt(question, choices)
                answer = str(row.get("answer", "")).strip()
                # Normalise answer to letter if it matches a choice
                if answer not in "ABCDE":
                    for i, c in enumerate(choices):
                        if normalize(answer) == normalize(str(c)):
                            answer = "ABCDE"[i]
                            break
            else:
                prompt = f"{question}\n\nGive a concise numerical answer."
                answer = str(row.get("answer", "")).strip()

            samples.append(Sample(
                id=f"mathvista_{row.get('pid', len(samples))}",
                images=[img],
                prompt=prompt,
                choices=choices if choices else None,
                answer=answer,
                capability=self.capability,
                metadata={
                    "question_type": qtype,
                    "skill": row.get("skill", ""),
                    "task": row.get("task", ""),
                },
            ))

        return samples

    def score(self, prediction: str, sample: Sample) -> bool:
        if sample.choices:
            # Multiple choice
            pred_letter = extract_letter(prediction)
            if pred_letter:
                return pred_letter == sample.answer.upper()
            # Fallback: check if prediction text matches the choice text
            if sample.answer in "ABCDE":
                idx = "ABCDE".index(sample.answer)
                if idx < len(sample.choices):
                    return normalize(prediction) == normalize(sample.choices[idx])
            return False
        else:
            # Free-form: exact or near-exact numeric match
            pred = normalize(prediction)
            gt = normalize(sample.answer)
            if pred == gt:
                return True
            # Try numeric comparison
            try:
                return abs(float(pred.replace(",", "")) - float(gt.replace(",", ""))) < 1e-3
            except ValueError:
                return False
