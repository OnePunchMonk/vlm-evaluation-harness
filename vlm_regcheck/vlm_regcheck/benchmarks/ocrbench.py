"""
OCRBench  —  OCR and document understanding.

Dataset : echo840/OCRBench
Split   : test
Format  : open-ended generation, single image
SOTA    : ~63% (InternVL2, 2024)
Headroom: large

Five sub-tasks:
  Text Recognition, Scene Text-Centric VQA, Document-Oriented VQA,
  Key Information Extraction, Handwritten Mathematical Expression Recognition

Why important for regression detection:
  OCR capability is surprisingly fragile under post-training.
  It degrades when:
  - The vision encoder is LoRA-tuned with small rank (resolution loss)
  - SFT data has no document/text-image examples
  - Quantization is applied (INT4 loses fine-grained token reading)
  Provides a clean signal for "did we break reading ability."
"""

from __future__ import annotations

import io
import re
from typing import Optional

from datasets import load_dataset
from PIL import Image

from .base import Benchmark, Sample, normalize


TASK_TYPES = [
    "Text Recognition",
    "Scene Text-Centric VQA",
    "Document-Oriented VQA",
    "Key Information Extraction",
    "Handwritten Mathematical Expression Recognition",
]


class OCRBench(Benchmark):
    name = "ocrbench"
    capability = "ocr_document_understanding"
    sota_score = 0.63

    def __init__(self, task_types: Optional[list[str]] = None):
        self.task_types = task_types or TASK_TYPES

    def load(self, n_samples: Optional[int] = None) -> list[Sample]:
        ds = load_dataset("echo840/OCRBench", split="test", trust_remote_code=True)

        samples = []
        for row in ds:
            if n_samples and len(samples) >= n_samples:
                break

            task = row.get("type", "")
            if self.task_types and task not in self.task_types:
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
            answers = row.get("answers", row.get("answer", []))
            if isinstance(answers, str):
                answers = [answers]
            if not answers:
                continue

            prompt = f"{question}\n\nGive a concise, exact answer."
            # Use first answer as canonical; others as acceptable alternatives
            canonical = str(answers[0])

            samples.append(Sample(
                id=f"ocrbench_{row.get('index', len(samples))}",
                images=[img],
                prompt=prompt,
                choices=None,
                answer=canonical,
                capability=self.capability,
                metadata={
                    "task_type": task,
                    "all_answers": [str(a) for a in answers],
                },
            ))

        return samples

    def score(self, prediction: str, sample: Sample) -> bool:
        pred = normalize(prediction)
        all_answers = sample.metadata.get("all_answers", [sample.answer])

        for ans in all_answers:
            gt = normalize(str(ans))
            if pred == gt:
                return True
            # Substring match (prediction contains the answer)
            if gt and gt in pred:
                return True
            # Numeric equivalence
            try:
                if abs(float(pred.replace(",", "")) - float(gt.replace(",", ""))) < 0.01:
                    return True
            except ValueError:
                pass

        return False
