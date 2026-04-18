"""
MMMU-Pro  —  Hard multi-discipline multimodal QA.

Dataset : MMMU/MMMU_Pro
Split   : test (standard), validation (for fast dev runs)
Format  : multiple-choice (5 or 10 options), 1-7 images per question
SOTA    : ~63% (GPT-4o, 2024)
Headroom: large — well-suited for regression detection

Capabilities tested:
  science, engineering, humanities, medicine, art — all requiring
  genuine cross-modal reasoning, not pattern matching.
"""

from __future__ import annotations

import io
from typing import Optional

from datasets import load_dataset
from PIL import Image

from .base import Benchmark, Sample, build_mc_prompt, extract_letter


class MMMUPro(Benchmark):
    name = "mmmu_pro"
    capability = "multi_discipline_reasoning"
    sota_score = 0.63                   # GPT-4o ~63%

    # Subject subsets available in MMMU-Pro
    SUBJECTS = [
        "Accounting", "Agriculture", "Architecture_and_Engineering",
        "Art", "Art_Theory", "Basic_Medical_Science", "Biology",
        "Chemistry", "Clinical_Medicine", "Computer_Science",
        "Design", "Diagnostics_and_Laboratory_Medicine", "Economics",
        "Electronics", "Energy_and_Power", "Finance", "Geography",
        "History", "Literature", "Management", "Marketing",
        "Materials", "Mathematics", "Mechanical_Engineering",
        "Music", "Pharmacy", "Physics", "Psychology", "Public_Health",
        "Sociology",
    ]

    def __init__(self, subjects: Optional[list[str]] = None, vision_only: bool = True):
        """
        Args:
            subjects: subset of SUBJECTS to load. None = all.
            vision_only: if True, only load the vision split (not text-only).
        """
        self.subjects = subjects or self.SUBJECTS
        self.vision_only = vision_only

    def load(self, n_samples: Optional[int] = None) -> list[Sample]:
        samples = []
        per_subject = max(1, (n_samples or 300) // len(self.subjects))

        for subject in self.subjects:
            try:
                split = "test"
                ds = load_dataset(
                    "MMMU/MMMU_Pro",
                    subject,
                    split=split,
                    trust_remote_code=True,
                )
            except Exception:
                continue

            count = 0
            for row in ds:
                if n_samples and count >= per_subject:
                    break

                # Collect images (up to 7)
                images = []
                for i in range(1, 8):
                    img_field = f"image_{i}"
                    if img_field in row and row[img_field] is not None:
                        img = row[img_field]
                        if isinstance(img, dict) and "bytes" in img:
                            img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
                        elif isinstance(img, bytes):
                            img = Image.open(io.BytesIO(img)).convert("RGB")
                        images.append(img)

                if not images:
                    continue

                options = row.get("options", [])
                if isinstance(options, str):
                    import ast
                    try:
                        options = ast.literal_eval(options)
                    except Exception:
                        options = [options]

                question = row.get("question", "")
                prompt = build_mc_prompt(question, options)
                answer = str(row.get("answer", "A")).strip().upper()

                samples.append(Sample(
                    id=f"mmmu_pro_{subject}_{row.get('id', count)}",
                    images=images,
                    prompt=prompt,
                    choices=options,
                    answer=answer,
                    capability=self.capability,
                    metadata={"subject": subject},
                ))
                count += 1

        if n_samples:
            samples = samples[:n_samples]
        return samples

    def score(self, prediction: str, sample: Sample) -> bool:
        return extract_letter(prediction) == sample.answer.upper()
