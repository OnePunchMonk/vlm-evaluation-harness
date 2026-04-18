"""
POPE  —  Polling-based Object Probing Evaluation (hallucination detection).

Dataset : lmms-lab/POPE
Split   : test
Format  : binary yes/no question about object existence in image
SOTA    : ~88% (InternVL2, 2024)
Headroom: medium — but especially sensitive to RLHF/DPO regressions
          where the model becomes sycophantic or over-cautious.

Three adversarial settings:
  random    — random object sampling (easiest)
  popular   — popular objects from COCO (harder)
  adversarial — objects frequently co-occurring with present objects (hardest)

Why important for regression detection:
  Hallucination rate is a primary regression signal after RLHF/DPO.
  Models can become either more hallucinatory (reward hacking) or
  over-cautious (refusing to confirm real objects). Both show as regression.

Metrics reported:
  accuracy, precision, recall, F1 — all sensitive to different failure modes.
"""

from __future__ import annotations

import io
from typing import Optional

from datasets import load_dataset
from PIL import Image

from .base import Benchmark, Sample, normalize


class POPE(Benchmark):
    name = "pope"
    capability = "hallucination_resistance"
    sota_score = 0.88

    CATEGORIES = ["adversarial", "popular", "random"]

    def __init__(self, categories: Optional[list[str]] = None):
        self.categories = categories or self.CATEGORIES

    def load(self, n_samples: Optional[int] = None) -> list[Sample]:
        ds = load_dataset("lmms-lab/POPE", split="test", trust_remote_code=True)

        samples = []
        per_cat = max(1, (n_samples or 900) // len(self.categories))
        counts = {c: 0 for c in self.categories}

        for row in ds:
            category = row.get("category", "random")
            if category not in self.categories:
                continue
            if n_samples and counts[category] >= per_cat:
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
            prompt = f"{question}\nAnswer with Yes or No only."
            answer = str(row.get("answer", "")).strip()

            samples.append(Sample(
                id=f"pope_{category}_{row.get('question_id', len(samples))}",
                images=[img],
                prompt=prompt,
                choices=["Yes", "No"],
                answer=answer,
                capability=self.capability,
                metadata={"category": category},
            ))
            counts[category] += 1

        if n_samples:
            samples = samples[:n_samples]
        return samples

    def score(self, prediction: str, sample: Sample) -> bool:
        pred = normalize(prediction)
        gt = normalize(sample.answer)
        # Accept yes/no at start of string
        if pred.startswith("yes"):
            pred = "yes"
        elif pred.startswith("no"):
            pred = "no"
        return pred == gt

    def evaluate_extended(self, model, samples: list[Sample], verbose: bool = False):
        """
        Extended eval that also returns precision, recall, F1 per category.
        Use this instead of base evaluate() for full hallucination analysis.
        """
        from collections import defaultdict
        from .base import BenchmarkResult

        per_cat: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
        per_sample_results = []

        for s in samples:
            prediction = model.answer(s.images, s.prompt, s.choices)
            pred = normalize(prediction)
            gt = normalize(s.answer)

            pred_yes = pred.startswith("yes")
            gt_yes = gt.startswith("yes")
            cat = s.metadata.get("category", "unknown")

            if pred_yes and gt_yes:
                per_cat[cat]["tp"] += 1
            elif pred_yes and not gt_yes:
                per_cat[cat]["fp"] += 1
            elif not pred_yes and gt_yes:
                per_cat[cat]["fn"] += 1
            else:
                per_cat[cat]["tn"] += 1

            per_sample_results.append({
                "id": s.id,
                "prediction": prediction,
                "answer": s.answer,
                "correct": self.score(prediction, s),
                "category": cat,
            })

        total_correct = sum(
            r["correct"] for r in per_sample_results
        )
        accuracy = total_correct / len(samples) if samples else 0.0

        # Compute per-category F1
        cat_metrics = {}
        for cat, counts in per_cat.items():
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
            cat_metrics[cat] = {"precision": precision, "recall": recall, "f1": f1}

        result = BenchmarkResult(
            benchmark=self.name,
            capability=self.capability,
            accuracy=accuracy,
            n_samples=len(samples),
            sota_score=self.sota_score,
            per_sample=per_sample_results,
        )
        result.metadata = {"per_category_metrics": cat_metrics}
        return result
