"""Hallucination metrics: POPE and CHAIR.

Both are dispatchable from a manifest (`type: pope`, `type: chair`). POPE in
particular reports **yes_rate**, which is the number that actually matters: a
model that answers "yes" to every object-presence probe scores ~50% accuracy
and looks unremarkable, while its yes_rate of 1.0 exposes it immediately.
"""

from __future__ import annotations

import re

from vlm_harness.metrics.base import NAN, MetricResult, ScoredSample

# COCO object categories (80 classes) for CHAIR
COCO_OBJECTS = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
}


class POPEMetric:
    """Polling-based Object Probing Evaluation over binary yes/no probes."""

    def compute(self, samples: list[ScoredSample]) -> MetricResult:
        scorable = [s for s in samples if s.has_reference]
        if not scorable:
            return MetricResult(
                metric_name="pope", value=NAN, n_samples=len(samples), n_scored=0
            )

        per_sample: dict[str, float] = {}
        tp = fp = tn = fn = 0
        yes_predictions = 0

        for s in scorable:
            pred = self._normalize(s.prediction)
            refs = {self._normalize(r) for r in s.references}
            ref = "yes" if "yes" in refs else "no"
            per_sample[s.sample_id] = 1.0 if pred == ref else 0.0
            yes_predictions += pred == "yes"
            if pred == "yes" and ref == "yes":
                tp += 1
            elif pred == "yes":
                fp += 1
            elif ref == "no":
                tn += 1
            else:
                fn += 1

        n = len(scorable)
        accuracy = (tp + tn) / n
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        return MetricResult(
            metric_name="pope",
            value=accuracy,
            breakdown={
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "yes_rate": yes_predictions / n,
            },
            n_samples=len(samples),
            n_scored=n,
            per_sample=per_sample,
        )

    def _normalize(self, text: str) -> str:
        lower = text.lower().strip()
        if re.search(r"\byes\b", lower):
            return "yes"
        if re.search(r"\bno\b", lower):
            return "no"
        return "yes" if lower.startswith("y") else "no"


class CHAIRMetric:
    """Caption Hallucination Assessment with Image Relevance.

    Ground-truth object labels are read from the sample's metadata under
    `objects_field`, which the manifest must declare via
    `fields.metadata_fields`.
    """

    def __init__(self, objects_field: str = "objects"):
        self._objects_field = objects_field

    def compute(self, samples: list[ScoredSample]) -> MetricResult:
        usable = [s for s in samples if self._objects_field in s.metadata]
        if not usable:
            return MetricResult(
                metric_name="chair", value=NAN, n_samples=len(samples), n_scored=0
            )

        per_sample: dict[str, float] = {}
        chair_i_scores: list[float] = []

        for s in usable:
            mentioned = self.extract_objects(s.prediction)
            gt = {str(o).lower() for o in (s.metadata.get(self._objects_field) or [])}
            if not mentioned:
                per_sample[s.sample_id] = 0.0
                chair_i_scores.append(0.0)
                continue
            hallucinated = [o for o in mentioned if o not in gt]
            per_sample[s.sample_id] = len(hallucinated) / len(mentioned)
            chair_i_scores.append(1.0 if hallucinated else 0.0)

        chair_s = sum(per_sample.values()) / len(per_sample)
        return MetricResult(
            metric_name="chair",
            value=chair_s,
            breakdown={
                "chair_s": chair_s,
                "chair_i": sum(chair_i_scores) / len(chair_i_scores),
            },
            n_samples=len(samples),
            n_scored=len(usable),
            per_sample=per_sample,
        )

    def extract_objects(self, text: str) -> list[str]:
        """Find COCO object names mentioned in text."""
        text_lower = text.lower()
        return [
            obj
            for obj in sorted(COCO_OBJECTS)
            if re.search(r"\b" + re.escape(obj) + r"\b", text_lower)
        ]
