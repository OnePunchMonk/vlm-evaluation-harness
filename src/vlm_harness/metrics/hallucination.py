"""Hallucination metrics: CHAIR and POPE."""

from __future__ import annotations

import re

from vlm_harness.metrics.base import MetricResult

# COCO object categories (80 classes) for CHAIR
COCO_OBJECTS = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
}


class CHAIRMetric:
    """
    Caption Hallucination Assessment with Image Relevance.

    Measures the fraction of COCO objects mentioned in captions that are
    not present in the ground-truth annotations.
    """

    def compute(
        self,
        predictions: list[str],
        references: list[list[str]],  # per-sample list of ground-truth object labels
        metadata: list[dict],
    ) -> MetricResult:
        """
        Args:
            predictions: generated captions
            references: list of ground-truth object sets per sample
        """
        chair_s_scores = []  # per-sentence: fraction of objects hallucinated
        chair_i_scores = []  # per-image: 1 if any hallucination, else 0

        for caption, gt_objects in zip(predictions, references):
            mentioned = self._extract_objects(caption)
            gt_set = {o.lower() for o in gt_objects}
            if not mentioned:
                chair_s_scores.append(0.0)
                chair_i_scores.append(0.0)
                continue
            hallucinated = [o for o in mentioned if o not in gt_set]
            chair_s = len(hallucinated) / len(mentioned)
            chair_i = 1.0 if hallucinated else 0.0
            chair_s_scores.append(chair_s)
            chair_i_scores.append(chair_i)

        return MetricResult(
            metric_name="chair",
            value=sum(chair_s_scores) / len(chair_s_scores) if chair_s_scores else 0.0,
            breakdown={
                "chair_s": sum(chair_s_scores) / len(chair_s_scores) if chair_s_scores else 0.0,
                "chair_i": sum(chair_i_scores) / len(chair_i_scores) if chair_i_scores else 0.0,
            },
            n_samples=len(predictions),
        )

    def _extract_objects(self, text: str) -> list[str]:
        """Find COCO object names mentioned in text."""
        text_lower = text.lower()
        found = []
        for obj in COCO_OBJECTS:
            # Use word boundary matching for multi-word objects
            pattern = r"\b" + re.escape(obj) + r"\b"
            if re.search(pattern, text_lower):
                found.append(obj)
        return found


class POPEMetric:
    """
    Polling-based Object Probing Evaluation.

    For binary yes/no questions about object presence.
    Computes accuracy, precision, recall, and F1.
    """

    def compute(
        self,
        predictions: list[str],
        references: list[str],  # "yes" or "no"
        metadata: list[dict],
    ) -> MetricResult:
        normalized_preds = [self._normalize(p) for p in predictions]
        normalized_refs = [r.lower().strip() for r in references]

        tp = sum(p == "yes" and r == "yes" for p, r in zip(normalized_preds, normalized_refs))
        fp = sum(p == "yes" and r == "no" for p, r in zip(normalized_preds, normalized_refs))
        tn = sum(p == "no" and r == "no" for p, r in zip(normalized_preds, normalized_refs))
        fn = sum(p == "no" and r == "yes" for p, r in zip(normalized_preds, normalized_refs))

        accuracy = (tp + tn) / len(predictions) if predictions else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return MetricResult(
            metric_name="pope",
            value=accuracy,
            breakdown={
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "yes_rate": sum(p == "yes" for p in normalized_preds) / len(predictions),
            },
            n_samples=len(predictions),
        )

    def _normalize(self, text: str) -> str:
        lower = text.lower().strip()
        if re.search(r"\byes\b", lower):
            return "yes"
        if re.search(r"\bno\b", lower):
            return "no"
        # Fallback: treat as yes if first word is positive
        return "yes" if lower.startswith("y") else "no"
