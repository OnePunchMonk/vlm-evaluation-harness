"""Grounding metrics: IoU and Acc@0.5 over bounding-box predictions.

Pairs with `task_type: grounding` / `answer_extraction.strategy: bbox` in
benchmarks/schema.py and parsing/extractor.py. A sample's prediction and
reference are both "x1,y1,x2,y2" strings (see `parsing/extractor.py`'s
`_bbox` and `benchmarks/loader.py`'s `_coerce_references`, which passes a
plain "x1,y1,x2,y2" answer-column string through unchanged). A prediction
that failed to parse into a box (extraction left the raw, non-numeric model
text in place) scores 0 IoU rather than raising — that's a real model
failure to report, not a harness error.
"""

from __future__ import annotations

from vlm_evaluation_harness.metrics.base import MetricResult, ScoredSample, aggregate

Box = tuple[float, float, float, float]


def parse_box(text: str) -> Box | None:
    """Parse an "x1,y1,x2,y2" string into a 4-tuple of floats, or None."""
    parts = text.strip().split(",")
    if len(parts) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(p) for p in parts)
    except ValueError:
        return None
    return (x1, y1, x2, y2)


def iou(box_a: Box, box_b: Box) -> float:
    """Intersection-over-union of two axis-aligned boxes (x1, y1, x2, y2).

    Returns 0.0 for a degenerate (zero-area) or non-overlapping pair, never
    raises on division by zero.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection

    if union <= 0.0:
        return 0.0
    return intersection / union


def _iou_score(sample: ScoredSample) -> float:
    pred_box = parse_box(sample.prediction)
    if pred_box is None or not sample.references:
        return 0.0
    ref_box = parse_box(sample.references[0])
    if ref_box is None:
        return 0.0
    return iou(pred_box, ref_box)


class IoUMetric:
    """Mean IoU between predicted and ground-truth bounding boxes."""

    def compute(self, samples: list[ScoredSample]) -> MetricResult:
        return aggregate("iou", samples, _iou_score)


class AccAt50Metric:
    """Fraction of samples with IoU >= 0.5 against the ground-truth box."""

    def compute(self, samples: list[ScoredSample]) -> MetricResult:
        return aggregate("acc_at_50", samples, lambda s: 1.0 if _iou_score(s) >= 0.5 else 0.0)
