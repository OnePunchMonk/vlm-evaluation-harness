"""Accuracy metrics: exact match, VQA consensus accuracy, relaxed accuracy."""

from __future__ import annotations

import re

from vlm_harness.metrics.base import (
    MetricResult,
    ScoredSample,
    aggregate,
    group_breakdown,
)


def _exact(sample: ScoredSample) -> float:
    """1.0 if the prediction matches any acceptable reference."""
    return 1.0 if sample.prediction in sample.references else 0.0


class AccuracyMetric:
    """Exact-match accuracy against any of the sample's references."""

    def compute(self, samples: list[ScoredSample]) -> MetricResult:
        return aggregate("accuracy", samples, _exact)

    def compute_by_group(self, samples: list[ScoredSample], group_field: str) -> MetricResult:
        result = aggregate(
            f"accuracy_by_{group_field}",
            samples,
            _exact,
            breakdown=group_breakdown(samples, _exact, group_field),
        )
        return result


class VQAAccuracyMetric:
    """Official VQA accuracy: min(#annotators agreeing / 3, 1).

    VQAv2 ships ten independent annotator answers per question and an answer
    is fully correct only once at least three of them agree with it. Scoring
    against a single canonical answer — as this harness previously did —
    systematically under-reports and cannot be compared to any published
    VQAv2 number.
    """

    def compute(self, samples: list[ScoredSample]) -> MetricResult:
        return aggregate("vqa_accuracy", samples, self.score)

    def score(self, sample: ScoredSample) -> float:
        matches = sum(1 for ref in sample.references if ref == sample.prediction)
        return min(matches / 3.0, 1.0)


_NUMERIC_RE = re.compile(r"[-+]?\d*\.?\d+")


class RelaxedAccuracyMetric:
    """ChartQA's relaxed accuracy: numeric answers count within a tolerance.

    Non-numeric answers fall back to exact match against any reference.
    """

    def __init__(self, tolerance: float = 0.05):
        self._tolerance = tolerance

    def compute(self, samples: list[ScoredSample]) -> MetricResult:
        return aggregate(
            "relaxed_accuracy",
            samples,
            self.score,
            metadata={"tolerance": self._tolerance},
        )

    def score(self, sample: ScoredSample) -> float:
        pred_num = self._as_number(sample.prediction)
        for ref in sample.references:
            if sample.prediction == ref:
                return 1.0
            ref_num = self._as_number(ref)
            if pred_num is None or ref_num is None:
                continue
            if ref_num == 0:
                if abs(pred_num) <= self._tolerance:
                    return 1.0
            elif abs(pred_num - ref_num) / abs(ref_num) <= self._tolerance:
                return 1.0
        return 0.0

    def _as_number(self, text: str) -> float | None:
        cleaned = text.strip().replace(",", "").rstrip("%")
        m = _NUMERIC_RE.fullmatch(cleaned)
        if not m:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None


class PairwiseGroupMetric:
    """Winoground-style group accuracy over paired prompts.

    Each source sample is scored with two prompts (one per image); the pair
    counts only when both are answered correctly. Reported alongside the
    per-prompt accuracies, because a model that always answers "A" scores
    50% on each prompt and 0% here — which is the whole point of the
    benchmark.

    Note this is the two-question adaptation of Winoground's *text* score for
    generative VLMs, not the original image-text-matching formulation.
    """

    def compute(self, samples: list[ScoredSample]) -> MetricResult:
        from vlm_harness.metrics.base import NAN

        scorable = [s for s in samples if s.has_reference and "pair_id" in s.metadata]
        if not scorable:
            return MetricResult(
                metric_name="pairwise_group", value=NAN, n_samples=len(samples), n_scored=0
            )

        pairs: dict[str, dict[str, float]] = {}
        for s in scorable:
            slot = str(s.metadata.get("pair_slot", "a"))
            pairs.setdefault(str(s.metadata["pair_id"]), {})[slot] = _exact(s)

        per_sample = {
            pair_id: 1.0 if slots and all(v == 1.0 for v in slots.values()) else 0.0
            for pair_id, slots in pairs.items()
        }
        slot_scores: dict[str, list[float]] = {}
        for slots in pairs.values():
            for slot, score in slots.items():
                slot_scores.setdefault(slot, []).append(score)

        return MetricResult(
            metric_name="pairwise_group",
            value=sum(per_sample.values()) / len(per_sample),
            breakdown={
                f"prompt_{slot}_accuracy": sum(v) / len(v)
                for slot, v in sorted(slot_scores.items())
            },
            n_samples=len(samples),
            n_scored=len(per_sample),
            per_sample=per_sample,
        )
