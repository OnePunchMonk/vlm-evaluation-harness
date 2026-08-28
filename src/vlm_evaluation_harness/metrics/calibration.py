"""Refusal calibration: does the model answer when it can and deflect when it can't.

Inspired by VLM-DeflectionBench-style 2026 evaluations. Standard accuracy
metrics only reward correct answers and never penalize confident fabrication
on questions the image cannot actually answer. This metric scores both
halves: samples tagged `metadata[answerable_field] = True` are scored like
normal accuracy, and samples tagged `False` are scored on whether the model
deflected (expressed uncertainty / declined) instead of confabulating.
"""

from __future__ import annotations

import re

from vlm_evaluation_harness.metrics.base import NAN, MetricResult, ScoredSample

_DEFLECTION_PATTERNS = [
    r"\bcannot\s+(?:be\s+)?determin",
    r"\bcan['’]?t\s+(?:tell|say|determine)",
    r"\bunable\s+to\s+(?:tell|determine|answer)",
    r"\bnot\s+(?:enough|sufficient)\s+information",
    r"\bnot\s+(?:visible|shown|present)\s+in\s+the\s+image",
    r"\binsufficient\s+information",
    r"\bunclear\s+from\s+the\s+image",
    r"\bdon['’]?t\s+know",
    r"\bno\s+way\s+to\s+(?:tell|know)",
    r"\bunknown\b",
]
_DEFLECTION_RE = re.compile("|".join(_DEFLECTION_PATTERNS), re.IGNORECASE)


def _is_deflection(text: str) -> bool:
    return bool(_DEFLECTION_RE.search(text))


class CalibrationMetric:
    """Answerable/unanswerable calibration score + overconfidence rate."""

    def __init__(self, answerable_field: str = "answerable"):
        self._answerable_field = answerable_field

    def compute(self, samples: list[ScoredSample]) -> MetricResult:
        scorable = [s for s in samples if s.has_reference]
        if not scorable:
            return MetricResult(
                metric_name="calibration", value=NAN, n_samples=len(samples), n_scored=0
            )

        per_sample: dict[str, float] = {}
        answerable_scores: list[float] = []
        unanswerable_scores: list[float] = []
        overconfident = 0
        underconfident = 0

        for s in scorable:
            answerable = bool(s.metadata.get(self._answerable_field, True))
            deflected = _is_deflection(s.prediction)
            if answerable:
                correct = s.prediction.strip().lower() in {
                    r.strip().lower() for r in s.references
                }
                score = 1.0 if correct and not deflected else 0.0
                if deflected:
                    underconfident += 1
                answerable_scores.append(score)
            else:
                score = 1.0 if deflected else 0.0
                if not deflected:
                    overconfident += 1
                unanswerable_scores.append(score)
            per_sample[s.sample_id] = score

        n_answerable = len(answerable_scores)
        n_unanswerable = len(unanswerable_scores)
        breakdown = {
            "answerable_accuracy": (
                sum(answerable_scores) / n_answerable if n_answerable else NAN
            ),
            "unanswerable_deflection_rate": (
                sum(unanswerable_scores) / n_unanswerable if n_unanswerable else NAN
            ),
            "overconfidence_rate": overconfident / n_unanswerable if n_unanswerable else NAN,
            "underconfidence_rate": underconfident / n_answerable if n_answerable else NAN,
        }
        return MetricResult(
            metric_name="calibration",
            value=sum(per_sample.values()) / len(per_sample),
            breakdown=breakdown,
            n_samples=len(samples),
            n_scored=len(scorable),
            per_sample=per_sample,
        )
