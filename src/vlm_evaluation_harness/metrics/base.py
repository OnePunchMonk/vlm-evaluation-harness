"""Shared types and metric dispatch.

Two invariants hold throughout this package:

1. A sample with no ground truth is *excluded* from scoring, never compared
   against the empty string. Metrics report `n_scored` alongside `n_samples`.
2. A metric with nothing to score is NaN, never 0.0. A benchmark that failed
   to load must not be indistinguishable from a model that got everything
   wrong.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol

NAN = float("nan")


@dataclass
class ScoredSample:
    """One model prediction paired with its ground truth."""

    sample_id: str
    prediction: str
    references: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    confident: bool = True

    @property
    def has_reference(self) -> bool:
        return bool(self.references)


@dataclass
class MetricResult:
    """Aggregated result for one metric over a benchmark."""

    metric_name: str
    value: float
    breakdown: dict[str, float] = field(default_factory=dict)
    n_samples: int = 0
    # Number of samples that actually carried ground truth and were scored.
    n_scored: int = 0
    # Per-sample score in [0, 1], keyed by sample_id. Populated by sample-level
    # metrics and used for paired significance testing across runs.
    per_sample: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_defined(self) -> bool:
        return not math.isnan(self.value)

    def __repr__(self) -> str:
        value = "nan" if math.isnan(self.value) else f"{self.value:.4f}"
        return f"MetricResult({self.metric_name}={value}, n={self.n_scored}/{self.n_samples})"


class SampleMetric(Protocol):
    """A metric that produces one score per sample."""

    name: str

    def score(self, sample: ScoredSample) -> float: ...


def aggregate(
    metric_name: str,
    samples: list[ScoredSample],
    scorer,
    breakdown: dict[str, float] | None = None,
    metadata: dict[str, Any] | None = None,
) -> MetricResult:
    """Mean of `scorer` over every sample that has ground truth.

    Returns NaN — not 0.0 — when no sample is scorable.
    """
    per_sample = {s.sample_id: float(scorer(s)) for s in samples if s.has_reference}
    value = sum(per_sample.values()) / len(per_sample) if per_sample else NAN
    return MetricResult(
        metric_name=metric_name,
        value=value,
        breakdown=breakdown or {},
        n_samples=len(samples),
        n_scored=len(per_sample),
        per_sample=per_sample,
        metadata=metadata or {},
    )


def group_breakdown(
    samples: list[ScoredSample], scorer, group_field: str
) -> dict[str, float]:
    groups: dict[str, list[float]] = {}
    for s in samples:
        if not s.has_reference:
            continue
        key = str(s.metadata.get(group_field, "unknown"))
        groups.setdefault(key, []).append(float(scorer(s)))
    return {k: sum(v) / len(v) for k, v in sorted(groups.items())}


def extraction_failure_rate(samples: list[ScoredSample]) -> MetricResult:
    """Fraction of samples whose answer could not be extracted unambiguously.

    Reported on every discriminative run. A jump here is the signature of
    output-format drift, which otherwise masquerades as a capability
    regression.
    """
    if not samples:
        return MetricResult(
            metric_name="extraction_failure_rate", value=NAN, n_samples=0, n_scored=0
        )
    failures = {s.sample_id: 0.0 if s.confident else 1.0 for s in samples}
    return MetricResult(
        metric_name="extraction_failure_rate",
        value=sum(failures.values()) / len(failures),
        n_samples=len(samples),
        n_scored=len(samples),
        per_sample=failures,
    )


def compute_metrics(samples: list[ScoredSample], metric_configs: list) -> list[MetricResult]:
    """Dispatch and compute all configured metrics.

    Unknown metric types raise: a manifest asking for a metric the harness
    cannot compute must fail, not quietly produce a shorter report.
    """
    from vlm_evaluation_harness.benchmarks.schema import MetricConfig
    from vlm_evaluation_harness.metrics.accuracy import (
        AccuracyMetric,
        PairwiseGroupMetric,
        RelaxedAccuracyMetric,
        VQAAccuracyMetric,
    )
    from vlm_evaluation_harness.metrics.calibration import CalibrationMetric
    from vlm_evaluation_harness.metrics.grounding import AccAt50Metric, IoUMetric
    from vlm_evaluation_harness.metrics.hallucination import (
        CHAIRMetric,
        FineGrainedHallucinationMetric,
        POPEMetric,
    )
    from vlm_evaluation_harness.metrics.nlp import ANLSMetric, BLEUMetric, F1Metric, RougeMetric

    results: list[MetricResult] = []
    for cfg in metric_configs:
        if not isinstance(cfg, MetricConfig):
            continue
        if cfg.type == "accuracy":
            results.append(AccuracyMetric().compute(samples))
        elif cfg.type == "accuracy_by_group":
            if not cfg.group_field:
                raise ValueError("metric 'accuracy_by_group' requires group_field")
            results.append(AccuracyMetric().compute_by_group(samples, cfg.group_field))
        elif cfg.type == "vqa_accuracy":
            results.append(VQAAccuracyMetric().compute(samples))
        elif cfg.type == "relaxed_accuracy":
            results.append(RelaxedAccuracyMetric(cfg.tolerance).compute(samples))
        elif cfg.type == "f1":
            results.append(F1Metric().compute(samples))
        elif cfg.type == "anls":
            results.append(ANLSMetric().compute(samples))
        elif cfg.type == "bleu":
            results.append(BLEUMetric().compute(samples))
        elif cfg.type == "rouge":
            results.append(RougeMetric().compute(samples))
        elif cfg.type == "pairwise_group":
            results.append(PairwiseGroupMetric().compute(samples))
        elif cfg.type == "pope":
            results.append(POPEMetric().compute(samples))
        elif cfg.type == "chair":
            if not cfg.objects_field:
                raise ValueError("metric 'chair' requires objects_field")
            results.append(CHAIRMetric(cfg.objects_field).compute(samples))
        elif cfg.type == "fine_grained_hallucination":
            results.append(
                FineGrainedHallucinationMetric(cfg.field_name or "hallu_category").compute(
                    samples
                )
            )
        elif cfg.type == "calibration":
            results.append(
                CalibrationMetric(cfg.field_name or "answerable").compute(samples)
            )
        elif cfg.type == "iou":
            results.append(IoUMetric().compute(samples))
        elif cfg.type == "acc_at_50":
            results.append(AccAt50Metric().compute(samples))
        else:
            raise ValueError(f"Unknown metric type: {cfg.type!r}")

    results.append(extraction_failure_rate(samples))
    return results
