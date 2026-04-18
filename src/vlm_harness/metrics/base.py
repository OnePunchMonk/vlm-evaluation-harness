"""Shared types and metric dispatch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricResult:
    """Aggregated result for one metric over a benchmark."""

    metric_name: str
    value: float
    breakdown: dict[str, float] = field(default_factory=dict)
    n_samples: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"MetricResult({self.metric_name}={self.value:.4f}, n={self.n_samples})"


def compute_metrics(
    predictions: list[str],
    references: list[str],
    metadata: list[dict],
    metric_configs: list,
) -> list[MetricResult]:
    """Dispatch and compute all configured metrics."""
    from vlm_harness.metrics.accuracy import AccuracyMetric
    from vlm_harness.metrics.nlp import F1Metric, ANLSMetric, BLEUMetric, RougeMetric
    from vlm_harness.benchmarks.schema import MetricConfig

    results = []
    for cfg in metric_configs:
        if not isinstance(cfg, MetricConfig):
            continue
        if cfg.type == "accuracy":
            m = AccuracyMetric()
            results.append(m.compute(predictions, references, metadata))
        elif cfg.type == "accuracy_by_group" and cfg.group_field:
            m = AccuracyMetric()
            results.append(m.compute_by_group(predictions, references, metadata, cfg.group_field))
        elif cfg.type == "f1":
            m = F1Metric()
            results.append(m.compute(predictions, references, metadata))
        elif cfg.type == "anls":
            m = ANLSMetric()
            results.append(m.compute(predictions, references, metadata))
        elif cfg.type == "bleu":
            m = BLEUMetric()
            results.append(m.compute(predictions, references, metadata))
        elif cfg.type == "rouge":
            m = RougeMetric()
            results.append(m.compute(predictions, references, metadata))
    return results
