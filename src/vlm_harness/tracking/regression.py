"""Regression detection: diff two tracked runs and flag meaningful drops.

Severity thresholds adapted from the vlm_regcheck/ prototype in this repo,
which validated them empirically for base-vs-finetuned VLM comparisons.
They're intentionally conservative relative to typical benchmark noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from vlm_harness.tracking.history import HistoryEntry, HistoryStore

SEVERITY_CRITICAL = -0.10  # >10% drop
SEVERITY_HIGH = -0.05  # 5-10% drop
SEVERITY_MEDIUM = -0.03  # 3-5% drop
SEVERITY_LOW = -0.01  # 1-3% drop


def _severity_label(delta: float) -> str:
    if delta <= SEVERITY_CRITICAL:
        return "CRITICAL"
    if delta <= SEVERITY_HIGH:
        return "HIGH"
    if delta <= SEVERITY_MEDIUM:
        return "MEDIUM"
    if delta <= SEVERITY_LOW:
        return "LOW"
    if delta < 0:
        return "MINIMAL"
    return "OK"


@dataclass
class MetricDelta:
    benchmark: str
    metric_name: str
    baseline_model: str
    current_model: str
    baseline_value: float
    current_value: float
    delta: float
    severity: str
    flagged: bool

    @property
    def is_regression(self) -> bool:
        return self.delta < 0 and not math.isnan(self.delta)


def compare_entries(
    baseline: HistoryEntry, current: HistoryEntry, threshold: float = 0.03
) -> list[MetricDelta]:
    """Diff every metric the two entries have in common."""
    deltas = []
    common_metrics = sorted(set(baseline.metrics) & set(current.metrics))
    for name in common_metrics:
        base_val = baseline.metrics[name]
        cur_val = current.metrics[name]
        delta = cur_val - base_val
        severity = _severity_label(delta)
        deltas.append(
            MetricDelta(
                benchmark=current.benchmark,
                metric_name=name,
                baseline_model=baseline.model,
                current_model=current.model,
                baseline_value=base_val,
                current_value=cur_val,
                delta=delta,
                severity=severity,
                flagged=delta < -threshold,
            )
        )
    return sorted(deltas, key=lambda d: d.delta)


def compare_models(
    store: HistoryStore,
    baseline_model: str,
    current_model: str,
    benchmarks: list[str] | None = None,
    threshold: float = 0.03,
) -> list[MetricDelta]:
    """Diff the latest tracked run per benchmark between two models."""
    if benchmarks is None:
        benchmarks = sorted(
            set(store.benchmarks_for(baseline_model)) & set(store.benchmarks_for(current_model))
        )

    deltas: list[MetricDelta] = []
    for bench in benchmarks:
        baseline_entry = store.latest(baseline_model, bench)
        current_entry = store.latest(current_model, bench)
        if baseline_entry is None or current_entry is None:
            continue
        deltas.extend(compare_entries(baseline_entry, current_entry, threshold))

    return sorted(deltas, key=lambda d: d.delta)
