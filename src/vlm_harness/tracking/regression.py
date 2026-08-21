"""Regression detection: diff two tracked runs and flag statistically real drops.

The previous implementation classified every delta by fixed absolute
thresholds (10%/5%/3%/1%) regardless of sample count. On 50 samples a 4-point
swing is ordinary noise; labelling it "MEDIUM" manufactures false alarms.

When both runs have per-sample scores (recorded by `HistoryStore` alongside
the aggregate metrics), the comparison uses a paired McNemar test — the
statistically correct tool for "did the same samples flip" — plus a
bootstrap confidence interval on the paired delta. Severity is then driven by
significance, not just magnitude. When only aggregate scores are available
(e.g. history recorded before this change), it falls back to the old
threshold-based label and says so explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from vlm_harness.stats import McNemarResult, bootstrap_delta_ci, mcnemar
from vlm_harness.tracking.history import HistoryEntry, HistoryStore

# Fallback thresholds, used only when per-sample data is unavailable.
SEVERITY_CRITICAL = -0.10
SEVERITY_HIGH = -0.05
SEVERITY_MEDIUM = -0.03
SEVERITY_LOW = -0.01


def _magnitude_severity(delta: float) -> str:
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


def _significance_severity(delta: float, mcnemar_result: McNemarResult) -> str:
    if not mcnemar_result.significant:
        return "OK" if delta >= 0 else "NOT_SIGNIFICANT"
    if delta >= 0:
        return "OK"
    if delta <= SEVERITY_CRITICAL:
        return "CRITICAL"
    if delta <= SEVERITY_HIGH:
        return "HIGH"
    return "MEDIUM"


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
    # Present only when per-sample scores were available for both runs.
    mcnemar: McNemarResult | None = None
    delta_ci95: tuple[float, float] | None = None
    method: str = "magnitude_threshold"

    @property
    def is_regression(self) -> bool:
        return self.delta < 0 and not math.isnan(self.delta)

    def summary(self) -> str:
        base = (
            f"{self.benchmark}/{self.metric_name}: "
            f"{self.baseline_value:.4f} -> {self.current_value:.4f} "
            f"({self.delta:+.4f}, {self.severity})"
        )
        if self.mcnemar is not None:
            base += f" [{self.mcnemar.summary()}]"
        return base


def compare_entries(
    baseline: HistoryEntry,
    current: HistoryEntry,
    threshold: float = 0.03,
    store: HistoryStore | None = None,
) -> list[MetricDelta]:
    """Diff every metric the two entries have in common.

    When `store` is given and both runs recorded per-sample scores for a
    metric, the comparison is a paired McNemar test; otherwise it falls back
    to a plain magnitude threshold on the aggregate values.
    """
    deltas = []
    common_metrics = sorted(set(baseline.metrics) & set(current.metrics))

    baseline_samples = store.per_sample_scores(baseline.run_id) if store else {}
    current_samples = store.per_sample_scores(current.run_id) if store else {}

    for name in common_metrics:
        base_val = baseline.metrics[name]
        cur_val = current.metrics[name]
        delta = cur_val - base_val

        base_per_sample = baseline_samples.get(name)
        cur_per_sample = current_samples.get(name)

        if base_per_sample and cur_per_sample:
            result = mcnemar(base_per_sample, cur_per_sample)
            severity = _significance_severity(delta, result)
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
                    flagged=result.significant and delta < 0,
                    mcnemar=result,
                    delta_ci95=bootstrap_delta_ci(base_per_sample, cur_per_sample),
                    method="mcnemar",
                )
            )
        else:
            deltas.append(
                MetricDelta(
                    benchmark=current.benchmark,
                    metric_name=name,
                    baseline_model=baseline.model,
                    current_model=current.model,
                    baseline_value=base_val,
                    current_value=cur_val,
                    delta=delta,
                    severity=_magnitude_severity(delta),
                    flagged=delta < -threshold,
                    method="magnitude_threshold",
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
        deltas.extend(compare_entries(baseline_entry, current_entry, threshold, store=store))

    return sorted(deltas, key=lambda d: d.delta)
