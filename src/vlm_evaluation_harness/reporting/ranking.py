"""Significance-aware ranking for a 3+ model comparison.

Sorting models by raw metric value treats a 0.2-point difference on 50
samples the same as a 20-point difference — most of the time the former is
noise. This groups models into tiers using the same paired McNemar test
`tracking/regression.py` uses for baseline-vs-current: model A ranks
strictly above model B only when their per-sample scores differ
significantly; otherwise they land in the same tier, ordered by point value
within it but not claimed to be meaningfully different.

Works on plain per-sample-score dicts (`{sample_id: score}`) rather than any
particular result class, so it can rank live `EvalResult`s from `compare`,
tracked `HistoryEntry`s, or anything else with per-sample scores.
"""

from __future__ import annotations

from dataclasses import dataclass

from vlm_evaluation_harness.stats import mcnemar


@dataclass
class RankedModel:
    model: str
    value: float
    tier: int  # 0 = best tier. Ties within a tier are not significant.


def rank_by_significance(
    per_model_scores: dict[str, dict[str, float]],
    per_model_values: dict[str, float],
    alpha: float = 0.05,
) -> list[RankedModel]:
    """Group models into significance tiers for one metric, best tier first.

    `per_model_scores` maps model name -> {sample_id: score}. `per_model_values`
    maps model name -> the metric's aggregate value (used to order models
    within a tier and to decide which of a pair is "ahead").
    """
    models = sorted(per_model_values, key=lambda m: per_model_values[m], reverse=True)
    tiers: list[list[str]] = []

    for model in models:
        placed = False
        for tier in tiers:
            # A model joins an existing tier only if it is NOT significantly
            # different from every member already in it.
            if all(
                not _significantly_different(
                    per_model_scores.get(model), per_model_scores.get(other), alpha
                )
                for other in tier
            ):
                tier.append(model)
                placed = True
                break
        if not placed:
            tiers.append([model])

    ranked = []
    for tier_idx, tier in enumerate(tiers):
        for model in tier:
            ranked.append(RankedModel(model=model, value=per_model_values[model], tier=tier_idx))
    return ranked


def _significantly_different(
    a: dict[str, float] | None, b: dict[str, float] | None, alpha: float
) -> bool:
    if not a or not b:
        # No per-sample data for one side: cannot claim significance, so
        # treat as not-significantly-different (conservative: don't split
        # a tier on data we can't test).
        return False
    result = mcnemar(a, b)
    return result.p_value < alpha


def rank_eval_results(
    results: list, metric_name: str | None = None, alpha: float = 0.05
) -> list[RankedModel]:
    """Rank a list of `EvalResult`-like objects on one metric.

    Duck-typed on `result.config.model_spec` and `result.metrics` (a list of
    `MetricResult`-like objects with `.metric_name`, `.value`, `.per_sample`)
    to avoid a hard dependency on `engine.runner.EvalResult`. Defaults to the
    first metric that has per-sample scores (excluding
    `extraction_failure_rate`, matching `EvalResult.primary_metric`).
    """
    per_model_scores: dict[str, dict[str, float]] = {}
    per_model_values: dict[str, float] = {}

    for result in results:
        model = result.config.model_spec
        metric = _select_metric(result.metrics, metric_name)
        if metric is None:
            continue
        per_model_values[model] = metric.value
        if metric.per_sample:
            per_model_scores[model] = metric.per_sample

    if not per_model_values:
        return []

    return rank_by_significance(per_model_scores, per_model_values, alpha)


def _select_metric(metrics: list, metric_name: str | None):
    if metric_name is not None:
        for m in metrics:
            if m.metric_name == metric_name:
                return m
        return None
    for m in metrics:
        if m.per_sample and m.metric_name != "extraction_failure_rate":
            return m
    return metrics[0] if metrics else None
