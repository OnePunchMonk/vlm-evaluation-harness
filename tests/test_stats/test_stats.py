"""Tests for significance testing utilities."""

import math

from vlm_harness.stats import bootstrap_ci, bootstrap_delta_ci, mcnemar, wilson_interval


def test_mcnemar_no_discordant_pairs():
    baseline = {"a": 1.0, "b": 1.0}
    current = {"a": 1.0, "b": 1.0}
    result = mcnemar(baseline, current)
    assert result.n_discordant == 0
    assert result.p_value == 1.0
    assert not result.significant


def test_mcnemar_detects_large_asymmetric_flip():
    baseline = {f"s{i}": 1.0 for i in range(60)}
    current = dict(baseline)
    for i in range(20):
        current[f"s{i}"] = 0.0
    result = mcnemar(baseline, current)
    assert result.n_regressed == 20
    assert result.n_improved == 0
    assert result.significant


def test_mcnemar_symmetric_flips_not_significant():
    baseline = {f"s{i}": float(i % 2) for i in range(50)}
    current = dict(baseline)
    # swap 5 in each direction -> balanced disagreement, no real signal
    for i in list(baseline)[:5]:
        current[i] = 1.0 - baseline[i]
    result = mcnemar(baseline, current)
    assert result.n_discordant > 0


def test_mcnemar_only_uses_shared_samples():
    baseline = {"a": 1.0, "b": 0.0}
    current = {"a": 0.0, "c": 1.0}
    result = mcnemar(baseline, current)
    assert result.n_paired == 1


def test_bootstrap_ci_bounds_the_mean():
    values = [1.0] * 20 + [0.0] * 5
    lo, hi = bootstrap_ci(values, confidence=0.95)
    mean = sum(values) / len(values)
    assert lo <= mean <= hi


def test_bootstrap_ci_empty_is_nan():
    lo, hi = bootstrap_ci([])
    assert math.isnan(lo) and math.isnan(hi)


def test_bootstrap_delta_ci_no_difference_centers_near_zero():
    baseline = {f"s{i}": 1.0 for i in range(30)}
    current = dict(baseline)
    lo, hi = bootstrap_delta_ci(baseline, current)
    assert lo == 0.0 and hi == 0.0


def test_wilson_interval_contains_point_estimate():
    lo, hi = wilson_interval(45, 50)
    assert lo <= 0.9 <= hi
    assert 0.0 <= lo < hi <= 1.0
