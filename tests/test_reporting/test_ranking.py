"""Tests for significance-aware multi-model ranking."""

from vlm_evaluation_harness.reporting.ranking import rank_by_significance


def test_clearly_separated_models_land_in_different_tiers():
    # 20 samples, model B wrong on every one model A gets right -> highly
    # significant discordance.
    a_scores = {str(i): 1.0 for i in range(20)}
    b_scores = {str(i): 0.0 for i in range(20)}

    ranked = rank_by_significance(
        {"a": a_scores, "b": b_scores}, {"a": 1.0, "b": 0.0}
    )
    tiers = {r.model: r.tier for r in ranked}
    assert tiers["a"] == 0
    assert tiers["b"] == 1
    assert tiers["a"] != tiers["b"]


def test_indistinguishable_models_land_in_same_tier():
    # Same per-sample pattern -> zero discordant pairs -> not significant.
    scores = {str(i): float(i % 2) for i in range(20)}
    ranked = rank_by_significance(
        {"a": scores, "b": dict(scores)}, {"a": 0.5, "b": 0.5}
    )
    tiers = {r.model: r.tier for r in ranked}
    assert tiers["a"] == tiers["b"] == 0


def test_missing_per_sample_data_falls_back_to_one_tier():
    # No per-sample scores for either model: can't test significance, so
    # both stay in tier 0 rather than being silently reordered.
    ranked = rank_by_significance({}, {"a": 0.9, "b": 0.2})
    assert {r.tier for r in ranked} == {0}
    assert [r.model for r in ranked] == ["a", "b"]  # ordered by value


def test_three_models_two_tiers():
    high = {str(i): 1.0 for i in range(20)}
    low = {str(i): 0.0 for i in range(20)}
    ranked = rank_by_significance(
        {"a": high, "b": high, "c": low},
        {"a": 1.0, "b": 1.0, "c": 0.0},
    )
    tiers = {r.model: r.tier for r in ranked}
    assert tiers["a"] == tiers["b"] == 0
    assert tiers["c"] == 1
