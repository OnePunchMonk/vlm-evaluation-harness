"""Tests for regression detection and severity classification."""

from vlm_evaluation_harness.tracking.history import HistoryEntry
from vlm_evaluation_harness.tracking.regression import compare_entries, compare_models


def _entry(model, benchmark, metrics):
    return HistoryEntry(
        run_id="r", timestamp="t", model=model, benchmark=benchmark, split="s",
        modality="discriminative", metrics=metrics, n_samples=10,
    )


def test_no_change_is_ok_and_not_flagged():
    base = _entry("m1", "B", {"accuracy": 0.80})
    cur = _entry("m2", "B", {"accuracy": 0.80})
    deltas = compare_entries(base, cur)
    assert deltas[0].severity == "OK"
    assert deltas[0].flagged is False


def test_small_drop_below_threshold_is_not_flagged():
    base = _entry("m1", "B", {"accuracy": 0.80})
    cur = _entry("m2", "B", {"accuracy": 0.795})  # -0.5%
    deltas = compare_entries(base, cur, threshold=0.03)
    assert deltas[0].flagged is False
    assert deltas[0].severity in {"MINIMAL", "LOW"}


def test_large_drop_is_critical_and_flagged():
    base = _entry("m1", "B", {"accuracy": 0.80})
    cur = _entry("m2", "B", {"accuracy": 0.60})  # -20%
    deltas = compare_entries(base, cur, threshold=0.03)
    assert deltas[0].severity == "CRITICAL"
    assert deltas[0].flagged is True
    assert deltas[0].is_regression is True


def test_improvement_is_not_a_regression():
    base = _entry("m1", "B", {"accuracy": 0.60})
    cur = _entry("m2", "B", {"accuracy": 0.80})
    deltas = compare_entries(base, cur)
    assert deltas[0].is_regression is False
    assert deltas[0].flagged is False


def test_only_common_metrics_are_compared():
    base = _entry("m1", "B", {"accuracy": 0.8, "f1": 0.7})
    cur = _entry("m2", "B", {"accuracy": 0.7})
    deltas = compare_entries(base, cur)
    assert len(deltas) == 1
    assert deltas[0].metric_name == "accuracy"


def test_sorted_worst_first():
    base = _entry("m1", "B", {"a": 0.9, "b": 0.9})
    cur = _entry("m2", "B", {"a": 0.85, "b": 0.5})
    deltas = compare_entries(base, cur)
    assert deltas[0].metric_name == "b"


def test_compare_models_uses_latest_tracked_runs(tmp_path):
    from vlm_evaluation_harness.tracking.history import HistoryStore

    store = HistoryStore(path=tmp_path / "history.jsonl")
    store.record(model="base", benchmark="X", split="s", metrics={"accuracy": 0.9}, n_samples=1)
    store.record(model="ft", benchmark="X", split="s", metrics={"accuracy": 0.5}, n_samples=1)

    deltas = compare_models(store, "base", "ft")
    assert len(deltas) == 1
    assert deltas[0].severity == "CRITICAL"
    assert deltas[0].flagged is True


def test_compare_models_skips_benchmarks_missing_from_either_side(tmp_path):
    from vlm_evaluation_harness.tracking.history import HistoryStore

    store = HistoryStore(path=tmp_path / "history.jsonl")
    store.record(model="base", benchmark="X", split="s", metrics={"accuracy": 0.9}, n_samples=1)
    store.record(model="base", benchmark="Y", split="s", metrics={"accuracy": 0.9}, n_samples=1)
    store.record(model="ft", benchmark="X", split="s", metrics={"accuracy": 0.5}, n_samples=1)

    deltas = compare_models(store, "base", "ft")
    assert {d.benchmark for d in deltas} == {"X"}
