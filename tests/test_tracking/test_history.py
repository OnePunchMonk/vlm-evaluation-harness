"""Tests for the run history store."""

from vlm_harness.tracking.history import HistoryStore


def test_record_and_query_roundtrip(tmp_path):
    store = HistoryStore(path=tmp_path / "history.jsonl")
    store.record(
        model="mock:v1", benchmark="DemoMC", split="validation",
        metrics={"accuracy": 0.5}, n_samples=12,
    )
    entries = store.query(model="mock:v1")
    assert len(entries) == 1
    assert entries[0].benchmark == "DemoMC"
    assert entries[0].metrics == {"accuracy": 0.5}


def test_query_filters_by_model_and_benchmark(tmp_path):
    store = HistoryStore(path=tmp_path / "history.jsonl")
    store.record(model="mock:v1", benchmark="A", split="s", metrics={"accuracy": 0.1}, n_samples=1)
    store.record(model="mock:v1", benchmark="B", split="s", metrics={"accuracy": 0.2}, n_samples=1)
    store.record(model="mock:v2", benchmark="A", split="s", metrics={"accuracy": 0.3}, n_samples=1)

    assert len(store.query(model="mock:v1")) == 2
    assert len(store.query(benchmark="A")) == 2
    assert len(store.query(model="mock:v1", benchmark="A")) == 1


def test_latest_returns_most_recent(tmp_path):
    store = HistoryStore(path=tmp_path / "history.jsonl")
    store.record(model="mock:v1", benchmark="A", split="s", metrics={"accuracy": 0.1}, n_samples=1)
    store.record(model="mock:v1", benchmark="A", split="s", metrics={"accuracy": 0.9}, n_samples=1)

    latest = store.latest("mock:v1", "A")
    assert latest.metrics == {"accuracy": 0.9}


def test_latest_returns_none_when_missing(tmp_path):
    store = HistoryStore(path=tmp_path / "history.jsonl")
    assert store.latest("nope", "nothing") is None


def test_benchmarks_for(tmp_path):
    store = HistoryStore(path=tmp_path / "history.jsonl")
    store.record(model="mock:v1", benchmark="A", split="s", metrics={}, n_samples=1)
    store.record(model="mock:v1", benchmark="B", split="s", metrics={}, n_samples=1)
    assert store.benchmarks_for("mock:v1") == ["A", "B"]


def test_all_on_missing_file_returns_empty(tmp_path):
    store = HistoryStore(path=tmp_path / "does_not_exist.jsonl")
    assert store.all() == []


def test_record_result_uses_result_to_dict(tmp_path):
    class FakeResult:
        def to_dict(self):
            return {"model": "mock:v1", "benchmark": "A", "split": "s",
                     "metrics": {"accuracy": 0.5}, "n_samples": 3}

    store = HistoryStore(path=tmp_path / "history.jsonl")
    entry = store.record_result(FakeResult(), modality="generative")
    assert entry.modality == "generative"
    assert entry.metrics == {"accuracy": 0.5}
