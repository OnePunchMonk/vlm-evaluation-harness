"""Tests for the offline demo/fixture benchmarks (no network or API keys)."""

from vlm_harness.benchmarks.registry import get_registry


def test_all_offline_manifests_load():
    registry = get_registry()
    for name in ("demo_mc", "geneval_mini", "clipscore_mini", "genjudge_mini"):
        manifest = registry.get(name)
        assert manifest.name

    by_cat = registry.list_by_category()
    assert "generative" in by_cat
    assert "GenEvalMini" in by_cat["generative"]


def test_demo_mc_loads_local_fixture_images():
    from vlm_harness.benchmarks.loader import BenchmarkLoader

    registry = get_registry()
    manifest = registry.get("demo_mc")
    loader = BenchmarkLoader()
    samples = list(loader.load(manifest, split="validation"))

    assert len(samples) == 12
    for s in samples:
        assert len(s.images) == 1
        assert s.answer in {"A", "B", "C", "D", "E", "F"}
        assert s.metadata["subject"] == "color"


def test_geneval_mini_loads_structured_checks():
    from vlm_harness.benchmarks.loader import BenchmarkLoader

    registry = get_registry()
    manifest = registry.get("geneval_mini")
    loader = BenchmarkLoader()
    samples = list(loader.load(manifest, split="prompts"))

    assert len(samples) == 12
    for s in samples:
        assert s.images == []
        assert s.text_fields["question"]  # the prompt
        checks = s.metadata["checks"]
        assert {"count", "color", "shape"} <= checks.keys()


def test_full_offline_discriminative_run():
    from vlm_harness.adapters.mock import MockAdapter
    from vlm_harness.engine.runner import EvalConfig, EvalRunner

    adapter = MockAdapter(model_id="offline-demo")
    runner = EvalRunner(adapter)
    config = EvalConfig(model_spec="mock:offline-demo", benchmark="demo_mc", split="validation")
    result = runner.run(config)

    assert len(result.sample_results) == 12
    accuracy = next(m for m in result.metrics if m.metric_name == "accuracy")
    assert 0.0 <= accuracy.value <= 1.0
