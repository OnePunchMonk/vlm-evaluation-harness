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
        assert s.references and s.references[0] in {"A", "B", "C", "D", "E", "F"}
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


def test_every_builtin_manifest_passes_validation():
    """Regression guard for the Winoground bug this harness used to ship:
    a manifest whose prompt template referenced fields the loader could
    never populate, silently sending literal '{caption_0}' to the model.
    `validate()` now runs at load time, so a manifest like that can never
    reach the registry in the first place.
    """
    registry = get_registry()
    assert registry.errors() == {}


def test_winoground_prompt_resolves_all_placeholders():
    registry = get_registry()
    manifest = registry.get("winoground")
    missing = manifest.template_variables() - manifest.available_variables()
    assert missing == set()


def test_comp_hardneg_loads_four_way_choices():
    from vlm_harness.benchmarks.loader import BenchmarkLoader

    registry = get_registry()
    manifest = registry.get("comp_hardneg")
    loader = BenchmarkLoader()
    samples = list(loader.load(manifest, split="validation"))

    assert len(samples) == 8
    for s in samples:
        choices = s.text_fields["choices"]
        assert len(choices) == 4
        assert len(set(choices)) == 4  # correct + 3 distinct hard negatives
        assert s.references and s.references[0] in {"A", "B", "C", "D"}


def test_hallu_fg_loads_category_metadata():
    from vlm_harness.benchmarks.loader import BenchmarkLoader

    registry = get_registry()
    manifest = registry.get("hallu_fg")
    loader = BenchmarkLoader()
    samples = list(loader.load(manifest, split="validation"))

    assert samples
    categories = {s.metadata["hallu_category"] for s in samples}
    assert categories == {"object", "attribute", "relation"}
    for s in samples:
        assert s.references[0] in {"yes", "no"}


def test_calib_deflect_loads_answerable_flag():
    from vlm_harness.benchmarks.loader import BenchmarkLoader

    registry = get_registry()
    manifest = registry.get("calib_deflect")
    loader = BenchmarkLoader()
    samples = list(loader.load(manifest, split="validation"))

    assert samples
    answerable_flags = {s.metadata["answerable"] for s in samples}
    assert answerable_flags == {True, False}


def test_comp_hardneg_hallu_fg_calib_deflect_run_offline():
    """End-to-end smoke test for all three new benchmarks against the mock
    adapter — proves the manifest + fixture + metric wiring is consistent,
    not just that the YAML parses."""
    from vlm_harness.adapters.mock import MockAdapter
    from vlm_harness.engine.runner import EvalConfig, EvalRunner

    for bench in ("comp_hardneg", "hallu_fg", "calib_deflect"):
        adapter = MockAdapter(model_id="offline-demo")
        runner = EvalRunner(adapter)
        config = EvalConfig(
            model_spec="mock:offline-demo", benchmark=bench, split="validation", use_cache=False
        )
        result = runner.run(config)
        assert result.sample_results
        assert result.metrics
