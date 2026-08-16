"""End-to-end test of the generative eval pipeline using the offline mock T2I
adapter and the zero-dependency judge-only benchmark (GenJudgeMini)."""

from vlm_harness.adapters.generative.mock import MockT2IAdapter
from vlm_harness.engine.generative_runner import GenerativeEvalRunner, GenEvalConfig


def test_full_offline_generative_run(tmp_path):
    adapter = MockT2IAdapter(model_id="offline-demo")
    runner = GenerativeEvalRunner(adapter)
    config = GenEvalConfig(
        model_spec="mock:offline-demo",
        benchmark="genjudge_mini",
        max_samples=3,
        output_dir=tmp_path,
    )

    result = runner.run(config)

    assert len(result.sample_results) == 3
    assert len(result.images) == 3
    assert result.metrics[0].metric_name == "llm_judge"
    assert 0.0 <= result.metrics[0].value <= 1.0

    saved = list(tmp_path.glob("*_results.json"))
    assert len(saved) == 1
    import json
    data = json.loads(saved[0].read_text())
    assert data["benchmark"] == "GenJudgeMini"
    assert data["n_samples"] == 3
    assert "llm_judge" in data["metrics"]

    image_files = list(tmp_path.glob("*_images/*.png"))
    assert len(image_files) == 3
