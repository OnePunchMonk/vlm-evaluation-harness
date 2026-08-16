"""Tests for LLM/VLM-as-judge scoring, using the offline mock adapter."""

from PIL import Image

from vlm_harness.metrics.generative.judge import LLMJudgeMetric


def test_judge_produces_score_in_range():
    metric = LLMJudgeMetric(
        judge_model="mock:judge-v1",
        rubric="Rate prompt-image alignment.",
        max_score=10,
    )
    images = [Image.new("RGB", (32, 32), color=(200, 20, 20))]
    result = metric.compute(["a red square"], images)
    assert result.metric_name == "llm_judge"
    assert 0.0 <= result.value <= 1.0
    assert result.n_samples == 1
    raw = result.metadata["raw_scores"][0]
    assert 1 <= raw <= 10


def test_judge_is_deterministic_for_repeated_calls():
    metric = LLMJudgeMetric("mock:judge-v1", "Rate the image.", max_score=10)
    images = [Image.new("RGB", (16, 16))]
    r1 = metric.compute(["a prompt"], images)
    r2 = metric.compute(["a prompt"], images)
    assert r1.value == r2.value
