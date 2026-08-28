"""Tests for LLM/VLM-as-judge and VQAScore, using the offline mock adapter."""

from PIL import Image

from vlm_evaluation_harness.metrics.generative.judge import LLMJudgeMetric, VQAScoreMetric


def test_judge_produces_score_in_range():
    metric = LLMJudgeMetric(
        judge_model="mock:judge-v1",
        rubric="Rate prompt-image alignment.",
        max_score=10,
    )
    images = [Image.new("RGB", (32, 32), color=(200, 20, 20))]
    result = metric.compute(["a red square"], images, sample_ids=["0"])
    assert result.metric_name == "llm_judge"
    assert 0.0 <= result.value <= 1.0
    assert result.n_samples == 1
    assert result.n_scored == 1


def test_judge_is_deterministic_for_repeated_calls():
    metric = LLMJudgeMetric("mock:judge-v1", "Rate the image.", max_score=10)
    images = [Image.new("RGB", (16, 16))]
    r1 = metric.compute(["a prompt"], images, sample_ids=["0"])
    r2 = metric.compute(["a prompt"], images, sample_ids=["0"])
    assert r1.value == r2.value


def test_vqa_score_in_unit_interval():
    metric = VQAScoreMetric(judge_model="mock:t2i-v1")
    images = [Image.new("RGB", (16, 16))]
    result = metric.compute(["a red square"], images, sample_ids=["0"])
    assert result.metric_name == "vqa_score"
    assert 0.0 <= result.value <= 1.0
    assert result.n_scored == 1
