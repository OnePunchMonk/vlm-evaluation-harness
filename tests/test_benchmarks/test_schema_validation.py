"""Tests for BenchmarkManifest.validate() — the guard against silently
scoring a benchmark whose manifest can never produce a real number."""

import pytest

from vlm_evaluation_harness.benchmarks.schema import BenchmarkManifest, ManifestError, MetricConfig


def _base_manifest(**overrides):
    data = {
        "name": "Test",
        "source": {"type": "local", "path": "/tmp/x"},
        "splits": [{"name": "validation", "scorable": True}],
        "fields": {"question": "question", "answer": "answer"},
        "prompt_template": "{question}",
        "metrics": [{"type": "accuracy"}],
    }
    data.update(overrides)
    return BenchmarkManifest.from_dict(data)


def test_valid_manifest_passes():
    _base_manifest().validate()  # should not raise


def test_unresolvable_template_placeholder_raises():
    manifest = _base_manifest(prompt_template="{question} {caption_0}")
    with pytest.raises(ManifestError, match="caption_0"):
        manifest.validate()


def test_scorable_split_without_reference_field_raises():
    manifest = _base_manifest(fields={"question": "question", "answer": None})
    with pytest.raises(ManifestError, match="no fields.answer"):
        manifest.validate()


def test_unknown_metric_type_raises():
    manifest = _base_manifest()
    manifest.metrics = [MetricConfig(type="not_a_real_metric")]
    with pytest.raises(ManifestError, match="unknown metric type"):
        manifest.validate()


def test_accuracy_by_group_requires_group_field():
    manifest = _base_manifest()
    manifest.metrics = [MetricConfig(type="accuracy_by_group")]
    with pytest.raises(ManifestError, match="group_field"):
        manifest.validate()


def test_loglikelihood_scoring_requires_multiple_choice():
    manifest = _base_manifest(scoring="loglikelihood")
    with pytest.raises(ManifestError, match="loglikelihood"):
        manifest.validate()


def test_pairwise_matching_requires_second_template():
    manifest = _base_manifest(task_type="pairwise_matching")
    with pytest.raises(ManifestError, match="prompt_template_b"):
        manifest.validate()


def test_generative_manifest_does_not_require_answer_field():
    manifest = _base_manifest(
        task_type="text_to_image",
        fields={"question": "prompt", "answer": None},
        metrics=[{"type": "clip_score"}],
    )
    manifest.validate()  # should not raise


def test_invalid_regex_pattern_raises():
    manifest = _base_manifest(
        answer_extraction={"strategy": "regex", "regex_pattern": "["},
    )
    with pytest.raises(ManifestError, match="regex_pattern"):
        manifest.validate()
