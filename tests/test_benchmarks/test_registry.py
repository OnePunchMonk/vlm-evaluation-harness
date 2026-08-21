"""Tests for the benchmark registry."""

import pytest

from vlm_harness.benchmarks.registry import BenchmarkRegistry


def test_registry_loads_builtins():
    registry = BenchmarkRegistry()
    names = registry.list()
    assert len(names) > 0
    # Check known benchmarks are present
    assert "MMMU" in names
    assert "VQAv2" in names
    assert "ChartQA" in names


def test_get_existing_benchmark():
    registry = BenchmarkRegistry()
    manifest = registry.get("mmmu")
    assert manifest.name == "MMMU"
    assert manifest.task_type == "multiple_choice"


def test_get_case_insensitive():
    registry = BenchmarkRegistry()
    m1 = registry.get("MMMU")
    m2 = registry.get("mmmu")
    assert m1.name == m2.name


def test_get_missing_raises():
    registry = BenchmarkRegistry()
    with pytest.raises(KeyError, match="not found"):
        registry.get("nonexistent_benchmark_xyz")


def test_list_by_category():
    registry = BenchmarkRegistry()
    by_cat = registry.list_by_category()
    assert isinstance(by_cat, dict)
    # Should have reasoning and perception categories from manifests
    assert len(by_cat) > 0


def test_3d_benchmark_loaded():
    registry = BenchmarkRegistry()
    manifest = registry.get("scanqa")
    assert manifest.modality == "3d"
    assert manifest.taxonomy_category == "3d_vision"


def test_cross_modal_benchmark_loaded():
    registry = BenchmarkRegistry()
    manifest = registry.get("winoground")
    assert manifest.modality == "cross_modal"
    assert manifest.task_type == "pairwise_matching"
    assert manifest.prompt_template_b is not None
    assert manifest.pairwise_answers == ["A", "B"]
