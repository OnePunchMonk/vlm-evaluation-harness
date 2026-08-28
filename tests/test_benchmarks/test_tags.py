"""Tests for benchmark manifest tags and tag-based filtering."""

from __future__ import annotations

from vlm_evaluation_harness.benchmarks.registry import BenchmarkRegistry
from vlm_evaluation_harness.benchmarks.schema import BenchmarkManifest


def test_tags_defaults_to_empty_list():
    manifest = BenchmarkManifest.from_dict(
        {
            "name": "Untagged",
            "source": {"type": "local", "path": "."},
            "splits": ["validation"],
        }
    )
    assert manifest.tags == []


def test_tags_parsed_from_manifest_dict():
    manifest = BenchmarkManifest.from_dict(
        {
            "name": "Tagged",
            "source": {"type": "local", "path": "."},
            "splits": ["validation"],
            "tags": ["safety", "hallucination"],
        }
    )
    assert manifest.tags == ["safety", "hallucination"]


def test_shipped_safety_benchmarks_are_tagged():
    registry = BenchmarkRegistry()
    assert "safety" in set(registry.get("pope").tags)
    assert "safety" in set(registry.get("hallu_fg").tags)
    assert "safety" in set(registry.get("calib_deflect").tags)


def test_list_by_tags_matches_any_tag():
    registry = BenchmarkRegistry()
    safety_benches = set(registry.list_by_tags(["safety"]))
    assert {"POPE", "FineGrainedHallucination", "CalibrationDeflect"} <= safety_benches
    assert "DemoMC" not in safety_benches


def test_list_by_tags_multiple_tags_is_union():
    registry = BenchmarkRegistry()
    result = set(registry.list_by_tags(["safety", "compositional"]))
    assert "POPE" in result
    assert "Winoground" in result


def test_list_by_tags_unknown_tag_returns_empty():
    registry = BenchmarkRegistry()
    assert registry.list_by_tags(["no_such_tag"]) == []
