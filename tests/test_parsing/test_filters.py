"""Tests for the composable post-extraction answer filter pipeline."""

from __future__ import annotations

import pytest

from vlm_evaluation_harness.benchmarks.schema import AnswerExtractionConfig, BenchmarkManifest
from vlm_evaluation_harness.parsing.extractor import AnswerExtractor
from vlm_evaluation_harness.parsing.filters import apply_filters


def test_apply_filters_runs_in_order():
    # strip_punctuation before lowercase vs. after would both work here, but
    # ordering does matter for e.g. collapse_whitespace after strip_punctuation
    # turning "a,  b" into "a b" only if punctuation is stripped first.
    result = apply_filters("A,  B!", ["strip_punctuation", "collapse_whitespace", "lowercase"])
    assert result == "a b"


def test_apply_filters_empty_list_is_noop():
    assert apply_filters("Hello World", []) == "Hello World"


def test_unknown_filter_raises():
    with pytest.raises(ValueError, match="unknown answer filter"):
        apply_filters("x", ["not_a_real_filter"])


def test_extractor_applies_filters_after_normalize():
    extractor = AnswerExtractor()
    config = AnswerExtractionConfig(
        strategy="exact", normalize="none", filters=["lowercase", "strip_punctuation"]
    )
    result = extractor.extract("  Cat! ", config)
    assert result.normalized == "cat"


def test_extractor_with_no_filters_is_unchanged_from_normalize_only():
    extractor = AnswerExtractor()
    config = AnswerExtractionConfig(strategy="exact", normalize="lowercase")
    result = extractor.extract("HELLO", config)
    assert result.normalized == "hello"


def test_manifest_validates_unknown_filter_name():
    manifest = BenchmarkManifest.from_dict(
        {
            "name": "BadFilters",
            "source": {"type": "local", "path": "."},
            "splits": [{"name": "validation", "scorable": True}],
            "fields": {"question": "question", "answer": "answer"},
            "answer_extraction": {"strategy": "exact", "filters": ["nonexistent"]},
        }
    )
    with pytest.raises(Exception, match="unknown answer_extraction.filters"):
        manifest.validate()


def test_manifest_accepts_known_filters():
    manifest = BenchmarkManifest.from_dict(
        {
            "name": "GoodFilters",
            "source": {"type": "local", "path": "."},
            "splits": [{"name": "validation", "scorable": True}],
            "fields": {"question": "question", "answer": "answer"},
            "answer_extraction": {
                "strategy": "exact",
                "filters": ["lowercase", "strip_whitespace"],
            },
        }
    )
    manifest.validate()  # must not raise
