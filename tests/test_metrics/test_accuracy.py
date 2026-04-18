"""Tests for accuracy metrics."""

import pytest
from vlm_harness.metrics.accuracy import AccuracyMetric


metric = AccuracyMetric()


def test_perfect_accuracy():
    result = metric.compute(["A", "B", "C"], ["A", "B", "C"], [{}, {}, {}])
    assert result.value == 1.0
    assert result.n_samples == 3


def test_zero_accuracy():
    result = metric.compute(["A", "B", "C"], ["D", "E", "F"], [{}, {}, {}])
    assert result.value == 0.0


def test_partial_accuracy():
    result = metric.compute(["A", "B", "X"], ["A", "B", "C"], [{}, {}, {}])
    assert abs(result.value - 2 / 3) < 1e-6


def test_accuracy_by_group():
    preds = ["A", "B", "C", "D"]
    refs =  ["A", "X", "C", "X"]
    meta = [{"subject": "math"}, {"subject": "math"}, {"subject": "art"}, {"subject": "art"}]
    result = metric.compute_by_group(preds, refs, meta, "subject")
    assert result.breakdown["math"] == 0.5
    assert result.breakdown["art"] == 0.5
    assert result.metric_name == "accuracy_by_subject"


def test_empty_inputs():
    result = metric.compute([], [], [])
    assert result.value == 0.0
    assert result.n_samples == 0
