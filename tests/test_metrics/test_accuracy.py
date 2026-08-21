"""Tests for accuracy metrics."""

import math

from vlm_harness.metrics.accuracy import AccuracyMetric, RelaxedAccuracyMetric, VQAAccuracyMetric
from vlm_harness.metrics.base import ScoredSample

metric = AccuracyMetric()


def samples(predictions, references, metadata=None):
    metadata = metadata or [{}] * len(predictions)
    return [
        ScoredSample(
            sample_id=str(i),
            prediction=p,
            references=[r] if r is not None else [],
            metadata=m,
        )
        for i, (p, r, m) in enumerate(zip(predictions, references, metadata))
    ]


def test_perfect_accuracy():
    result = metric.compute(samples(["A", "B", "C"], ["A", "B", "C"]))
    assert result.value == 1.0
    assert result.n_samples == 3
    assert result.n_scored == 3


def test_zero_accuracy():
    result = metric.compute(samples(["A", "B", "C"], ["D", "E", "F"]))
    assert result.value == 0.0


def test_partial_accuracy():
    result = metric.compute(samples(["A", "B", "X"], ["A", "B", "C"]))
    assert abs(result.value - 2 / 3) < 1e-6


def test_accuracy_by_group():
    preds = ["A", "B", "C", "D"]
    refs = ["A", "X", "C", "X"]
    meta = [{"subject": "math"}, {"subject": "math"}, {"subject": "art"}, {"subject": "art"}]
    result = metric.compute_by_group(samples(preds, refs, meta), "subject")
    assert result.breakdown["math"] == 0.5
    assert result.breakdown["art"] == 0.5
    assert result.metric_name == "accuracy_by_subject"


def test_empty_inputs():
    # No scorable samples -> NaN, not 0.0. A benchmark that produced zero
    # scorable samples must not look identical to a model that scored zero.
    result = metric.compute([])
    assert math.isnan(result.value)
    assert result.n_samples == 0


def test_sample_with_no_reference_is_excluded():
    scored = samples(["A", "B"], ["A", None])
    result = metric.compute(scored)
    assert result.n_samples == 2
    assert result.n_scored == 1
    assert result.value == 1.0


def test_vqa_accuracy_consensus():
    metric = VQAAccuracyMetric()
    # 3 of 10 annotators say "cat" -> fully correct (min(3/3, 1) == 1.0)
    s = ScoredSample(sample_id="0", prediction="cat", references=["cat"] * 3 + ["dog"] * 7)
    assert metric.compute([s]).value == 1.0

    # only 1 annotator agrees -> partial credit
    s2 = ScoredSample(sample_id="1", prediction="cat", references=["cat"] + ["dog"] * 9)
    result = metric.compute([s2])
    assert abs(result.value - 1 / 3) < 1e-6


def test_relaxed_accuracy_numeric_tolerance():
    metric = RelaxedAccuracyMetric(tolerance=0.05)
    within = ScoredSample(sample_id="0", prediction="104", references=["100"])
    outside = ScoredSample(sample_id="1", prediction="120", references=["100"])
    result = metric.compute([within, outside])
    assert metric.score(within) == 1.0
    assert metric.score(outside) == 0.0
    assert result.value == 0.5
