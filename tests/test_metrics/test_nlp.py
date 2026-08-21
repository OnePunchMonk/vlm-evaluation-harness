"""Tests for NLP metrics."""

import math

from vlm_harness.metrics.base import ScoredSample
from vlm_harness.metrics.nlp import ANLSMetric, F1Metric, RougeMetric


def one(prediction, reference):
    return ScoredSample(sample_id="0", prediction=prediction, references=[reference])


class TestF1Metric:
    metric = F1Metric()

    def test_perfect(self):
        r = self.metric.compute([one("hello world", "hello world")])
        assert r.value == 1.0

    def test_no_overlap(self):
        r = self.metric.compute([one("cat", "dog")])
        assert r.value == 0.0

    def test_partial(self):
        r = self.metric.compute([one("the quick fox", "the slow fox")])
        assert 0 < r.value < 1.0

    def test_case_insensitive(self):
        r = self.metric.compute([one("Hello", "hello")])
        assert r.value == 1.0


class TestANLSMetric:
    metric = ANLSMetric()

    def test_exact(self):
        r = self.metric.compute([one("Paris", "Paris")])
        assert r.value == 1.0

    def test_slight_diff(self):
        r = self.metric.compute([one("Pariss", "Paris")])
        assert r.value > 0.5

    def test_completely_wrong(self):
        r = self.metric.compute([one("London", "Paris")])
        assert r.value == 0.0

    def test_empty(self):
        r = self.metric.compute([])
        assert math.isnan(r.value)

    def test_best_of_multiple_references(self):
        s = ScoredSample(sample_id="0", prediction="Paris", references=["London", "Paris"])
        r = self.metric.compute([s])
        assert r.value == 1.0


class TestRougeMetric:
    metric = RougeMetric()

    def test_identical(self):
        r = self.metric.compute([one("the cat sat on the mat", "the cat sat on the mat")])
        assert r.value == 1.0

    def test_no_overlap(self):
        r = self.metric.compute([one("xyz", "abc")])
        assert r.value == 0.0
