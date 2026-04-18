"""Tests for NLP metrics."""

import pytest
from vlm_harness.metrics.nlp import F1Metric, ANLSMetric, RougeMetric


class TestF1Metric:
    metric = F1Metric()

    def test_perfect(self):
        r = self.metric.compute(["hello world"], ["hello world"], [{}])
        assert r.value == 1.0

    def test_no_overlap(self):
        r = self.metric.compute(["cat"], ["dog"], [{}])
        assert r.value == 0.0

    def test_partial(self):
        r = self.metric.compute(["the quick fox"], ["the slow fox"], [{}])
        assert 0 < r.value < 1.0

    def test_case_insensitive(self):
        r = self.metric.compute(["Hello"], ["hello"], [{}])
        assert r.value == 1.0


class TestANLSMetric:
    metric = ANLSMetric()

    def test_exact(self):
        r = self.metric.compute(["Paris"], ["Paris"], [{}])
        assert r.value == 1.0

    def test_slight_diff(self):
        # One character difference — still above threshold
        r = self.metric.compute(["Pariss"], ["Paris"], [{}])
        assert r.value > 0.5

    def test_completely_wrong(self):
        r = self.metric.compute(["London"], ["Paris"], [{}])
        # Edit distance / max_len is too high → 0
        assert r.value == 0.0

    def test_empty(self):
        r = self.metric.compute([], [], [])
        assert r.value == 0.0


class TestRougeMetric:
    metric = RougeMetric()

    def test_identical(self):
        r = self.metric.compute(["the cat sat on the mat"], ["the cat sat on the mat"], [{}])
        assert r.value == 1.0

    def test_no_overlap(self):
        r = self.metric.compute(["xyz"], ["abc"], [{}])
        assert r.value == 0.0
