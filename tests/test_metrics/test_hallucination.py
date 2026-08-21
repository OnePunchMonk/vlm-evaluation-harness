"""Tests for hallucination metrics."""

from vlm_harness.metrics.base import ScoredSample
from vlm_harness.metrics.hallucination import CHAIRMetric, POPEMetric


class TestCHAIR:
    metric = CHAIRMetric(objects_field="objects")

    def _sample(self, caption, objects):
        return ScoredSample(
            sample_id="0", prediction=caption, references=[], metadata={"objects": objects}
        )

    def test_no_hallucination(self):
        r = self.metric.compute([self._sample("I see a cat and a dog.", ["cat", "dog"])])
        assert r.breakdown["chair_s"] == 0.0
        assert r.breakdown["chair_i"] == 0.0

    def test_full_hallucination(self):
        r = self.metric.compute([self._sample("There is a bicycle and a car.", [])])
        assert r.breakdown["chair_s"] > 0.0
        assert r.breakdown["chair_i"] == 1.0

    def test_partial_hallucination(self):
        r = self.metric.compute([self._sample("I see a cat and a bicycle.", ["cat"])])
        assert 0 < r.breakdown["chair_s"] < 1.0

    def test_empty_caption(self):
        r = self.metric.compute([self._sample("", ["cat"])])
        assert r.breakdown["chair_s"] == 0.0

    def test_sample_without_objects_field_is_excluded(self):
        untagged = ScoredSample(sample_id="0", prediction="a cat", references=[], metadata={})
        r = self.metric.compute([untagged])
        assert r.n_scored == 0


class TestPOPE:
    metric = POPEMetric()

    def _sample(self, sample_id, prediction, reference):
        return ScoredSample(sample_id=sample_id, prediction=prediction, references=[reference])

    def test_all_correct(self):
        samples = [
            self._sample("0", "Yes", "yes"),
            self._sample("1", "No", "no"),
            self._sample("2", "Yes", "yes"),
        ]
        r = self.metric.compute(samples)
        assert r.value == 1.0
        assert r.breakdown["f1"] == 1.0

    def test_all_wrong(self):
        samples = [self._sample("0", "No", "yes"), self._sample("1", "Yes", "no")]
        r = self.metric.compute(samples)
        assert r.value == 0.0

    def test_yes_rate(self):
        samples = [
            self._sample("0", "Yes", "yes"),
            self._sample("1", "Yes", "no"),
            self._sample("2", "No", "no"),
        ]
        r = self.metric.compute(samples)
        assert abs(r.breakdown["yes_rate"] - 2 / 3) < 1e-6
