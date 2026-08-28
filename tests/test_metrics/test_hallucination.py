"""Tests for hallucination metrics."""

from vlm_harness.metrics.base import ScoredSample
from vlm_harness.metrics.hallucination import (
    CHAIRMetric,
    FineGrainedHallucinationMetric,
    POPEMetric,
)


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


class TestFineGrainedHallucination:
    metric = FineGrainedHallucinationMetric(category_field="hallu_category")

    def _sample(self, sample_id, prediction, reference, category):
        return ScoredSample(
            sample_id=sample_id,
            prediction=prediction,
            references=[reference],
            metadata={"hallu_category": category},
        )

    def test_wrongly_claiming_presence_is_hallucination(self):
        r = self.metric.compute([self._sample("0", "Yes", "no", "object")])
        assert r.value == 1.0
        assert r.breakdown["object_hallucination_rate"] == 1.0

    def test_correctly_denying_presence_is_not_hallucination(self):
        r = self.metric.compute([self._sample("0", "No", "no", "object")])
        assert r.value == 0.0

    def test_missing_a_real_object_is_not_counted_as_hallucination(self):
        # Under-claiming (saying "no" when the answer is "yes") is a
        # different failure mode than fabrication and must not inflate the
        # hallucination rate.
        r = self.metric.compute([self._sample("0", "No", "yes", "object")])
        assert r.value == 0.0

    def test_breakdown_is_per_category(self):
        samples = [
            self._sample("0", "Yes", "no", "object"),
            self._sample("1", "No", "no", "attribute"),
            self._sample("2", "Yes", "no", "relation"),
        ]
        r = self.metric.compute(samples)
        assert r.breakdown["object_hallucination_rate"] == 1.0
        assert r.breakdown["attribute_hallucination_rate"] == 0.0
        assert r.breakdown["relation_hallucination_rate"] == 1.0

    def test_no_scorable_samples_is_nan(self):
        import math

        r = self.metric.compute([ScoredSample(sample_id="0", prediction="Yes", references=[])])
        assert math.isnan(r.value)
