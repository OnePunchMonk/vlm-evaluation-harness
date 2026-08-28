"""Tests for the refusal-calibration metric."""

from vlm_harness.metrics.base import ScoredSample
from vlm_harness.metrics.calibration import CalibrationMetric


class TestCalibration:
    metric = CalibrationMetric(answerable_field="answerable")

    def _sample(self, sample_id, prediction, reference, answerable):
        return ScoredSample(
            sample_id=sample_id,
            prediction=prediction,
            references=[reference],
            metadata={"answerable": answerable},
        )

    def test_correct_confident_answer_scores_1(self):
        r = self.metric.compute([self._sample("0", "red", "red", True)])
        assert r.value == 1.0
        assert r.breakdown["answerable_accuracy"] == 1.0

    def test_wrong_confident_answer_scores_0(self):
        r = self.metric.compute([self._sample("0", "blue", "red", True)])
        assert r.value == 0.0

    def test_deflecting_an_answerable_question_scores_0_and_is_underconfident(self):
        r = self.metric.compute(
            [self._sample("0", "I cannot determine this from the image.", "red", True)]
        )
        assert r.value == 0.0
        assert r.breakdown["underconfidence_rate"] == 1.0

    def test_deflecting_an_unanswerable_question_scores_1(self):
        r = self.metric.compute(
            [
                self._sample(
                    "0", "This cannot be determined from the image.", "unanswerable", False
                )
            ]
        )
        assert r.value == 1.0
        assert r.breakdown["unanswerable_deflection_rate"] == 1.0
        assert r.breakdown["overconfidence_rate"] == 0.0

    def test_fabricating_an_answer_to_an_unanswerable_question_is_overconfident(self):
        r = self.metric.compute([self._sample("0", "It is sunny.", "unanswerable", False)])
        assert r.value == 0.0
        assert r.breakdown["overconfidence_rate"] == 1.0

    def test_mixed_batch_breakdown(self):
        samples = [
            self._sample("0", "red", "red", True),
            self._sample("1", "It is sunny.", "unanswerable", False),
            self._sample("2", "Cannot be determined from the image.", "unanswerable", False),
        ]
        r = self.metric.compute(samples)
        assert r.breakdown["answerable_accuracy"] == 1.0
        assert r.breakdown["unanswerable_deflection_rate"] == 0.5
        assert r.breakdown["overconfidence_rate"] == 0.5

    def test_no_scorable_samples_is_nan(self):
        import math

        r = self.metric.compute([ScoredSample(sample_id="0", prediction="red", references=[])])
        assert math.isnan(r.value)
