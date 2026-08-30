"""Tests for hallucination metrics."""

from vlm_harness.metrics.hallucination import CHAIRMetric, POPEMetric


class TestCHAIR:
    metric = CHAIRMetric()

    def test_no_hallucination(self):
        captions = ["I see a cat and a dog."]
        gt = [["cat", "dog"]]
        r = self.metric.compute(captions, gt, [{}])
        assert r.breakdown["chair_s"] == 0.0
        assert r.breakdown["chair_i"] == 0.0

    def test_full_hallucination(self):
        captions = ["There is a bicycle and a car."]
        gt = [[]]  # nothing in the image
        r = self.metric.compute(captions, gt, [{}])
        assert r.breakdown["chair_s"] > 0.0
        assert r.breakdown["chair_i"] == 1.0

    def test_partial_hallucination(self):
        captions = ["I see a cat and a bicycle."]
        gt = [["cat"]]  # bicycle not in GT
        r = self.metric.compute(captions, gt, [{}])
        assert 0 < r.breakdown["chair_s"] < 1.0

    def test_empty_caption(self):
        captions = [""]
        gt = [["cat"]]
        r = self.metric.compute(captions, gt, [{}])
        assert r.breakdown["chair_s"] == 0.0


class TestPOPE:
    metric = POPEMetric()

    def test_all_correct(self):
        preds = ["Yes", "No", "Yes"]
        refs = ["yes", "no", "yes"]
        r = self.metric.compute(preds, refs, [{}, {}, {}])
        assert r.value == 1.0
        assert r.breakdown["f1"] == 1.0

    def test_all_wrong(self):
        preds = ["No", "Yes"]
        refs = ["yes", "no"]
        r = self.metric.compute(preds, refs, [{}, {}])
        assert r.value == 0.0

    def test_yes_rate(self):
        preds = ["Yes", "Yes", "No"]
        refs = ["yes", "no", "no"]
        r = self.metric.compute(preds, refs, [{}, {}, {}])
        assert abs(r.breakdown["yes_rate"] - 2 / 3) < 1e-6
