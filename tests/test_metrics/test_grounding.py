"""Tests for grounding (bounding-box) metrics."""

from vlm_evaluation_harness.metrics.base import ScoredSample
from vlm_evaluation_harness.metrics.grounding import AccAt50Metric, IoUMetric, iou, parse_box


class TestIoUMath:
    def test_identical_boxes(self):
        box = (10.0, 10.0, 50.0, 50.0)
        assert iou(box, box) == 1.0

    def test_disjoint_boxes(self):
        a = (0.0, 0.0, 10.0, 10.0)
        b = (20.0, 20.0, 30.0, 30.0)
        assert iou(a, b) == 0.0

    def test_known_partial_overlap(self):
        # a: [0,0,10,10] area=100. b: [5,5,15,15] area=100.
        # intersection: [5,5,10,10] area=25. union = 100+100-25=175.
        a = (0.0, 0.0, 10.0, 10.0)
        b = (5.0, 5.0, 15.0, 15.0)
        assert abs(iou(a, b) - (25.0 / 175.0)) < 1e-9

    def test_one_box_contains_the_other(self):
        # a: [0,0,10,10] area=100. b (inside a): [2,2,8,8] area=36.
        # intersection = 36 (all of b), union = 100+36-36=100.
        a = (0.0, 0.0, 10.0, 10.0)
        b = (2.0, 2.0, 8.0, 8.0)
        assert abs(iou(a, b) - 0.36) < 1e-9

    def test_touching_edges_is_zero_area_intersection(self):
        a = (0.0, 0.0, 10.0, 10.0)
        b = (10.0, 0.0, 20.0, 10.0)
        assert iou(a, b) == 0.0

    def test_zero_area_box_does_not_raise(self):
        a = (5.0, 5.0, 5.0, 5.0)  # degenerate, zero area
        b = (0.0, 0.0, 10.0, 10.0)
        assert iou(a, b) == 0.0

    def test_two_zero_area_boxes_does_not_raise_division_by_zero(self):
        a = (5.0, 5.0, 5.0, 5.0)
        b = (5.0, 5.0, 5.0, 5.0)
        assert iou(a, b) == 0.0

    def test_symmetric(self):
        a = (0.0, 0.0, 10.0, 10.0)
        b = (5.0, 5.0, 15.0, 15.0)
        assert iou(a, b) == iou(b, a)


class TestParseBox:
    def test_valid(self):
        assert parse_box("10.0,20.0,90.0,80.0") == (10.0, 20.0, 90.0, 80.0)

    def test_wrong_arity(self):
        assert parse_box("10.0,20.0,90.0") is None

    def test_non_numeric(self):
        assert parse_box("I cannot locate that object.") is None

    def test_empty(self):
        assert parse_box("") is None


def _sample(sample_id, prediction, reference):
    return ScoredSample(sample_id=sample_id, prediction=prediction, references=[reference])


class TestIoUMetric:
    def test_perfect_predictions(self):
        samples = [
            _sample("1", "0.0,0.0,10.0,10.0", "0.0,0.0,10.0,10.0"),
            _sample("2", "5.0,5.0,15.0,15.0", "5.0,5.0,15.0,15.0"),
        ]
        result = IoUMetric().compute(samples)
        assert result.value == 1.0
        assert result.n_scored == 2

    def test_unparseable_prediction_scores_zero_not_error(self):
        samples = [_sample("1", "I don't know where that is.", "0.0,0.0,10.0,10.0")]
        result = IoUMetric().compute(samples)
        assert result.value == 0.0

    def test_no_scorable_samples_is_nan(self):
        samples = [ScoredSample(sample_id="1", prediction="x", references=[])]
        result = IoUMetric().compute(samples)
        assert result.value != result.value  # NaN


class TestAccAt50Metric:
    def test_above_threshold_counts(self):
        # IoU here is 0.36 (see test_one_box_contains_the_other) -> below 0.5
        samples = [_sample("1", "2.0,2.0,8.0,8.0", "0.0,0.0,10.0,10.0")]
        result = AccAt50Metric().compute(samples)
        assert result.value == 0.0

    def test_at_or_above_half_counts_as_correct(self):
        samples = [_sample("1", "0.0,0.0,10.0,10.0", "0.0,0.0,10.0,10.0")]  # IoU 1.0
        result = AccAt50Metric().compute(samples)
        assert result.value == 1.0

    def test_mixed(self):
        samples = [
            _sample("1", "0.0,0.0,10.0,10.0", "0.0,0.0,10.0,10.0"),  # IoU 1.0 -> pass
            _sample("2", "2.0,2.0,8.0,8.0", "0.0,0.0,10.0,10.0"),  # IoU 0.36 -> fail
        ]
        result = AccAt50Metric().compute(samples)
        assert result.value == 0.5
