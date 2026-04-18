"""Accuracy metrics."""

from __future__ import annotations

from vlm_harness.metrics.base import MetricResult


class AccuracyMetric:
    """Exact-match accuracy."""

    def compute(
        self,
        predictions: list[str],
        references: list[str],
        metadata: list[dict],
    ) -> MetricResult:
        assert len(predictions) == len(references)
        correct = sum(p == r for p, r in zip(predictions, references))
        return MetricResult(
            metric_name="accuracy",
            value=correct / len(predictions) if predictions else 0.0,
            n_samples=len(predictions),
        )

    def compute_by_group(
        self,
        predictions: list[str],
        references: list[str],
        metadata: list[dict],
        group_field: str,
    ) -> MetricResult:
        groups: dict[str, list[tuple[str, str]]] = {}
        for pred, ref, meta in zip(predictions, references, metadata):
            group = str(meta.get(group_field, "unknown"))
            groups.setdefault(group, []).append((pred, ref))

        breakdown = {}
        for group, pairs in sorted(groups.items()):
            preds, refs = zip(*pairs)
            correct = sum(p == r for p, r in zip(preds, refs))
            breakdown[group] = correct / len(pairs)

        overall_correct = sum(p == r for p, r in zip(predictions, references))
        return MetricResult(
            metric_name=f"accuracy_by_{group_field}",
            value=overall_correct / len(predictions) if predictions else 0.0,
            breakdown=breakdown,
            n_samples=len(predictions),
        )
