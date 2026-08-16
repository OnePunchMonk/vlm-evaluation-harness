"""Metrics for generative (text-to-image) benchmarks."""

from __future__ import annotations

from vlm_harness.metrics.base import MetricResult


def compute_generative_metrics(
    prompts: list[str],
    images: list,
    metadata: list[dict],
    metric_configs: list,
) -> list[MetricResult]:
    """Dispatch and compute all configured generative metrics."""
    from vlm_harness.benchmarks.schema import MetricConfig
    from vlm_harness.metrics.generative.clip_score import CLIPScorer
    from vlm_harness.metrics.generative.fid import FIDMetric
    from vlm_harness.metrics.generative.geneval import GenEvalClipMetric
    from vlm_harness.metrics.generative.judge import LLMJudgeMetric

    results = []
    for cfg in metric_configs:
        if not isinstance(cfg, MetricConfig):
            continue
        if cfg.type == "clip_score":
            m = CLIPScorer(cfg.clip_model_id)
            results.append(m.compute(prompts, images, metadata))
        elif cfg.type == "geneval_clip":
            checks_field = cfg.checks_field or "checks"
            checks_list = [meta.get(checks_field) for meta in metadata]
            m = GenEvalClipMetric(cfg.clip_model_id)
            results.append(m.compute(images, checks_list, metadata))
        elif cfg.type == "fid":
            if not cfg.reference_dir:
                raise ValueError("The 'fid' metric requires 'reference_dir' in the manifest.")
            m = FIDMetric(cfg.reference_dir)
            results.append(m.compute(images, metadata))
        elif cfg.type == "llm_judge":
            if not cfg.judge_model:
                raise ValueError("The 'llm_judge' metric requires 'judge_model' in the manifest.")
            rubric = cfg.rubric or "Rate the image quality."
            m = LLMJudgeMetric(cfg.judge_model, rubric, cfg.max_score)
            results.append(m.compute(prompts, images, metadata))
    return results


__all__ = ["compute_generative_metrics"]
