"""Metrics for generative (text-to-image) benchmarks."""

from __future__ import annotations

from vlm_evaluation_harness.metrics.base import MetricResult


def compute_generative_metrics(
    prompts: list[str],
    images: list,
    metadata: list[dict],
    metric_configs: list,
    sample_ids: list[str] | None = None,
) -> list[MetricResult]:
    """Dispatch and compute all configured generative metrics.

    Unknown metric types raise rather than being skipped.
    """
    from vlm_evaluation_harness.benchmarks.schema import MetricConfig
    from vlm_evaluation_harness.metrics.generative.clip_score import CLIPScorer
    from vlm_evaluation_harness.metrics.generative.fid import FIDMetric
    from vlm_evaluation_harness.metrics.generative.geneval import GenEvalClipMetric
    from vlm_evaluation_harness.metrics.generative.judge import LLMJudgeMetric, VQAScoreMetric

    results = []
    for cfg in metric_configs:
        if not isinstance(cfg, MetricConfig):
            continue
        if cfg.type == "clip_score":
            results.append(
                CLIPScorer(cfg.clip_model_id).compute(prompts, images, metadata, sample_ids)
            )
        elif cfg.type == "geneval_clip":
            checks_field = cfg.checks_field or "checks"
            checks_list = [meta.get(checks_field) for meta in metadata]
            results.append(
                GenEvalClipMetric(cfg.clip_model_id).compute(
                    images, checks_list, metadata, sample_ids
                )
            )
        elif cfg.type == "fid":
            if not cfg.reference_dir:
                raise ValueError("The 'fid' metric requires 'reference_dir' in the manifest.")
            results.append(FIDMetric(cfg.reference_dir).compute(images, metadata))
        elif cfg.type == "llm_judge":
            if not cfg.judge_model:
                raise ValueError("The 'llm_judge' metric requires 'judge_model' in the manifest.")
            rubric = cfg.rubric or "Rate the image quality."
            results.append(
                LLMJudgeMetric(cfg.judge_model, rubric, cfg.max_score).compute(
                    prompts, images, metadata, sample_ids
                )
            )
        elif cfg.type == "vqa_score":
            if not cfg.judge_model:
                raise ValueError("The 'vqa_score' metric requires 'judge_model' in the manifest.")
            results.append(
                VQAScoreMetric(cfg.judge_model).compute(prompts, images, metadata, sample_ids)
            )
        else:
            raise ValueError(f"Unknown generative metric type: {cfg.type!r}")
    return results


__all__ = ["compute_generative_metrics"]
