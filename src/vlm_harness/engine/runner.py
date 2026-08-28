"""Main evaluation runner."""

from __future__ import annotations

import json
import math
import platform
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from vlm_harness.adapters.base import VLMAdapter, VLMResponse, supports_choice_scoring
from vlm_harness.benchmarks.loader import BenchmarkLoader, BenchmarkSample
from vlm_harness.benchmarks.registry import get_registry
from vlm_harness.benchmarks.schema import BenchmarkManifest
from vlm_harness.cache import ResponseCache, response_key
from vlm_harness.images.corruption import apply_corruption
from vlm_harness.images.pipeline import ImagePipeline, ImagePipelineConfig
from vlm_harness.metrics.base import MetricResult, ScoredSample, compute_metrics
from vlm_harness.metrics.cost import CostTracker
from vlm_harness.parsing.extractor import AnswerExtractor
from vlm_harness.prompt.formatter import PromptFormatter
from vlm_harness.stats import bootstrap_ci

HARNESS_VERSION = "0.2.0"


class EvalError(RuntimeError):
    """Raised when a run cannot produce trustworthy numbers."""


@dataclass
class EvalConfig:
    """Configuration for a single evaluation run."""

    model_spec: str
    benchmark: str
    split: str = "validation"
    max_samples: int | None = None
    max_concurrent: int = 1
    temperature: float = 0.0
    max_tokens: int = 1024
    output_dir: Path | None = None
    # Corruptions applied, in order, to every image before it reaches the
    # model. Recorded in provenance so a robustness run is never confused
    # with a clean one.
    robustness_corruptions: list[str] = field(default_factory=list)
    corruption_severity: int = 2
    seed: int = 42
    use_cache: bool = True
    cache_path: Path | None = None
    # Fail the run if more than this fraction of samples error out.
    max_error_rate: float = 0.1
    # Self-consistency (Wang et al.): sample the model N times at
    # `temperature` and majority-vote the extracted answers, instead of
    # trusting a single greedy/sampled generation. N=1 (default) is today's
    # single-call behavior, unchanged. Only applies to `scoring: generate`
    # benchmarks — loglikelihood scoring is already exact, not sampled.
    self_consistency_n: int = 1


@dataclass
class SampleResult:
    sample_id: str
    prediction: str
    references: list[str]
    raw_output: str
    confident: bool
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cached: bool
    image_hashes: list[str]
    metadata: dict[str, Any]
    error: str | None = None

    @property
    def scorable(self) -> bool:
        return self.error is None and bool(self.references)


@dataclass
class EvalResult:
    """Aggregated result of a complete evaluation run."""

    config: EvalConfig
    manifest: BenchmarkManifest
    sample_results: list[SampleResult]
    metrics: list[MetricResult]
    cost_summary: Any  # CostSummary
    started_at: str
    finished_at: str
    provenance: dict[str, Any] = field(default_factory=dict)
    harness_version: str = HARNESS_VERSION

    @property
    def primary_metric(self) -> MetricResult | None:
        """First per-sample metric, used for paired regression testing."""
        for m in self.metrics:
            if m.per_sample and m.metric_name != "extraction_failure_rate":
                return m
        return None

    def metric_confidence_intervals(self) -> dict[str, tuple[float, float]]:
        return {
            m.metric_name: bootstrap_ci(list(m.per_sample.values()))
            for m in self.metrics
            if m.per_sample
        }

    def to_dict(self) -> dict:
        cis = self.metric_confidence_intervals()
        return {
            "harness_version": self.harness_version,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "model": self.config.model_spec,
            "benchmark": self.manifest.name,
            "split": self.config.split,
            "n_samples": len(self.sample_results),
            "n_scored": max((m.n_scored for m in self.metrics), default=0),
            "n_errors": sum(1 for s in self.sample_results if s.error),
            "metrics": {m.metric_name: m.value for m in self.metrics},
            "metric_n_scored": {m.metric_name: m.n_scored for m in self.metrics},
            "metric_ci95": {k: list(v) for k, v in cis.items()},
            "metric_breakdowns": {
                m.metric_name: m.breakdown for m in self.metrics if m.breakdown
            },
            "provenance": self.provenance,
            "cost": {
                "total_usd": self.cost_summary.total_cost_usd,
                "per_sample_usd": self.cost_summary.cost_per_sample_usd,
                "total_input_tokens": self.cost_summary.total_input_tokens,
                "total_output_tokens": self.cost_summary.total_output_tokens,
            },
            "latency": {
                "p50_ms": self.cost_summary.latency_p50_ms,
                "p95_ms": self.cost_summary.latency_p95_ms,
                "p99_ms": self.cost_summary.latency_p99_ms,
                "throughput_per_min": self.cost_summary.throughput_samples_per_min,
            },
        }

    def per_sample_scores(self) -> dict[str, dict[str, float]]:
        """Per-sample score for every metric that produces one."""
        return {m.metric_name: m.per_sample for m in self.metrics if m.per_sample}

    def save(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        slug = f"{self.manifest.name.lower()}_{ts}"
        result_file = output_dir / f"{slug}_results.json"

        per_sample = self.per_sample_scores()
        full = self.to_dict()
        full["samples"] = [
            {
                "id": s.sample_id,
                "prediction": s.prediction,
                "references": s.references,
                "raw_output": s.raw_output,
                "scores": {
                    name: scores[s.sample_id]
                    for name, scores in per_sample.items()
                    if s.sample_id in scores
                },
                "confident": s.confident,
                "cached": s.cached,
                "latency_ms": s.latency_ms,
                "image_hashes": s.image_hashes,
                "metadata": s.metadata,
                "error": s.error,
            }
            for s in self.sample_results
        ]
        result_file.write_text(json.dumps(full, indent=2, default=str))
        return result_file


class EvalRunner:
    """Orchestrates a complete evaluation run."""

    def __init__(self, adapter: VLMAdapter, image_config: ImagePipelineConfig | None = None):
        self._adapter = adapter
        self._formatter = PromptFormatter()
        self._extractor = AnswerExtractor()
        self._loader = BenchmarkLoader()
        self._image_config = image_config or ImagePipelineConfig()
        self._image_pipeline = ImagePipeline(self._image_config)

    def run(self, config: EvalConfig) -> EvalResult:
        registry = get_registry()
        manifest = registry.get(config.benchmark)

        split_config = next((s for s in manifest.splits if s.name == config.split), None)
        if split_config is None:
            raise EvalError(
                f"Split '{config.split}' is not declared by benchmark '{manifest.name}'. "
                f"Available: {[s.name for s in manifest.splits]}"
            )
        if manifest.scoring == "loglikelihood" and not supports_choice_scoring(self._adapter):
            raise EvalError(
                f"Benchmark '{manifest.name}' requires scoring='loglikelihood', but adapter "
                f"'{config.model_spec}' does not implement score_choices()."
            )

        cache = ResponseCache(config.cache_path, enabled=config.use_cache)
        cost_tracker = CostTracker(
            cost_per_million_input=self._adapter.cost_per_million_input_tokens,
            cost_per_million_output=self._adapter.cost_per_million_output_tokens,
        )

        samples: list[BenchmarkSample] = list(
            self._loader.load(manifest, config.split, config.max_samples)
        )
        if not samples:
            raise EvalError(
                f"Benchmark '{manifest.name}' split '{config.split}' yielded zero samples."
            )

        few_shot = self._build_few_shot(manifest)
        started_at = datetime.now(timezone.utc).isoformat()

        try:
            sample_results = self._run_all(samples, manifest, config, cache, few_shot)
        finally:
            cache.close()

        finished_at = datetime.now(timezone.utc).isoformat()

        errors = [s for s in sample_results if s.error]
        error_rate = len(errors) / len(sample_results)
        if error_rate > config.max_error_rate:
            raise EvalError(
                f"{len(errors)}/{len(sample_results)} samples failed "
                f"({error_rate:.1%} > max_error_rate {config.max_error_rate:.1%}). "
                f"First error: {errors[0].error}"
            )

        for r in sample_results:
            if not r.error and not r.cached:
                cost_tracker.record(
                    VLMResponse(
                        text=r.raw_output,
                        input_tokens=r.input_tokens,
                        output_tokens=r.output_tokens,
                        latency_ms=r.latency_ms,
                    )
                )

        scored = [
            ScoredSample(
                sample_id=r.sample_id,
                prediction=r.prediction,
                references=r.references,
                metadata=r.metadata,
                confident=r.confident,
            )
            for r in sample_results
            if r.error is None
        ]

        if split_config.scorable:
            metrics = compute_metrics(scored, manifest.metrics)
            if all(math.isnan(m.value) for m in metrics):
                raise EvalError(
                    f"No sample in '{manifest.name}' split '{config.split}' carried ground "
                    "truth, so no metric could be computed. Check fields.answer in the manifest."
                )
        else:
            metrics = []

        result = EvalResult(
            config=config,
            manifest=manifest,
            sample_results=sample_results,
            metrics=metrics,
            cost_summary=cost_tracker.summary(),
            started_at=started_at,
            finished_at=finished_at,
            provenance=self._provenance(manifest, config, registry, cache, few_shot),
        )

        if config.output_dir:
            saved_path = result.save(config.output_dir)
            print(f"Results saved to {saved_path}")

        return result

    # -- internals ---------------------------------------------------------

    def _provenance(
        self,
        manifest: BenchmarkManifest,
        config: EvalConfig,
        registry,
        cache: ResponseCache,
        few_shot: list[dict],
    ) -> dict[str, Any]:
        """Everything needed to reproduce or invalidate this run.

        Results that do not record their decoding parameters, prompt version,
        and image preprocessing cannot be compared to each other, which
        defeats the purpose of tracking them over time.
        """
        return {
            "harness_version": HARNESS_VERSION,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "model_spec": config.model_spec,
            "adapter_model_id": self._adapter.model_id,
            "benchmark_version": manifest.version,
            "manifest_hash": registry.manifest_hash(manifest.name),
            "scoring": manifest.scoring,
            "task_type": manifest.task_type,
            "decoding": {
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "seed": config.seed,
            },
            "prompt": {
                "template": manifest.prompt_template,
                "template_b": manifest.prompt_template_b,
                "system": manifest.system_prompt,
                "few_shot_count": len(few_shot),
                "answer_extraction": asdict(manifest.answer_extraction),
            },
            "images": {
                "max_resolution": list(self._image_config.max_resolution),
                "min_resolution": list(self._image_config.min_resolution),
                "color_space": self._image_config.color_space,
                "corruptions": list(config.robustness_corruptions),
                "corruption_severity": config.corruption_severity,
            },
            "cache": {
                "enabled": config.use_cache,
                "hits": cache.stats.hits,
                "misses": cache.stats.misses,
            },
        }

    def _build_few_shot(self, manifest: BenchmarkManifest) -> list[dict]:
        if not manifest.few_shot.count:
            return []
        examples = self._loader.load_few_shot(manifest)
        rendered = []
        for ex in examples:
            fields = dict(ex.text_fields)
            fields["answer"] = ex.references[0] if ex.references else ""
            rendered.append(fields)
        return rendered

    def _run_all(
        self,
        samples: list[BenchmarkSample],
        manifest: BenchmarkManifest,
        config: EvalConfig,
        cache: ResponseCache,
        few_shot: list[dict],
    ) -> list[SampleResult]:
        desc = f"Evaluating {manifest.name}"
        workers = max(1, config.max_concurrent)

        def work(sample: BenchmarkSample) -> list[SampleResult]:
            return self._eval_sample(sample, manifest, config, cache, few_shot)

        results: list[SampleResult] = []
        if workers == 1:
            for sample in tqdm(samples, desc=desc):
                results.extend(work(sample))
        else:
            # Ordered map: results stay in dataset order regardless of
            # completion order, so runs remain byte-comparable.
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for batch in tqdm(
                    pool.map(work, samples), total=len(samples), desc=desc
                ):
                    results.extend(batch)
        return results

    def _eval_sample(
        self,
        sample: BenchmarkSample,
        manifest: BenchmarkManifest,
        config: EvalConfig,
        cache: ResponseCache,
        few_shot: list[dict],
    ) -> list[SampleResult]:
        try:
            processed = self._image_pipeline.process_batch(sample.images)
            images = [p.image for p in processed]
            image_hashes = [p.hash for p in processed]

            if config.robustness_corruptions:
                images = [
                    self._corrupt(img, config) for img in images
                ]
                # Corrupted pixels are a different cache key.
                image_hashes = [
                    f"{h}+{'+'.join(config.robustness_corruptions)}@{config.corruption_severity}"
                    for h in image_hashes
                ]

            if manifest.task_type == "pairwise_matching":
                return self._eval_pairwise(
                    sample, manifest, config, cache, few_shot, images, image_hashes
                )

            return [
                self._eval_single(
                    sample,
                    manifest,
                    config,
                    cache,
                    few_shot,
                    images,
                    image_hashes,
                    template=None,
                    sample_id=sample.sample_id,
                    references=sample.references,
                )
            ]
        except Exception as exc:  # noqa: BLE001 - recorded per sample
            return [
                SampleResult(
                    sample_id=sample.sample_id,
                    prediction="",
                    references=sample.references,
                    raw_output="",
                    confident=False,
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=0.0,
                    cached=False,
                    image_hashes=[],
                    metadata=sample.metadata,
                    error=f"{type(exc).__name__}: {exc}",
                )
            ]

    def _corrupt(self, image, config: EvalConfig):
        for name in config.robustness_corruptions:
            image = apply_corruption(image, name, severity=config.corruption_severity)
        return image

    def _eval_pairwise(
        self,
        sample: BenchmarkSample,
        manifest: BenchmarkManifest,
        config: EvalConfig,
        cache: ResponseCache,
        few_shot: list[dict],
        images: list,
        image_hashes: list[str],
    ) -> list[SampleResult]:
        """Winoground-style: each sample is scored with two prompts.

        Both must be answered correctly for the pair to count, which is what
        `pairwise_group` aggregates. A single accuracy number over one prompt
        — as the old manifest asked for — is not a Winoground score.
        """
        results = []
        for slot, (template, answer) in enumerate(
            zip(
                [manifest.prompt_template, manifest.prompt_template_b],
                manifest.pairwise_answers,
            )
        ):
            suffix = "ab"[slot]
            metadata = dict(sample.metadata)
            metadata["pair_id"] = sample.sample_id
            metadata["pair_slot"] = suffix
            results.append(
                self._eval_single(
                    sample,
                    manifest,
                    config,
                    cache,
                    few_shot,
                    images,
                    image_hashes,
                    template=template,
                    sample_id=f"{sample.sample_id}#{suffix}",
                    references=[answer],
                    metadata=metadata,
                )
            )
        return results

    def _eval_single(
        self,
        sample: BenchmarkSample,
        manifest: BenchmarkManifest,
        config: EvalConfig,
        cache: ResponseCache,
        few_shot: list[dict],
        images: list,
        image_hashes: list[str],
        template: str | None,
        sample_id: str,
        references: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> SampleResult:
        formatted = self._formatter.format(
            manifest, images, sample.text_fields, few_shot_examples=few_shot, template=template
        )

        if manifest.scoring == "loglikelihood":
            return self._eval_loglikelihood(
                formatted, sample_id, references, metadata or sample.metadata, image_hashes
            )

        n_samples = max(1, config.self_consistency_n)
        responses: list[VLMResponse] = []
        extractions: list = []
        for i in range(n_samples):
            params = {
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "mode": "generate",
            }
            if n_samples > 1:
                # Distinct cache key per vote: with temperature > 0 these are
                # meant to be independent samples, not the same cached call
                # replayed N times.
                params["self_consistency_index"] = i
            key = response_key(
                self._adapter.model_id, formatted.text, formatted.system, image_hashes, params
            )
            cached_payload = cache.get(key)
            if cached_payload is not None:
                response = VLMResponse.from_cache_payload(cached_payload)
            else:
                response = self._adapter.generate(
                    images=formatted.images,
                    prompt=formatted.text,
                    system=formatted.system,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                )
                cache.put(key, self._adapter.model_id, response.to_cache_payload())
            responses.append(response)
            extractions.append(self._extractor.extract(response.text, manifest.answer_extraction))

        if n_samples == 1:
            winner = extractions[0]
            confident = winner.confident
        else:
            votes: dict[str, int] = {}
            first_seen: dict[str, int] = {}
            for i, ext in enumerate(extractions):
                votes[ext.normalized] = votes.get(ext.normalized, 0) + 1
                first_seen.setdefault(ext.normalized, i)
            best = max(votes, key=lambda v: (votes[v], -first_seen[v]))
            winner = next(e for e in extractions if e.normalized == best)
            # Confident only when the winning answer took a strict majority
            # of the N votes — a 2-2-1 split isn't consensus.
            confident = votes[best] > n_samples / 2

        return SampleResult(
            sample_id=sample_id,
            prediction=winner.normalized,
            references=self._normalize_references(references, manifest),
            raw_output=responses[-1].text,
            confident=confident,
            input_tokens=sum(r.input_tokens for r in responses),
            output_tokens=sum(r.output_tokens for r in responses),
            latency_ms=sum(r.latency_ms for r in responses),
            cached=all(r.cached for r in responses),
            image_hashes=image_hashes,
            metadata=metadata if metadata is not None else sample.metadata,
        )

    def _eval_loglikelihood(
        self,
        formatted,
        sample_id: str,
        references: list[str],
        metadata: dict[str, Any],
        image_hashes: list[str],
    ) -> SampleResult:
        choices = formatted.raw_fields.get("choices") or []
        if not choices:
            raise EvalError(f"sample {sample_id} has no choices to score")

        scores = self._adapter.score_choices(
            images=formatted.images,
            prompt=formatted.text,
            choices=[str(c) for c in choices],
            system=formatted.system,
        )
        best = scores.argmax(length_normalized=True)
        letter = self._formatter.format_choices(choices).splitlines()[best].split(".")[0]

        return SampleResult(
            sample_id=sample_id,
            prediction=letter,
            references=references,
            raw_output=json.dumps(
                {"logprobs": scores.logprobs, "logprobs_per_token": scores.logprobs_per_token}
            ),
            # Log-likelihood scoring has no extraction step, so it can never
            # fail ambiguously.
            confident=True,
            input_tokens=0,
            output_tokens=0,
            latency_ms=scores.latency_ms,
            cached=False,
            image_hashes=image_hashes,
            metadata=metadata,
        )

    def _normalize_references(
        self, references: list[str], manifest: BenchmarkManifest
    ) -> list[str]:
        """Apply the manifest's normalization to references as well as predictions.

        Comparing a normalized prediction against a raw reference is a silent
        source of false negatives (e.g. 'B' vs 'b.').
        """
        from vlm_harness.parsing.normalizer import normalize_answer

        mode = manifest.answer_extraction.normalize
        return [normalize_answer(r, mode=mode) for r in references]
