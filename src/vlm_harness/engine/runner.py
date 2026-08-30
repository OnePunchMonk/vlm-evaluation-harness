"""Main evaluation runner."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from vlm_harness.adapters.base import VLMAdapter
from vlm_harness.benchmarks.loader import BenchmarkLoader, BenchmarkSample
from vlm_harness.benchmarks.registry import get_registry
from vlm_harness.benchmarks.schema import BenchmarkManifest
from vlm_harness.images.pipeline import ImagePipeline, ImagePipelineConfig
from vlm_harness.metrics.base import MetricResult, compute_metrics
from vlm_harness.metrics.cost import CostTracker
from vlm_harness.parsing.extractor import AnswerExtractor
from vlm_harness.prompt.formatter import PromptFormatter


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
    robustness_corruptions: list[str] = field(default_factory=list)
    seed: int = 42
    # 1 = the original per-sample generate() loop. An int > 1 batches that
    # many samples per adapter.generate_batch() call. "auto" batches via
    # adapter.generate_auto_batch() (currently only HuggingFaceAdapter
    # implements it), which backs off the batch size on GPU OOM. Ignored
    # for adapters that don't implement generate_batch.
    batch_size: int | str = 1


@dataclass
class SampleResult:
    sample_id: str
    prediction: str
    reference: str | None
    raw_output: str
    confident: bool
    input_tokens: int
    output_tokens: int
    latency_ms: float
    image_hashes: list[str]
    metadata: dict[str, Any]


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
    harness_version: str = "0.1.0"

    def to_dict(self) -> dict:
        return {
            "harness_version": self.harness_version,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "model": self.config.model_spec,
            "benchmark": self.manifest.name,
            "split": self.config.split,
            "n_samples": len(self.sample_results),
            "metrics": {m.metric_name: m.value for m in self.metrics},
            "metric_breakdowns": {m.metric_name: m.breakdown for m in self.metrics if m.breakdown},
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

    def save(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        slug = f"{self.manifest.name.lower()}_{ts}"
        result_file = output_dir / f"{slug}_results.json"

        full = self.to_dict()
        full["samples"] = [
            {
                "id": s.sample_id,
                "prediction": s.prediction,
                "reference": s.reference,
                "correct": s.prediction == s.reference if s.reference is not None else None,
                "confident": s.confident,
                "latency_ms": s.latency_ms,
                "image_hashes": s.image_hashes,
                "metadata": s.metadata,
            }
            for s in self.sample_results
        ]
        result_file.write_text(json.dumps(full, indent=2))
        return result_file


class EvalRunner:
    """Orchestrates a complete evaluation run."""

    def __init__(self, adapter: VLMAdapter, image_config: ImagePipelineConfig | None = None):
        self._adapter = adapter
        self._formatter = PromptFormatter()
        self._extractor = AnswerExtractor()
        self._loader = BenchmarkLoader()
        self._image_pipeline = ImagePipeline(image_config)

    def run(self, config: EvalConfig) -> EvalResult:
        registry = get_registry()
        manifest = registry.get(config.benchmark)

        cost_tracker = CostTracker(
            cost_per_million_input=self._adapter.cost_per_million_input_tokens,
            cost_per_million_output=self._adapter.cost_per_million_output_tokens,
        )

        samples: list[BenchmarkSample] = list(
            self._loader.load(manifest, config.split, config.max_samples)
        )

        started_at = datetime.now(timezone.utc).isoformat()

        if config.batch_size != 1 and hasattr(self._adapter, "generate_batch"):
            sample_results = self._run_batched(samples, manifest, config, cost_tracker)
        else:
            sample_results = [
                self._eval_sample(sample, manifest, config, cost_tracker)
                for sample in tqdm(samples, desc=f"Evaluating {manifest.name}")
            ]

        finished_at = datetime.now(timezone.utc).isoformat()

        predictions = [s.prediction for s in sample_results]
        references = [s.reference or "" for s in sample_results]
        metadata = [s.metadata for s in sample_results]

        metrics = compute_metrics(predictions, references, metadata, manifest.metrics)
        cost_summary = cost_tracker.summary()

        result = EvalResult(
            config=config,
            manifest=manifest,
            sample_results=sample_results,
            metrics=metrics,
            cost_summary=cost_summary,
            started_at=started_at,
            finished_at=finished_at,
        )

        if config.output_dir:
            saved_path = result.save(config.output_dir)
            print(f"Results saved to {saved_path}")

        return result

    def _prepare_sample(self, sample: BenchmarkSample, manifest: BenchmarkManifest):
        """Image processing + prompt formatting shared by both eval paths."""
        processed = self._image_pipeline.process_batch(sample.images)
        images: list[Image.Image | str] = [p.image for p in processed]
        image_hashes = [p.hash for p in processed]
        formatted = self._formatter.format(manifest, images, sample.text_fields)
        return formatted, image_hashes

    def _finalize_sample(
        self,
        sample: BenchmarkSample,
        manifest: BenchmarkManifest,
        response,
        image_hashes: list[str],
        cost_tracker: CostTracker,
    ) -> SampleResult:
        cost_tracker.record(response)
        extraction = self._extractor.extract(response.text, manifest.answer_extraction)
        return SampleResult(
            sample_id=sample.sample_id,
            prediction=extraction.normalized,
            reference=sample.answer,
            raw_output=response.text,
            confident=extraction.confident,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
            image_hashes=image_hashes,
            metadata=sample.metadata,
        )

    def _eval_sample(
        self,
        sample: BenchmarkSample,
        manifest: BenchmarkManifest,
        config: EvalConfig,
        cost_tracker: CostTracker,
    ) -> SampleResult:
        formatted, image_hashes = self._prepare_sample(sample, manifest)
        response = self._adapter.generate(
            images=formatted.images,
            prompt=formatted.text,
            system=formatted.system,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )
        return self._finalize_sample(sample, manifest, response, image_hashes, cost_tracker)

    def _run_batched(
        self,
        samples: list[BenchmarkSample],
        manifest: BenchmarkManifest,
        config: EvalConfig,
        cost_tracker: CostTracker,
    ) -> list[SampleResult]:
        prepared = [self._prepare_sample(sample, manifest) for sample in samples]
        requests = [
            {"images": formatted.images, "prompt": formatted.text, "system": formatted.system}
            for formatted, _ in prepared
        ]

        if config.batch_size == "auto" and hasattr(self._adapter, "generate_auto_batch"):
            auto_batch = self._adapter.generate_auto_batch  # type: ignore[attr-defined]
            responses = auto_batch(
                requests, max_tokens=config.max_tokens, temperature=config.temperature
            )
        else:
            generate_batch = self._adapter.generate_batch  # type: ignore[attr-defined]
            chunk_size = int(config.batch_size) if config.batch_size != "auto" else len(requests)
            responses = []
            for start in tqdm(
                range(0, len(requests), chunk_size), desc=f"Evaluating {manifest.name}"
            ):
                chunk = requests[start : start + chunk_size]
                responses.extend(
                    generate_batch(
                        chunk, max_tokens=config.max_tokens, temperature=config.temperature
                    )
                )

        return [
            self._finalize_sample(sample, manifest, response, image_hashes, cost_tracker)
            for sample, (_, image_hashes), response in zip(samples, prepared, responses)
        ]
