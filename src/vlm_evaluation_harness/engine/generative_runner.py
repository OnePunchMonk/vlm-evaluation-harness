"""Evaluation runner for generative (text-to-image) benchmarks.

Mirrors engine/runner.py's shape (config -> load samples -> per-sample call
-> aggregate metrics -> save/report) but the per-sample call takes a text
prompt and returns an image instead of taking images+text and returning
text, and scoring happens over images rather than extracted answer strings.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from vlm_evaluation_harness.adapters.generative.base import T2IAdapter
from vlm_evaluation_harness.benchmarks.loader import BenchmarkLoader, BenchmarkSample
from vlm_evaluation_harness.benchmarks.registry import get_registry
from vlm_evaluation_harness.benchmarks.schema import BenchmarkManifest
from vlm_evaluation_harness.engine.runner import (
    RESULTS_SCHEMA_VERSION,
    _harness_sha,
    _harness_version,
)
from vlm_evaluation_harness.metrics.base import MetricResult
from vlm_evaluation_harness.metrics.cost import GenCostTracker
from vlm_evaluation_harness.metrics.generative import compute_generative_metrics
from vlm_evaluation_harness.seeding import seed_everything
from vlm_evaluation_harness.stats import bootstrap_ci

_MAX_SAVED_IMAGES = 12
HARNESS_VERSION = _harness_version()


class GenEvalError(RuntimeError):
    """Raised when a generative run cannot produce trustworthy numbers."""


@dataclass
class GenEvalConfig:
    """Configuration for a single generative evaluation run."""

    model_spec: str
    benchmark: str
    split: str = "prompts"
    max_samples: int | None = None
    seed: int | None = 42
    width: int = 512
    height: int = 512
    guidance_scale: float = 7.0
    num_inference_steps: int = 30
    output_dir: Path | None = None
    # Number of images to generate per prompt, each with a different seed.
    # One image per prompt measures a single draw from the model's output
    # distribution; FID and CLIPScore over a single draw are dominated by
    # seed variance.
    images_per_prompt: int = 1
    # Whether the saved results JSON includes the full per-sample "samples"
    # array. On by default, matching the harness's historical behavior.
    log_samples: bool = True
    # Generate images and cache them, but skip metric scoring (CLIPScore /
    # judge / FID etc.) entirely.
    predict_only: bool = False


@dataclass
class GenSampleResult:
    sample_id: str
    prompt: str
    latency_ms: float
    cost_usd: float
    seed: int | None
    metadata: dict[str, Any]


@dataclass
class GenEvalResult:
    """Aggregated result of a complete generative evaluation run."""

    config: GenEvalConfig
    manifest: BenchmarkManifest
    sample_results: list[GenSampleResult]
    images: list  # PIL.Image, parallel to sample_results
    metrics: list[MetricResult]
    cost_summary: Any
    started_at: str
    finished_at: str
    provenance: dict = field(default_factory=dict)
    harness_version: str = HARNESS_VERSION

    def metric_confidence_intervals(self) -> dict[str, tuple[float, float]]:
        return {
            m.metric_name: bootstrap_ci(list(m.per_sample.values()))
            for m in self.metrics
            if m.per_sample
        }

    def per_sample_scores(self) -> dict[str, dict[str, float]]:
        return {m.metric_name: m.per_sample for m in self.metrics if m.per_sample}

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
            "metric_n_scored": {m.metric_name: m.n_scored for m in self.metrics},
            "metric_ci95": {
                k: list(v) for k, v in self.metric_confidence_intervals().items()
            },
            "metric_breakdowns": {m.metric_name: m.breakdown for m in self.metrics if m.breakdown},
            "provenance": self.provenance,
            "cost": {
                "total_usd": self.cost_summary.total_cost_usd,
                "per_sample_usd": self.cost_summary.cost_per_sample_usd,
            },
            "latency": {
                "p50_ms": self.cost_summary.latency_p50_ms,
                "p95_ms": self.cost_summary.latency_p95_ms,
                "p99_ms": self.cost_summary.latency_p99_ms,
                "throughput_per_min": self.cost_summary.throughput_samples_per_min,
            },
        }

    def save(self, output_dir: Path, log_samples: bool = True) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        model_slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", self.config.model_spec).strip("-")
        slug = f"{self.manifest.name.lower()}_{model_slug}_{ts}"

        full = self.to_dict()
        if not log_samples:
            result_file = output_dir / f"{slug}_results.json"
            result_file.write_text(json.dumps(full, indent=2))
            return result_file

        image_dir = output_dir / f"{slug}_images"
        image_paths: list[str | None] = [None] * len(self.sample_results)
        for i, (sample, image) in enumerate(zip(self.sample_results, self.images)):
            if i >= _MAX_SAVED_IMAGES:
                break
            image_dir.mkdir(parents=True, exist_ok=True)
            path = image_dir / f"{sample.sample_id}.png"
            image.save(path)
            image_paths[i] = str(path)

        full["samples"] = [
            {
                "id": s.sample_id,
                "prompt": s.prompt,
                "latency_ms": s.latency_ms,
                "cost_usd": s.cost_usd,
                "seed": s.seed,
                "metadata": s.metadata,
                "image_path": image_paths[i],
            }
            for i, s in enumerate(self.sample_results)
        ]

        result_file = output_dir / f"{slug}_results.json"
        result_file.write_text(json.dumps(full, indent=2))
        return result_file


class GenerativeEvalRunner:
    """Orchestrates a complete text-to-image evaluation run."""

    def __init__(self, adapter: T2IAdapter):
        self._adapter = adapter
        self._loader = BenchmarkLoader()

    def run(self, config: GenEvalConfig) -> GenEvalResult:
        seed_everything(config.seed)
        registry = get_registry()
        manifest = registry.get(config.benchmark)

        cost_tracker = GenCostTracker()
        samples: list[BenchmarkSample] = list(
            self._loader.load(manifest, config.split, config.max_samples)
        )
        if not samples:
            raise GenEvalError(
                f"Benchmark '{manifest.name}' split '{config.split}' yielded zero samples."
            )

        started_at = datetime.now(timezone.utc).isoformat()
        sample_results: list[GenSampleResult] = []
        images = []

        n_variants = max(1, config.images_per_prompt)
        for sample in tqdm(samples, desc=f"Generating {manifest.name}"):
            prompt = sample.text_fields.get("question", "")
            for variant in range(n_variants):
                seed = None if config.seed is None else config.seed + variant
                response = self._adapter.generate(
                    prompt=prompt,
                    seed=seed,
                    width=config.width,
                    height=config.height,
                    guidance_scale=config.guidance_scale,
                    num_inference_steps=config.num_inference_steps,
                )
                cost_tracker.record(response)
                images.append(response.image)
                sample_id = (
                    sample.sample_id if n_variants == 1 else f"{sample.sample_id}#{variant}"
                )
                sample_results.append(
                    GenSampleResult(
                        sample_id=sample_id,
                        prompt=prompt,
                        latency_ms=response.latency_ms,
                        cost_usd=response.cost_usd,
                        seed=seed,
                        metadata=sample.metadata,
                    )
                )

        finished_at = datetime.now(timezone.utc).isoformat()

        if config.predict_only:
            metrics = []
        else:
            prompts = [s.prompt for s in sample_results]
            metadata = [s.metadata for s in sample_results]
            sample_ids = [s.sample_id for s in sample_results]
            metrics = compute_generative_metrics(
                prompts, images, metadata, manifest.metrics, sample_ids
            )
            if all(m.n_scored == 0 for m in metrics):
                raise GenEvalError(
                    f"No metric produced a score for '{manifest.name}' — check the manifest's "
                    "metrics configuration."
                )
        cost_summary = cost_tracker.summary()

        result = GenEvalResult(
            config=config,
            manifest=manifest,
            sample_results=sample_results,
            images=images,
            metrics=metrics,
            cost_summary=cost_summary,
            started_at=started_at,
            finished_at=finished_at,
            provenance=self._provenance(manifest, config, registry),
        )

        if config.output_dir:
            saved_path = result.save(config.output_dir, log_samples=config.log_samples)
            print(f"Results saved to {saved_path}")

        return result

    def _provenance(self, manifest, config: GenEvalConfig, registry) -> dict[str, Any]:
        return {
            "results_schema_version": RESULTS_SCHEMA_VERSION,
            "harness_version": HARNESS_VERSION,
            "harness_sha": _harness_sha(),
            "model_spec": config.model_spec,
            "adapter_model_id": self._adapter.model_id,
            "benchmark_version": manifest.version,
            "manifest_hash": registry.manifest_hash(manifest.name),
            "generation": {
                "seed": config.seed,
                "width": config.width,
                "height": config.height,
                "guidance_scale": config.guidance_scale,
                "num_inference_steps": config.num_inference_steps,
                "images_per_prompt": config.images_per_prompt,
            },
        }
