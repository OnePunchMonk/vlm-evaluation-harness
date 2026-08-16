"""Evaluation runner for generative (text-to-image) benchmarks.

Mirrors engine/runner.py's shape (config -> load samples -> per-sample call
-> aggregate metrics -> save/report) but the per-sample call takes a text
prompt and returns an image instead of taking images+text and returning
text, and scoring happens over images rather than extracted answer strings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from vlm_harness.adapters.generative.base import T2IAdapter
from vlm_harness.benchmarks.loader import BenchmarkLoader, BenchmarkSample
from vlm_harness.benchmarks.registry import get_registry
from vlm_harness.benchmarks.schema import BenchmarkManifest
from vlm_harness.metrics.base import MetricResult
from vlm_harness.metrics.cost import GenCostTracker
from vlm_harness.metrics.generative import compute_generative_metrics

_MAX_SAVED_IMAGES = 12


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


@dataclass
class GenSampleResult:
    sample_id: str
    prompt: str
    latency_ms: float
    cost_usd: float
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

        image_dir = output_dir / f"{slug}_images"
        image_paths: list[str | None] = [None] * len(self.sample_results)
        for i, (sample, image) in enumerate(zip(self.sample_results, self.images)):
            if i >= _MAX_SAVED_IMAGES:
                break
            image_dir.mkdir(parents=True, exist_ok=True)
            path = image_dir / f"{sample.sample_id}.png"
            image.save(path)
            image_paths[i] = str(path)

        full = self.to_dict()
        full["samples"] = [
            {
                "id": s.sample_id,
                "prompt": s.prompt,
                "latency_ms": s.latency_ms,
                "cost_usd": s.cost_usd,
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
        registry = get_registry()
        manifest = registry.get(config.benchmark)

        cost_tracker = GenCostTracker()
        samples: list[BenchmarkSample] = list(
            self._loader.load(manifest, config.split, config.max_samples)
        )

        started_at = datetime.now(timezone.utc).isoformat()
        sample_results: list[GenSampleResult] = []
        images = []

        for sample in tqdm(samples, desc=f"Generating {manifest.name}"):
            prompt = sample.text_fields.get("question", "")
            response = self._adapter.generate(
                prompt=prompt,
                seed=config.seed,
                width=config.width,
                height=config.height,
                guidance_scale=config.guidance_scale,
                num_inference_steps=config.num_inference_steps,
            )
            cost_tracker.record(response)
            images.append(response.image)
            sample_results.append(
                GenSampleResult(
                    sample_id=sample.sample_id,
                    prompt=prompt,
                    latency_ms=response.latency_ms,
                    cost_usd=response.cost_usd,
                    metadata=sample.metadata,
                )
            )

        finished_at = datetime.now(timezone.utc).isoformat()

        prompts = [s.prompt for s in sample_results]
        metadata = [s.metadata for s in sample_results]
        metrics = compute_generative_metrics(prompts, images, metadata, manifest.metrics)
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
        )

        if config.output_dir:
            saved_path = result.save(config.output_dir)
            print(f"Results saved to {saved_path}")

        return result
