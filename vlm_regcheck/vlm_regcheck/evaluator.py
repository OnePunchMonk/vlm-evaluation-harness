"""
RegressionEvaluator  —  runs base + fine-tuned model on all benchmarks
and produces a RegressionReport.

Design:
  - Loads base model ONCE, runs all benchmarks, unloads.
  - Loads fine-tuned model ONCE, runs all benchmarks, unloads.
  - Computes deltas and flags regressions.
  - Caches results to JSON so partial runs can be resumed.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch

from .benchmarks.base import Benchmark, BenchmarkResult, Sample

_DEFAULT_SUITE_NAMES = [
    "mmstar", "mmvp", "mathvista", "pope",
    "cvbench", "ocrbench", "mmmu_pro", "winoground",
]
_FAST_SUITE_NAMES = ["mmstar", "pope", "mmvp", "cvbench"]
from .models import VLMWrapper
from .report import RegressionReport

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    base_model_id: str
    finetuned_model_id: str
    benchmarks: list[str]              # benchmark names to run
    n_samples: Optional[int] = 200    # samples per benchmark (None = all)
    strategy: str = "generate"         # "generate" or "logprob"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    device: str = "auto"
    dtype: str = "bfloat16"
    cache_dir: Optional[str] = None   # save partial results here
    verbose: bool = False
    regression_threshold: float = 0.03 # flag drops > 3%
    fast: bool = False                 # use FAST_SUITE subset


class RegressionEvaluator:
    """
    Usage:
        evaluator = RegressionEvaluator(EvalConfig(
            base_model_id="llava-hf/llava-1.5-7b-hf",
            finetuned_model_id="./my-finetuned-vlm",
            benchmarks=["mmstar", "pope", "mmvp", "cvbench"],
            n_samples=200,
        ))
        report = evaluator.run()
        report.print_summary()
    """

    def __init__(self, config: EvalConfig):
        self.config = config
        self._dtype = getattr(torch, config.dtype, torch.bfloat16)

        # Resolve benchmark list
        if config.fast:
            names = _FAST_SUITE_NAMES
        elif config.benchmarks:
            names = config.benchmarks
        else:
            names = _DEFAULT_SUITE_NAMES
        self.benchmark_names = names

        self._cache_path = (
            Path(config.cache_dir) if config.cache_dir else Path(".vlm_regcheck_cache")
        )
        self._cache_path.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> RegressionReport:
        """Run full regression evaluation. Returns a RegressionReport."""
        logger.info("=== VLM Regression Evaluator ===")
        logger.info(f"  Base model    : {self.config.base_model_id}")
        logger.info(f"  Fine-tuned    : {self.config.finetuned_model_id}")
        logger.info(f"  Benchmarks    : {self.benchmark_names}")
        logger.info(f"  Samples/bench : {self.config.n_samples}")

        base_results = self._run_model(self.config.base_model_id, tag="base")
        ft_results = self._run_model(self.config.finetuned_model_id, tag="finetuned")

        return RegressionReport(
            base_model_id=self.config.base_model_id,
            finetuned_model_id=self.config.finetuned_model_id,
            base_results=base_results,
            ft_results=ft_results,
            threshold=self.config.regression_threshold,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _cache_file(self, model_tag: str, benchmark_name: str) -> Path:
        safe_tag = model_tag.replace("/", "_").replace("\\", "_")
        return self._cache_path / f"{safe_tag}__{benchmark_name}.json"

    def _load_cache(self, model_tag: str, benchmark_name: str) -> Optional[BenchmarkResult]:
        path = self._cache_file(model_tag, benchmark_name)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            result = BenchmarkResult(
                benchmark=data["benchmark"],
                capability=data["capability"],
                accuracy=data["accuracy"],
                n_samples=data["n_samples"],
                sota_score=data["sota_score"],
                per_sample=data.get("per_sample", []),
            )
            logger.info(f"  [cache] {benchmark_name}: {result.accuracy:.3f}")
            return result
        except Exception as e:
            logger.warning(f"Cache load failed for {path}: {e}")
            return None

    def _save_cache(self, model_tag: str, result: BenchmarkResult):
        path = self._cache_file(model_tag, result.benchmark)
        try:
            with open(path, "w") as f:
                json.dump({
                    "benchmark": result.benchmark,
                    "capability": result.capability,
                    "accuracy": result.accuracy,
                    "n_samples": result.n_samples,
                    "sota_score": result.sota_score,
                    "per_sample": result.per_sample,
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")

    def _run_model(
        self, model_id: str, tag: str
    ) -> dict[str, BenchmarkResult]:
        """Load model, run all benchmarks, unload model. Returns results dict."""
        results: dict[str, BenchmarkResult] = {}

        # Check which benchmarks are already cached
        missing = []
        for name in self.benchmark_names:
            cached = self._load_cache(tag, name)
            if cached is not None:
                results[name] = cached
            else:
                missing.append(name)

        if not missing:
            logger.info(f"All benchmarks cached for {tag}.")
            return results

        # Load model only if there are benchmarks to run
        logger.info(f"\nLoading {tag} model: {model_id}")
        model = VLMWrapper(
            model_id=model_id,
            device=self.config.device,
            dtype=self._dtype,
            load_in_4bit=self.config.load_in_4bit,
            load_in_8bit=self.config.load_in_8bit,
            strategy=self.config.strategy,
        )

        for bench_name in missing:
            logger.info(f"\n  Running benchmark: {bench_name}")
            from .benchmarks import get_benchmark
            benchmark = get_benchmark(bench_name)

            try:
                t0 = time.time()
                samples = benchmark.load(n_samples=self.config.n_samples)
                logger.info(f"    Loaded {len(samples)} samples")

                result: BenchmarkResult = benchmark.evaluate(
                    model, samples, verbose=self.config.verbose
                )
                elapsed = time.time() - t0

                logger.info(
                    f"    {bench_name}: {result.accuracy:.3f} "
                    f"(SOTA {result.sota_score:.2f}, "
                    f"headroom {result.headroom:.2f}) [{elapsed:.0f}s]"
                )
                results[bench_name] = result
                self._save_cache(tag, result)

            except Exception as e:
                logger.error(f"    FAILED {bench_name}: {e}", exc_info=True)
                # Create a null result so we don't block the report
                results[bench_name] = BenchmarkResult(
                    benchmark=bench_name,
                    capability=getattr(benchmark, "capability", "unknown"),
                    accuracy=float("nan"),
                    n_samples=0,
                    sota_score=getattr(benchmark, "sota_score", 0.0),
                )

        # Explicitly release GPU memory
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results
