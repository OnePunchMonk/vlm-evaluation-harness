"""
CLI entry point for vlm-regcheck.

Usage examples:

  # Full eval with default benchmark suite
  vlm-regcheck \\
      --base llava-hf/llava-1.5-7b-hf \\
      --finetuned ./my-medical-vlm \\
      --output report.json

  # Fast check (4 high-signal benchmarks)
  vlm-regcheck \\
      --base Qwen/Qwen2-VL-7B-Instruct \\
      --finetuned ./my-finetuned \\
      --fast \\
      --n-samples 100

  # Specific benchmarks only
  vlm-regcheck \\
      --base llava-hf/llava-1.5-7b-hf \\
      --finetuned ./my-vlm \\
      --benchmarks mmstar,pope,mmvp,mathvista \\
      --n-samples 200 \\
      --load-in-4bit

  # 4-bit quantized models (saves memory)
  vlm-regcheck \\
      --base meta-llama/Llama-3.2-11B-Vision-Instruct \\
      --finetuned ./my-llama-vision \\
      --load-in-4bit \\
      --output report.html --html
"""

from __future__ import annotations

import logging
import sys

import click

from .evaluator import EvalConfig, RegressionEvaluator

# Benchmark names only — no class import needed at CLI level
_BENCHMARK_NAMES = [
    "mmmu_pro", "mathvista", "mmstar", "pope",
    "winoground", "mmvp", "cvbench", "ocrbench",
]


@click.command()
@click.option("--base", required=True,
              help="HuggingFace model ID or local path for the BASE model.")
@click.option("--finetuned", required=True,
              help="HuggingFace model ID or local path for the FINE-TUNED model.")
@click.option("--benchmarks", default=None,
              help=f"Comma-separated benchmarks. Available: {','.join(_BENCHMARK_NAMES)}. "
                   "Default: all.")
@click.option("--n-samples", default=200, type=int, show_default=True,
              help="Number of samples per benchmark. Use 0 for all samples (slow).")
@click.option("--fast", is_flag=True,
              help="Use fast suite (mmstar, pope, mmvp, cvbench) — high signal, lower compute.")
@click.option("--strategy", default="generate", type=click.Choice(["generate", "logprob"]),
              show_default=True,
              help="Answer strategy. 'logprob' is more accurate for MC but slower.")
@click.option("--load-in-4bit", is_flag=True,
              help="Load models in 4-bit (requires bitsandbytes). Reduces GPU memory ~4x.")
@click.option("--load-in-8bit", is_flag=True,
              help="Load models in 8-bit (requires bitsandbytes). Reduces GPU memory ~2x.")
@click.option("--device", default="auto", show_default=True,
              help="Device for model loading. 'auto' uses device_map='auto'.")
@click.option("--dtype", default="bfloat16",
              type=click.Choice(["bfloat16", "float16", "float32"]),
              show_default=True, help="Model dtype.")
@click.option("--threshold", default=0.03, type=float, show_default=True,
              help="Regression threshold (0.03 = flag drops >3%).")
@click.option("--output", default=None,
              help="Save report to this path (.json or .html).")
@click.option("--html", is_flag=True,
              help="Generate HTML report (auto-detected from --output if .html extension).")
@click.option("--cache-dir", default=".vlm_regcheck_cache", show_default=True,
              help="Directory to cache per-benchmark results (enables resuming).")
@click.option("--verbose", is_flag=True, help="Print per-sample predictions.")
@click.option("--log-level", default="INFO",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
              show_default=True)
def main(
    base, finetuned, benchmarks, n_samples, fast, strategy,
    load_in_4bit, load_in_8bit, device, dtype, threshold,
    output, html, cache_dir, verbose, log_level,
):
    """
    VLM Regression Checker — evaluate post-training regressions across
    hard, non-saturated vision-language benchmarks.

    Compares BASE and FINE-TUNED models and reports which capabilities
    degraded and by how much.
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    bench_list = None
    if benchmarks:
        bench_list = [b.strip() for b in benchmarks.split(",")]
        unknown = [b for b in bench_list if b not in _BENCHMARK_NAMES]
        if unknown:
            click.echo(f"Unknown benchmarks: {unknown}", err=True)
            click.echo(f"Available: {_BENCHMARK_NAMES}", err=True)
            sys.exit(1)

    config = EvalConfig(
        base_model_id=base,
        finetuned_model_id=finetuned,
        benchmarks=bench_list or _BENCHMARK_NAMES,
        n_samples=n_samples if n_samples > 0 else None,
        strategy=strategy,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        device=device,
        dtype=dtype,
        cache_dir=cache_dir,
        verbose=verbose,
        regression_threshold=threshold,
        fast=fast,
    )

    evaluator = RegressionEvaluator(config)
    report = evaluator.run()
    report.print_summary()

    if output:
        if output.endswith(".html") or html:
            report.to_html(output if output.endswith(".html") else output + ".html")
        else:
            report.to_json(output)
            click.echo(f"Report saved to {output}")
    elif html:
        report.to_html("regression_report.html")


if __name__ == "__main__":
    main()
