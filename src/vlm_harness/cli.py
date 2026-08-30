"""VLM-Harness CLI."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(
    name="vlm-harness",
    help="VLM-Harness: unified evaluation framework for Vision Language Models.",
    add_completion=False,
)
console = Console()


@app.command()
def eval(
    model: str = typer.Option(..., "--model", "-m", help="Model spec: 'provider:model_id'"),
    bench: str = typer.Option(..., "--bench", "-b", help="Benchmark name(s), comma-separated"),
    split: str = typer.Option("validation", "--split", help="Dataset split"),
    max_samples: int | None = typer.Option(None, "--max-samples", help="Limit samples"),
    max_tokens: int = typer.Option(1024, "--max-tokens"),
    temperature: float = typer.Option(0.0, "--temperature"),
    output_dir: Path | None = typer.Option(None, "--output-dir", "-o"),
    robustness: str | None = typer.Option(None, "--corruptions", help="Comma-sep corruptions"),
    track: bool = typer.Option(True, "--track/--no-track", help="Record this run to local history"),
):
    """Evaluate a model on one or more benchmarks."""
    from vlm_harness.adapters.registry import get_adapter
    from vlm_harness.engine.runner import EvalConfig, EvalRunner
    from vlm_harness.reporting.terminal import print_results
    from vlm_harness.tracking import HistoryStore

    adapter = get_adapter(model)
    runner = EvalRunner(adapter)
    benchmarks = [b.strip() for b in bench.split(",")]
    history = HistoryStore()

    for benchmark in benchmarks:
        config = EvalConfig(
            model_spec=model,
            benchmark=benchmark,
            split=split,
            max_samples=max_samples,
            max_tokens=max_tokens,
            temperature=temperature,
            output_dir=output_dir,
            robustness_corruptions=[c.strip() for c in robustness.split(",")] if robustness else [],
        )
        result = runner.run(config)
        print_results(result)
        if track:
            history.record_result(result, modality="discriminative")


@app.command()
def compare(
    models: str = typer.Option(..., "--models", help="Comma-separated model specs"),
    bench: str = typer.Option(..., "--bench", "-b", help="Benchmark name"),
    split: str = typer.Option("validation", "--split"),
    max_samples: int | None = typer.Option(None, "--max-samples"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Save HTML comparison"),
):
    """Compare multiple models on a benchmark."""
    from vlm_harness.adapters.registry import get_adapter
    from vlm_harness.engine.runner import EvalConfig, EvalRunner
    from vlm_harness.reporting.html import save_html_report
    from vlm_harness.reporting.terminal import print_comparison

    model_specs = [m.strip() for m in models.split(",")]
    results = []

    for model_spec in model_specs:
        adapter = get_adapter(model_spec)
        runner = EvalRunner(adapter)
        config = EvalConfig(
            model_spec=model_spec,
            benchmark=bench,
            split=split,
            max_samples=max_samples,
        )
        results.append(runner.run(config))

    print_comparison(results, bench)

    if output:
        path = save_html_report(
            [r.to_dict() for r in results], title=f"Model Comparison — {bench}", path=output
        )
        console.print(f"[green]HTML comparison saved to {path}[/green]")


@app.command("list-benchmarks")
def list_benchmarks(
    category: str | None = typer.Option(None, "--category", "-c", help="Filter by category"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """List all available benchmarks."""
    from rich import box
    from rich.table import Table

    from vlm_harness.benchmarks.registry import get_registry

    registry = get_registry()

    if category:
        by_cat = registry.list_by_category()
        names = by_cat.get(category, [])
    else:
        names = registry.list()

    if verbose:
        table = Table(box=box.ROUNDED)
        table.add_column("Name", style="cyan")
        table.add_column("Category")
        table.add_column("Modality")
        table.add_column("Task Type")
        for name in names:
            m = registry.get(name)
            table.add_row(name, m.taxonomy_category, m.modality, m.task_type)
        console.print(table)
    else:
        by_cat = registry.list_by_category()
        for cat, cat_names in by_cat.items():
            console.print(f"[bold cyan]{cat}[/bold cyan]")
            for n in cat_names:
                if not category or cat == category:
                    console.print(f"  {n}")


@app.command("validate-bench")
def validate_bench(
    bench: str = typer.Option(..., "--bench", "-b"),
):
    """Validate a benchmark manifest."""
    from vlm_harness.benchmarks.registry import get_registry

    try:
        registry = get_registry()
        manifest = registry.get(bench)
        console.print(f"[green]✓[/green] Manifest for '{manifest.name}' is valid.")
        console.print(f"  Task type: {manifest.task_type}")
        console.print(f"  Modality:  {manifest.modality}")
        console.print(f"  Splits:    {[s.name for s in manifest.splits]}")
        console.print(f"  Metrics:   {[m.type for m in manifest.metrics]}")
    except KeyError as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1)


@app.command("estimate-cost")
def estimate_cost(
    model: str = typer.Option(..., "--model", "-m"),
    bench: str = typer.Option(..., "--bench", "-b"),
    split: str = typer.Option("validation", "--split"),
    avg_input_tokens: int = typer.Option(500, "--avg-input-tokens"),
    avg_output_tokens: int = typer.Option(50, "--avg-output-tokens"),
):
    """Estimate the cost of an evaluation run before executing it."""
    from datasets import load_dataset

    from vlm_harness.adapters.registry import get_adapter
    from vlm_harness.benchmarks.registry import get_registry

    adapter = get_adapter(model)
    registry = get_registry()
    manifest = registry.get(bench)

    split_config = next((s for s in manifest.splits if s.name == split), None)
    if not split_config:
        console.print(f"[red]Split '{split}' not found in benchmark '{bench}'[/red]")
        raise typer.Exit(1)

    cost_in = adapter.cost_per_million_input_tokens
    cost_out = adapter.cost_per_million_output_tokens

    if cost_in is None and cost_out is None:
        console.print("[yellow]Local model — no API cost.[/yellow]")
        return

    # Try to get dataset size
    n_samples: int | str = "unknown"
    try:
        ds = load_dataset(manifest.source.path, manifest.source.subset, split=split)
        n_samples = len(ds)
    except Exception:
        pass

    if isinstance(n_samples, int):
        total_in = n_samples * avg_input_tokens
        total_out = n_samples * avg_output_tokens
        cost = 0.0
        if cost_in:
            cost += (total_in / 1_000_000) * cost_in
        if cost_out:
            cost += (total_out / 1_000_000) * cost_out
        console.print(f"\n[bold]Estimated cost for {manifest.name} / {split}[/bold]")
        console.print(f"  Samples:     {n_samples:,}")
        console.print(f"  Input rate:  ${cost_in or 0:.2f}/1M tokens")
        console.print(f"  Output rate: ${cost_out or 0:.2f}/1M tokens")
        console.print(f"  [green]Estimated total: ${cost:.4f} USD[/green]")
    else:
        console.print("[yellow]Could not determine dataset size automatically.[/yellow]")


@app.command()
def reproduce(
    manifest_file: Path = typer.Argument(..., help="Path to a results manifest JSON"),
):
    """Re-run an evaluation from a saved results manifest."""
    import json

    data = json.loads(manifest_file.read_text())
    console.print(f"Reproducing: {data.get('benchmark')} with {data.get('model')}")
    eval(
        model=data["model"],
        bench=data["benchmark"],
        split=data["split"],
        max_samples=None,
        max_tokens=1024,
        temperature=0.0,
        output_dir=None,
        robustness=None,
        track=True,
    )


@app.command("gen-eval")
def gen_eval(
    model: str = typer.Option(..., "--model", "-m", help="T2I model spec: 'provider:model_id'"),
    bench: str = typer.Option(..., "--bench", "-b", help="Benchmark name(s), comma-separated"),
    split: str = typer.Option("prompts", "--split", help="Dataset split"),
    max_samples: int | None = typer.Option(None, "--max-samples", help="Limit samples"),
    width: int = typer.Option(512, "--width"),
    height: int = typer.Option(512, "--height"),
    seed: int | None = typer.Option(42, "--seed"),
    output_dir: Path | None = typer.Option(None, "--output-dir", "-o"),
    track: bool = typer.Option(True, "--track/--no-track", help="Record this run to local history"),
):
    """Evaluate a text-to-image model on one or more generative benchmarks."""
    from vlm_harness.adapters.generative.registry import get_t2i_adapter
    from vlm_harness.engine.generative_runner import GenerativeEvalRunner, GenEvalConfig
    from vlm_harness.reporting.terminal import print_results
    from vlm_harness.tracking import HistoryStore

    adapter = get_t2i_adapter(model)
    runner = GenerativeEvalRunner(adapter)
    benchmarks = [b.strip() for b in bench.split(",")]
    history = HistoryStore()

    for benchmark in benchmarks:
        config = GenEvalConfig(
            model_spec=model,
            benchmark=benchmark,
            split=split,
            max_samples=max_samples,
            width=width,
            height=height,
            seed=seed,
            output_dir=output_dir,
        )
        result = runner.run(config)
        print_results(result)
        if track:
            history.record_result(result, modality="generative")


@app.command()
def history(
    model: str | None = typer.Option(None, "--model", "-m"),
    bench: str | None = typer.Option(None, "--bench", "-b"),
):
    """List tracked evaluation runs."""
    from rich import box
    from rich.table import Table

    from vlm_harness.tracking import HistoryStore

    store = HistoryStore()
    entries = store.query(model=model, benchmark=bench)

    if not entries:
        console.print("[yellow]No tracked runs found.[/yellow]")
        return

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    table.add_column("Timestamp", style="dim")
    table.add_column("Model", style="cyan")
    table.add_column("Benchmark")
    table.add_column("Modality")
    table.add_column("Metrics")
    table.add_column("N", justify="right")

    for e in entries:
        metrics_str = ", ".join(f"{k}={v:.3f}" for k, v in e.metrics.items())
        table.add_row(e.timestamp, e.model, e.benchmark, e.modality, metrics_str, str(e.n_samples))

    console.print(table)


@app.command()
def regression(
    baseline: str = typer.Option(..., "--baseline", help="Baseline model spec"),
    current: str = typer.Option(
        ..., "--current", "--model", help="Model spec to check for regressions"
    ),
    bench: str | None = typer.Option(
        None, "--bench", "-b", help="Comma-separated benchmarks (default: all shared)"
    ),
    threshold: float = typer.Option(
        0.03, "--threshold", help="Fractional drop that counts as a regression"
    ),
    output: Path | None = typer.Option(None, "--output", "-o", help="Save report (.html or .md)"),
):
    """Compare the latest tracked runs of two models and flag regressions."""
    from vlm_harness.reporting.html import save_html_report
    from vlm_harness.reporting.markdown import build_regression_markdown
    from vlm_harness.tracking import HistoryStore, compare_models

    store = HistoryStore()
    benchmarks = [b.strip() for b in bench.split(",")] if bench else None
    deltas = compare_models(store, baseline, current, benchmarks=benchmarks, threshold=threshold)

    if not deltas:
        console.print(
            f"[yellow]No comparable tracked runs found for '{baseline}' vs '{current}'. "
            "Run `vlm-harness eval --track` for both models first.[/yellow]"
        )
        raise typer.Exit(1)

    from rich import box
    from rich.table import Table

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold",
        title=f"{baseline}  →  {current}  (threshold {threshold:.1%})",
    )
    table.add_column("Benchmark", style="cyan")
    table.add_column("Metric")
    table.add_column("Baseline", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Severity")

    severity_style = {
        "CRITICAL": "bold red",
        "HIGH": "red",
        "MEDIUM": "yellow",
        "LOW": "blue",
        "MINIMAL": "dim",
        "OK": "green",
    }
    for d in deltas:
        style = severity_style.get(d.severity, "")
        table.add_row(
            d.benchmark,
            d.metric_name,
            f"{d.baseline_value:.4f}",
            f"{d.current_value:.4f}",
            f"{d.delta:+.4f}",
            f"[{style}]{d.severity}[/{style}]",
        )
    console.print(table)

    flagged = [d for d in deltas if d.flagged]
    if flagged:
        console.print(f"\n[bold red]{len(flagged)} flagged regression(s)[/bold red]")
    else:
        console.print("\n[bold green]No flagged regressions.[/bold green]")

    if output:
        if output.suffix == ".md":
            output.write_text(build_regression_markdown(deltas, threshold))
        else:
            save_html_report(
                [],
                deltas=deltas,
                threshold=threshold,
                path=output,
                title=f"Regression Report — {baseline} vs {current}",
            )
        console.print(f"[green]Report saved to {output}[/green]")

    if flagged:
        raise typer.Exit(1)


@app.command()
def report(
    results_dir: Path = typer.Option(
        ..., "--results-dir", help="Directory of saved *_results.json files"
    ),
    output: Path = typer.Option(Path("report.html"), "--output", "-o"),
    format: str = typer.Option("html", "--format", help="'html' or 'markdown'"),
):
    """Aggregate saved evaluation results into a single leaderboard report."""
    import json

    from vlm_harness.reporting.html import save_html_report
    from vlm_harness.reporting.markdown import build_report_markdown

    result_files = sorted(results_dir.glob("*_results.json"))
    if not result_files:
        console.print(f"[yellow]No *_results.json files found in {results_dir}[/yellow]")
        raise typer.Exit(1)

    results = [json.loads(f.read_text()) for f in result_files]

    if format == "markdown":
        output.write_text(build_report_markdown(results))
    else:
        save_html_report(results, path=output, title="VLM-Harness Report")

    console.print(f"[green]Report saved to {output}[/green] ({len(results)} run(s))")


if __name__ == "__main__":
    app()
