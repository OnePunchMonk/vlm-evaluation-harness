"""VLM-Harness CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

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
    max_samples: Optional[int] = typer.Option(None, "--max-samples", help="Limit samples"),
    max_tokens: int = typer.Option(1024, "--max-tokens"),
    temperature: float = typer.Option(0.0, "--temperature"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o"),
    robustness: Optional[str] = typer.Option(None, "--corruptions", help="Comma-sep corruptions"),
):
    """Evaluate a model on one or more benchmarks."""
    from vlm_harness.adapters.registry import get_adapter
    from vlm_harness.engine.runner import EvalConfig, EvalRunner
    from vlm_harness.reporting.terminal import print_results

    adapter = get_adapter(model)
    runner = EvalRunner(adapter)
    benchmarks = [b.strip() for b in bench.split(",")]

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


@app.command()
def compare(
    models: str = typer.Option(..., "--models", help="Comma-separated model specs"),
    bench: str = typer.Option(..., "--bench", "-b", help="Benchmark name"),
    split: str = typer.Option("validation", "--split"),
    max_samples: Optional[int] = typer.Option(None, "--max-samples"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save HTML comparison"),
):
    """Compare multiple models on a benchmark."""
    from vlm_harness.adapters.registry import get_adapter
    from vlm_harness.engine.runner import EvalConfig, EvalRunner
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


@app.command("list-benchmarks")
def list_benchmarks(
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by category"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """List all available benchmarks."""
    from vlm_harness.benchmarks.registry import get_registry
    from rich.table import Table
    from rich import box

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
    from vlm_harness.adapters.registry import get_adapter
    from vlm_harness.benchmarks.registry import get_registry
    from datasets import load_dataset

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
    n_samples = "unknown"
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
    )


if __name__ == "__main__":
    app()
