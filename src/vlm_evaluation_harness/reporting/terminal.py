"""Rich terminal reporting."""

from __future__ import annotations

from rich import box
from rich.console import Console
from rich.table import Table

console = Console()


def print_results(result) -> None:
    """Print a formatted eval result summary to the terminal."""
    console.print()
    console.rule(f"[bold cyan]{result.manifest.name}[/bold cyan] — {result.config.model_spec}")

    # Metrics table
    cis = result.metric_confidence_intervals()
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    table.add_column("95% CI", justify="right", style="dim")
    table.add_column("N Scored", justify="right")

    for m in result.metrics:
        ci = cis.get(m.metric_name)
        ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci and ci[0] == ci[0] else "—"
        table.add_row(m.metric_name, f"{m.value:.4f}", ci_str, f"{m.n_scored}/{m.n_samples}")

    console.print(table)

    # Per-group breakdowns
    for m in result.metrics:
        if m.breakdown:
            console.print(f"\n[bold]{m.metric_name} breakdown:[/bold]")
            btable = Table(box=box.SIMPLE, show_header=True)
            btable.add_column("Group", style="dim")
            btable.add_column("Score", justify="right")
            for group, score in sorted(m.breakdown.items()):
                btable.add_row(group, f"{score:.4f}")
            console.print(btable)

    # Cost / latency
    cs = result.cost_summary
    if cs.total_cost_usd > 0:
        console.print(
            f"\n[bold]Cost:[/bold] ${cs.total_cost_usd:.4f} total  "
            f"(${cs.cost_per_sample_usd:.6f}/sample)  "
            f"| {cs.total_input_tokens:,} in + {cs.total_output_tokens:,} out tokens"
        )
    console.print(
        f"[bold]Latency:[/bold] p50={cs.latency_p50_ms:.0f}ms  "
        f"p95={cs.latency_p95_ms:.0f}ms  "
        f"p99={cs.latency_p99_ms:.0f}ms  "
        f"| {cs.throughput_samples_per_min:.1f} samples/min"
    )
    console.print()


def print_comparison(results: list, benchmark: str) -> None:
    """Print a side-by-side model comparison table."""
    if not results:
        return

    all_metrics = {m.metric_name for r in results for m in r.metrics}

    console.print()
    console.rule(f"[bold cyan]Model Comparison — {benchmark}[/bold cyan]")
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    table.add_column("Model", style="cyan")
    for metric in sorted(all_metrics):
        table.add_column(metric, justify="right")
    table.add_column("Cost/sample $", justify="right", style="yellow")
    table.add_column("p50 ms", justify="right")

    for r in results:
        metric_map = {m.metric_name: m.value for m in r.metrics}
        row = [r.config.model_spec]
        for metric in sorted(all_metrics):
            val = metric_map.get(metric)
            row.append(f"{val:.4f}" if val is not None else "—")
        cs = r.cost_summary
        row.append(f"{cs.cost_per_sample_usd:.6f}" if cs.total_cost_usd > 0 else "—")
        row.append(f"{cs.latency_p50_ms:.0f}")
        table.add_row(*row)

    console.print(table)
    console.print()
