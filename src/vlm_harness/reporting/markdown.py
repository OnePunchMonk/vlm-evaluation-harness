"""Markdown report generator — same data as reporting/html.py, for CI comments
and README-friendly output."""

from __future__ import annotations

_SEVERITY_EMOJI = {
    "CRITICAL": "🔴",
    "HIGH": "🟠",
    "MEDIUM": "🟡",
    "LOW": "🔵",
    "MINIMAL": "⚪",
    "OK": "✅",
}


def build_leaderboard_markdown(results: list[dict]) -> str:
    metric_names: set[str] = set()
    for r in results:
        metric_names |= set(r.get("metrics", {}).keys())
    metric_cols = sorted(metric_names)

    header = ["Model", "Benchmark", *metric_cols, "Cost", "N"]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    for r in results:
        row = [r.get("model", ""), r.get("benchmark", "")]
        for name in metric_cols:
            val = r.get("metrics", {}).get(name)
            row.append(f"{val:.4f}" if val is not None else "—")
        cost = r.get("cost", {}).get("total_usd", 0.0)
        row.append(f"${cost:.4f}" if cost else "—")
        row.append(str(r.get("n_samples", 0)))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_regression_markdown(deltas: list, threshold: float = 0.03) -> str:
    if not deltas:
        return "No comparable runs found."

    flagged = [d for d in deltas if d.flagged]
    lines = [
        f"**Regression threshold:** {threshold:.1%} — **{len(flagged)}/{len(deltas)}** flagged",
        "",
        "| Benchmark | Metric | Baseline | Current | Delta | Severity |",
        "|---|---|---|---|---|---|",
    ]
    for d in deltas:
        emoji = _SEVERITY_EMOJI.get(d.severity, "")
        lines.append(
            f"| {d.benchmark} | {d.metric_name} | {d.baseline_value:.4f} | "
            f"{d.current_value:.4f} | {d.delta:+.4f} | {emoji} {d.severity} |"
        )
    return "\n".join(lines)


def build_report_markdown(
    results: list[dict], deltas: list | None = None, threshold: float = 0.03
) -> str:
    parts = ["## Leaderboard", "", build_leaderboard_markdown(results)]
    if deltas:
        parts += ["", "## Regression Report", "", build_regression_markdown(deltas, threshold)]
    return "\n".join(parts)
