"""Self-contained HTML report generator.

Builds a leaderboard across (model, benchmark, metric) plus an optional
regression section, styled after the vlm_regcheck/ prototype in this repo.
Works on plain dicts (as produced by EvalResult.to_dict() / GenEvalResult.to_dict()
or loaded back from saved *_results.json files) so it doesn't care whether
the runs were discriminative or generative.
"""

from __future__ import annotations

from pathlib import Path

_SEVERITY_COLOR = {
    "CRITICAL": "#ff4444",
    "HIGH": "#ff8800",
    "MEDIUM": "#ffcc00",
    "LOW": "#88aaff",
    "MINIMAL": "#cccccc",
    "OK": "#44bb44",
    "NOT_SIGNIFICANT": "#cccccc",
}

_STYLE = """
body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1100px;
       margin: 40px auto; padding: 0 20px; color: #222; }
h1 { color: #222; }
h2 { margin-top: 40px; border-bottom: 2px solid #eee; padding-bottom: 6px; }
.meta { background: #f5f5f5; padding: 12px 16px; border-radius: 6px; margin-bottom: 24px;
        font-size: 14px; color: #555; }
table { border-collapse: collapse; width: 100%; margin-top: 12px; }
th { background: #222; color: white; padding: 8px 12px; text-align: left; font-size: 13px; }
td { padding: 8px 12px; border-bottom: 1px solid #eee; font-size: 14px; }
tr:hover { background: #f9f9f9; }
.badge { padding: 2px 8px; border-radius: 4px; color: white; font-size: 12px; font-weight: 600; }
.ci { color: #888; font-size: 12px; }
"""


def _format_metric_cell(val: float | None, ci: list | None) -> str:
    if val is None:
        return "<td>—</td>"
    if ci and len(ci) == 2 and all(c == c for c in ci):  # c == c filters NaN
        return f"<td>{val:.4f} <span class='ci'>[{ci[0]:.4f}, {ci[1]:.4f}]</span></td>"
    return f"<td>{val:.4f}</td>"


def _leaderboard_rows(results: list[dict]) -> tuple[list[str], str]:
    metric_names: set[str] = set()
    for r in results:
        metric_names |= set(r.get("metrics", {}).keys())
    metric_cols = sorted(metric_names)

    rows = ""
    for r in results:
        cells = [f"<td>{r.get('model', '')}</td>", f"<td>{r.get('benchmark', '')}</td>"]
        ci95 = r.get("metric_ci95", {})
        for name in metric_cols:
            val = r.get("metrics", {}).get(name)
            cells.append(_format_metric_cell(val, ci95.get(name)))
        cost = r.get("cost", {}).get("total_usd", 0.0)
        cells.append(f"<td>${cost:.4f}</td>" if cost else "<td>—</td>")
        cells.append(f"<td>{r.get('n_samples', 0)}</td>")
        rows += f"<tr>{''.join(cells)}</tr>"
    return metric_cols, rows


def build_leaderboard_html(
    results: list[dict], title: str = "VLM-Evaluation-Harness Report"
) -> str:
    metric_cols, rows = _leaderboard_rows(results)
    header = "".join(f"<th>{m}</th>" for m in metric_cols)
    return f"""
<h2>Leaderboard</h2>
<table>
  <tr><th>Model</th><th>Benchmark</th>{header}<th>Cost</th><th>N</th></tr>
  {rows}
</table>
"""


def build_pareto_svg(
    results: list[dict],
    metric_name: str,
    x_field: str = "latency.p50_ms",
    width: int = 640,
    height: int = 420,
) -> str:
    """Hand-rolled SVG scatter of `metric_name` (y) vs. a cost/latency field
    (x, dotted path e.g. "cost.total_usd" or "latency.p50_ms"). Points on the
    Pareto frontier (no other model is both cheaper/faster AND better) are
    highlighted and connected. No plotting dependency: this project already
    keeps `stats.py` numpy-only, so a scatter plot earns a hand-rolled SVG
    rather than a new dependency.
    """
    x_keys = x_field.split(".")
    points = []
    for r in results:
        y = r.get("metrics", {}).get(metric_name)
        x: object = r
        for key in x_keys:
            x = x.get(key) if isinstance(x, dict) else None
        if y is None or x is None or (isinstance(y, float) and y != y):
            continue
        points.append((float(x), float(y), r.get("model", "")))

    margin = 56
    if not points:
        return (
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
            f'<text x="{margin}" y="{height // 2}">No comparable points for '
            f'"{metric_name}" vs "{x_field}".</text></svg>'
        )

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_span = (x_max - x_min) or 1.0
    y_span = (y_max - y_min) or 1.0

    def sx(x: float) -> float:
        return margin + (x - x_min) / x_span * (width - 2 * margin)

    def sy(y: float) -> float:
        return height - margin - (y - y_min) / y_span * (height - 2 * margin)

    # Pareto frontier: lower x (cheaper/faster) is better, higher y (metric)
    # is better. A point is on the frontier if no other point is both <= its
    # x and >= its y with at least one strict inequality.
    frontier = []
    for x, y, model in points:
        dominated = any(
            (ox <= x and oy >= y) and (ox < x or oy > y) for ox, oy, _ in points
        )
        if not dominated:
            frontier.append((x, y, model))
    frontier.sort(key=lambda p: p[0])

    circles = "".join(
        f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" '
        f'fill="{"#ff8800" if (x, y, model) in frontier else "#888"}">'
        f"<title>{model}: {metric_name}={y:.4f}, {x_field}={x:.4g}</title></circle>"
        for x, y, model in points
    )
    frontier_path = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y, _ in frontier)
    polyline = (
        f'<polyline points="{frontier_path}" fill="none" stroke="#ff8800" '
        f'stroke-width="1.5" stroke-dasharray="4,3"/>'
        if len(frontier) > 1
        else ""
    )

    return f"""
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg"
     font-family="-apple-system, sans-serif" font-size="11">
  <rect x="0" y="0" width="{width}" height="{height}" fill="none"/>
  <line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}"
        stroke="#ccc"/>
  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#ccc"/>
  <text x="{width / 2:.1f}" y="{height - 12}" text-anchor="middle">{x_field}</text>
  <text x="14" y="{height / 2:.1f}" text-anchor="middle"
        transform="rotate(-90 14 {height / 2:.1f})">{metric_name}</text>
  {polyline}
  {circles}
</svg>
"""


def build_regression_html(deltas: list, threshold: float = 0.03) -> str:
    if not deltas:
        return ""
    rows = ""
    for d in deltas:
        color = _SEVERITY_COLOR.get(d.severity, "#ffffff")
        rows += (
            f"<tr><td>{d.benchmark}</td><td>{d.metric_name}</td>"
            f"<td>{d.baseline_model}</td><td>{d.current_model}</td>"
            f"<td>{d.baseline_value:.4f}</td><td>{d.current_value:.4f}</td>"
            f"<td><b>{d.delta:+.4f}</b></td>"
            f"<td><span class='badge' style='background:{color}'>{d.severity}</span></td></tr>"
        )
    flagged = [d for d in deltas if d.flagged]
    return f"""
<h2>Regression Report</h2>
<div class="meta">Threshold: {threshold:.1%} &nbsp;|&nbsp;
  Flagged regressions: {len(flagged)} / {len(deltas)}</div>
<table>
  <tr><th>Benchmark</th><th>Metric</th><th>Baseline</th><th>Current</th>
      <th>Base value</th><th>Current value</th><th>Delta</th><th>Severity</th></tr>
  {rows}
</table>
"""


def save_html_report(
    results: list[dict],
    deltas: list | None = None,
    threshold: float = 0.03,
    path: str | Path = "report.html",
    title: str = "VLM-Evaluation-Harness Report",
) -> Path:
    sections = build_leaderboard_html(results, title)
    if deltas:
        sections += build_regression_html(deltas, threshold)

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>{title}</title><style>{_STYLE}</style></head>
<body>
<h1>{title}</h1>
<div class="meta">{len(results)} run(s) included.</div>
{sections}
</body>
</html>"""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path
