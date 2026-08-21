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
"""


def _leaderboard_rows(results: list[dict]) -> tuple[list[str], str]:
    metric_names: set[str] = set()
    for r in results:
        metric_names |= set(r.get("metrics", {}).keys())
    metric_cols = sorted(metric_names)

    rows = ""
    for r in results:
        cells = [f"<td>{r.get('model', '')}</td>", f"<td>{r.get('benchmark', '')}</td>"]
        for name in metric_cols:
            val = r.get("metrics", {}).get(name)
            cells.append(f"<td>{val:.4f}</td>" if val is not None else "<td>—</td>")
        cost = r.get("cost", {}).get("total_usd", 0.0)
        cells.append(f"<td>${cost:.4f}</td>" if cost else "<td>—</td>")
        cells.append(f"<td>{r.get('n_samples', 0)}</td>")
        rows += f"<tr>{''.join(cells)}</tr>"
    return metric_cols, rows


def build_leaderboard_html(results: list[dict], title: str = "VLM-Harness Report") -> str:
    metric_cols, rows = _leaderboard_rows(results)
    header = "".join(f"<th>{m}</th>" for m in metric_cols)
    return f"""
<h2>Leaderboard</h2>
<table>
  <tr><th>Model</th><th>Benchmark</th>{header}<th>Cost</th><th>N</th></tr>
  {rows}
</table>
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
    title: str = "VLM-Harness Report",
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
