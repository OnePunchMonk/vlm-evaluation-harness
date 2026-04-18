"""
RegressionReport  —  compares base vs fine-tuned results and surfaces regressions.

Output formats:
  - print_summary()     terminal table with colour coding
  - to_dict()           machine-readable dict
  - to_json(path)       save to file
  - to_html(path)       save browsable HTML report
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .benchmarks import BenchmarkResult


# ── Severity thresholds ───────────────────────────────────────────────────────
#
# These are intentionally conservative relative to typical saturation noise.
# A 3% drop on a non-saturated benchmark (MMVP SOTA=38%) is meaningful.
# The same 3% drop on VQAv2 (SOTA=86%) would be within noise — but we
# don't include saturated benchmarks in the suite.

SEVERITY_CRITICAL = -0.10   # >10% drop
SEVERITY_HIGH     = -0.05   # 5-10% drop
SEVERITY_MEDIUM   = -0.03   # 3-5% drop
SEVERITY_LOW      = -0.01   # 1-3% drop
SEVERITY_NONE     = 0.0     # no regression


def _severity_label(delta: float) -> str:
    if delta <= SEVERITY_CRITICAL:
        return "CRITICAL"
    if delta <= SEVERITY_HIGH:
        return "HIGH"
    if delta <= SEVERITY_MEDIUM:
        return "MEDIUM"
    if delta <= SEVERITY_LOW:
        return "LOW"
    if delta < 0:
        return "MINIMAL"
    return "OK"


def _severity_emoji(label: str) -> str:
    return {
        "CRITICAL": "🔴",
        "HIGH":     "🟠",
        "MEDIUM":   "🟡",
        "LOW":      "🔵",
        "MINIMAL":  "⚪",
        "OK":       "✅",
    }.get(label, "  ")


@dataclass
class BenchmarkDelta:
    benchmark: str
    capability: str
    base_accuracy: float
    ft_accuracy: float
    delta: float
    sota_score: float
    n_samples: int
    severity: str

    @property
    def is_regression(self) -> bool:
        return self.delta < 0 and not math.isnan(self.delta)

    @property
    def failed(self) -> bool:
        return math.isnan(self.base_accuracy) or math.isnan(self.ft_accuracy)


@dataclass
class RegressionReport:
    base_model_id: str
    finetuned_model_id: str
    base_results: dict[str, BenchmarkResult]
    ft_results: dict[str, BenchmarkResult]
    threshold: float = 0.03

    def _compute_deltas(self) -> list[BenchmarkDelta]:
        deltas = []
        for name in self.base_results:
            if name not in self.ft_results:
                continue
            base = self.base_results[name]
            ft = self.ft_results[name]
            delta = ft.accuracy - base.accuracy
            severity = _severity_label(delta)
            deltas.append(BenchmarkDelta(
                benchmark=name,
                capability=base.capability,
                base_accuracy=base.accuracy,
                ft_accuracy=ft.accuracy,
                delta=delta,
                sota_score=base.sota_score,
                n_samples=base.n_samples,
                severity=severity,
            ))
        # Sort: worst regressions first
        return sorted(deltas, key=lambda d: d.delta)

    def flagged_regressions(self) -> list[BenchmarkDelta]:
        return [d for d in self._compute_deltas()
                if d.delta < -self.threshold and not d.failed]

    def print_summary(self):
        deltas = self._compute_deltas()

        print("\n" + "=" * 72)
        print(" VLM REGRESSION REPORT")
        print("=" * 72)
        print(f" Base model      : {self.base_model_id}")
        print(f" Fine-tuned model: {self.finetuned_model_id}")
        print(f" Threshold       : {self.threshold:.1%}")
        print("=" * 72)

        # Header
        print(f"\n{'Benchmark':<20} {'Capability':<32} {'Base':>6} {'FT':>6} {'Delta':>7} {'Severity'}")
        print("-" * 80)

        for d in deltas:
            if d.failed:
                status = "  FAILED"
                delta_str = "  N/A"
                base_str = "  N/A"
                ft_str = "  N/A"
            else:
                emoji = _severity_emoji(d.severity)
                status = f"{emoji} {d.severity:<8}"
                delta_str = f"{d.delta:+.1%}"
                base_str = f"{d.base_accuracy:.1%}"
                ft_str = f"{d.ft_accuracy:.1%}"

            print(f"{d.benchmark:<20} {d.capability:<32} {base_str:>6} {ft_str:>6} {delta_str:>7}  {status}")

        print("-" * 80)

        # Summary
        regressions = self.flagged_regressions()
        print(f"\nFlagged regressions (>{self.threshold:.1%} drop): {len(regressions)}")

        if regressions:
            print("\nPRIORITY REGRESSIONS:")
            for d in regressions:
                print(f"  [{d.severity}] {d.benchmark} ({d.capability}): "
                      f"{d.base_accuracy:.1%} → {d.ft_accuracy:.1%} "
                      f"({d.delta:+.1%})")
                print(f"    SOTA is {d.sota_score:.1%} — delta is "
                      f"{abs(d.delta)/d.sota_score:.0%} of SOTA score")
        else:
            print("  No significant regressions detected.")

        # Improvements
        improvements = [d for d in deltas if d.delta > self.threshold and not d.failed]
        if improvements:
            print(f"\nIMPROVEMENTS (>{self.threshold:.1%} gain):")
            for d in improvements:
                print(f"  [+] {d.benchmark}: {d.base_accuracy:.1%} → {d.ft_accuracy:.1%} "
                      f"({d.delta:+.1%})")

        print("=" * 72 + "\n")

    def to_dict(self) -> dict:
        deltas = self._compute_deltas()
        return {
            "base_model": self.base_model_id,
            "finetuned_model": self.finetuned_model_id,
            "threshold": self.threshold,
            "n_flagged_regressions": len(self.flagged_regressions()),
            "benchmarks": [
                {
                    "benchmark": d.benchmark,
                    "capability": d.capability,
                    "base_accuracy": d.base_accuracy,
                    "ft_accuracy": d.ft_accuracy,
                    "delta": d.delta,
                    "sota_score": d.sota_score,
                    "severity": d.severity,
                    "n_samples": d.n_samples,
                    "flagged": d.delta < -self.threshold,
                }
                for d in deltas
            ],
        }

    def to_json(self, path: Optional[str] = None) -> str:
        data = self.to_dict()
        out = json.dumps(data, indent=2)
        if path:
            Path(path).write_text(out)
        return out

    def to_html(self, path: str = "regression_report.html"):
        """Generate a self-contained HTML report."""
        deltas = self._compute_deltas()

        color_map = {
            "CRITICAL": "#ff4444",
            "HIGH":     "#ff8800",
            "MEDIUM":   "#ffcc00",
            "LOW":      "#88aaff",
            "MINIMAL":  "#cccccc",
            "OK":       "#44bb44",
        }

        rows = ""
        for d in deltas:
            color = color_map.get(d.severity, "#ffffff")
            if d.failed:
                rows += f"""
                <tr>
                    <td>{d.benchmark}</td>
                    <td>{d.capability}</td>
                    <td>N/A</td><td>N/A</td><td>N/A</td>
                    <td style="color:gray">FAILED</td>
                </tr>"""
            else:
                rows += f"""
                <tr>
                    <td>{d.benchmark}</td>
                    <td>{d.capability}</td>
                    <td>{d.base_accuracy:.1%}</td>
                    <td>{d.ft_accuracy:.1%}</td>
                    <td><b>{d.delta:+.1%}</b></td>
                    <td style="color:{color}"><b>{d.severity}</b></td>
                </tr>"""

        flagged = self.flagged_regressions()
        summary_rows = ""
        for d in flagged:
            summary_rows += f"""
            <tr style="background:#fff0f0">
                <td><b>{d.benchmark}</b></td>
                <td>{d.capability}</td>
                <td>{d.delta:+.1%}</td>
                <td>{d.severity}</td>
                <td>SOTA is {d.sota_score:.1%}; delta = {abs(d.delta)/d.sota_score:.0%} of SOTA</td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>VLM Regression Report</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 20px; }}
  h1 {{ color: #333; }}
  .meta {{ background: #f5f5f5; padding: 12px; border-radius: 6px; margin-bottom: 24px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th {{ background: #333; color: white; padding: 8px 12px; text-align: left; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #eee; }}
  tr:hover {{ background: #f9f9f9; }}
  .section {{ margin-top: 40px; }}
  .badge {{ padding: 2px 8px; border-radius: 4px; color: white; font-size: 12px; }}
</style>
</head>
<body>
<h1>VLM Regression Report</h1>
<div class="meta">
  <b>Base model:</b> {self.base_model_id}<br>
  <b>Fine-tuned model:</b> {self.finetuned_model_id}<br>
  <b>Regression threshold:</b> {self.threshold:.1%}<br>
  <b>Flagged regressions:</b> {len(flagged)} / {len(deltas)} benchmarks
</div>

<div class="section">
<h2>All Benchmarks</h2>
<table>
  <tr>
    <th>Benchmark</th><th>Capability</th>
    <th>Base</th><th>Fine-tuned</th><th>Delta</th><th>Severity</th>
  </tr>
  {rows}
</table>
</div>

{"" if not flagged else f'''
<div class="section">
<h2>⚠ Flagged Regressions ({len(flagged)})</h2>
<table>
  <tr>
    <th>Benchmark</th><th>Capability</th><th>Delta</th><th>Severity</th><th>Context</th>
  </tr>
  {summary_rows}
</table>
</div>
'''}

</body>
</html>"""

        Path(path).write_text(html)
        print(f"HTML report saved to {path}")
        return path
