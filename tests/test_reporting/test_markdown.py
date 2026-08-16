"""Tests for the Markdown report generator."""

from vlm_harness.reporting.markdown import (
    build_leaderboard_markdown,
    build_regression_markdown,
    build_report_markdown,
)
from vlm_harness.tracking.history import HistoryEntry
from vlm_harness.tracking.regression import compare_entries

_RESULTS = [
    {"model": "mock:v1", "benchmark": "DemoMC", "metrics": {"accuracy": 0.5},
     "cost": {"total_usd": 0.0}, "n_samples": 12},
]


def test_leaderboard_markdown_is_a_table():
    md = build_leaderboard_markdown(_RESULTS)
    lines = md.splitlines()
    assert lines[0].startswith("| Model")
    assert "mock:v1" in md


def test_regression_markdown_empty_deltas():
    assert build_regression_markdown([]) == "No comparable runs found."


def test_regression_markdown_reports_flagged_count():
    base = HistoryEntry("r1", "t", "m1", "B", "s", "discriminative", {"accuracy": 0.9}, 12)
    cur = HistoryEntry("r2", "t", "m2", "B", "s", "discriminative", {"accuracy": 0.5}, 12)
    deltas = compare_entries(base, cur)
    md = build_regression_markdown(deltas)
    assert "1/1" in md
    assert "CRITICAL" in md


def test_build_report_markdown_combines_sections():
    md = build_report_markdown(_RESULTS)
    assert "## Leaderboard" in md
    assert "## Regression Report" not in md
