"""Tests for the self-contained HTML report generator."""

from vlm_evaluation_harness.reporting.html import build_leaderboard_html, save_html_report
from vlm_evaluation_harness.tracking.history import HistoryEntry
from vlm_evaluation_harness.tracking.regression import compare_entries

_RESULTS = [
    {"model": "mock:v1", "benchmark": "DemoMC", "metrics": {"accuracy": 0.5},
     "cost": {"total_usd": 0.0}, "n_samples": 12},
    {"model": "mock:v2", "benchmark": "DemoMC", "metrics": {"accuracy": 0.7},
     "cost": {"total_usd": 0.0}, "n_samples": 12},
]


def test_leaderboard_html_contains_models_and_scores():
    html = build_leaderboard_html(_RESULTS)
    assert "mock:v1" in html
    assert "mock:v2" in html
    assert "0.5000" in html
    assert "0.7000" in html


def test_leaderboard_html_shows_confidence_interval_when_present():
    results = [
        {"model": "mock:v1", "benchmark": "DemoMC", "metrics": {"accuracy": 0.5},
         "metric_ci95": {"accuracy": [0.3, 0.7]}, "cost": {"total_usd": 0.0}, "n_samples": 12},
    ]
    html = build_leaderboard_html(results)
    assert "[0.3000, 0.7000]" in html


def test_leaderboard_html_omits_ci_when_nan_or_absent():
    results = [
        {"model": "mock:v1", "benchmark": "DemoMC", "metrics": {"accuracy": 0.5},
         "metric_ci95": {"accuracy": [float("nan"), float("nan")]},
         "cost": {"total_usd": 0.0}, "n_samples": 12},
    ]
    html = build_leaderboard_html(results)
    assert "nan" not in html
    assert "0.5000" in html


def test_save_html_report_writes_file(tmp_path):
    path = save_html_report(_RESULTS, path=tmp_path / "report.html", title="Test Report")
    assert path.exists()
    content = path.read_text()
    assert "Test Report" in content
    assert "<table>" in content


def test_save_html_report_includes_regression_section(tmp_path):
    base = HistoryEntry(
        "r1", "t", "mock:v1", "DemoMC", "s", "discriminative", {"accuracy": 0.9}, 12
    )
    cur = HistoryEntry(
        "r2", "t", "mock:v2", "DemoMC", "s", "discriminative", {"accuracy": 0.5}, 12
    )
    deltas = compare_entries(base, cur)

    path = save_html_report(_RESULTS, deltas=deltas, path=tmp_path / "report.html")
    content = path.read_text()
    assert "Regression Report" in content
    assert "CRITICAL" in content
