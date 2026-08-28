"""Tests for the publishable static leaderboard site generator."""

from vlm_evaluation_harness.reporting.html import (
    build_static_leaderboard_html,
    save_static_leaderboard_html,
)

_RESULTS = [
    {"model": "mock:v1", "benchmark": "DemoMC", "metrics": {"accuracy": 0.5},
     "cost": {"total_usd": 0.0}, "n_samples": 12},
    {"model": "mock:v2", "benchmark": "DemoMC", "metrics": {"accuracy": 0.7},
     "cost": {"total_usd": 0.0}, "n_samples": 12},
    {"model": "mock:v1", "benchmark": "POPE", "metrics": {"pope": 0.6},
     "cost": {"total_usd": 0.0}, "n_samples": 8},
]


def test_one_section_per_benchmark():
    html = build_static_leaderboard_html(_RESULTS)
    assert "<h2>DemoMC</h2>" in html
    assert "<h2>POPE</h2>" in html
    assert html.index("<h2>DemoMC</h2>") < html.index("<h2>POPE</h2>")  # alphabetical


def test_tables_are_marked_sortable_and_have_sort_script():
    html = build_static_leaderboard_html(_RESULTS)
    assert 'class="sortable"' in html
    assert "sortTable" in html
    assert "<script>" in html


def test_single_self_contained_file_no_external_assets():
    html = build_static_leaderboard_html(_RESULTS)
    assert "<link" not in html
    assert "src=\"http" not in html


def test_save_static_leaderboard_html_writes_file(tmp_path):
    path = save_static_leaderboard_html(_RESULTS, path=tmp_path / "lb.html")
    assert path.exists()
    content = path.read_text()
    assert "mock:v1" in content
    assert "mock:v2" in content
