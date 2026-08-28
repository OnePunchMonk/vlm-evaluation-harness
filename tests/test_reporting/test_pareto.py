"""Tests for the hand-rolled Pareto-frontier SVG builder."""

import xml.etree.ElementTree as ET

from vlm_evaluation_harness.reporting.html import build_pareto_svg

_RESULTS = [
    {"model": "mock:cheap-ok", "metrics": {"accuracy": 0.6},
     "cost": {"total_usd": 0.01}, "latency": {"p50_ms": 100}},
    {"model": "mock:expensive-great", "metrics": {"accuracy": 0.9},
     "cost": {"total_usd": 0.05}, "latency": {"p50_ms": 500}},
    {"model": "mock:expensive-bad", "metrics": {"accuracy": 0.4},
     "cost": {"total_usd": 0.05}, "latency": {"p50_ms": 500}},
]


def test_pareto_svg_is_valid_xml_with_one_point_per_model():
    svg = build_pareto_svg(_RESULTS, metric_name="accuracy", x_field="cost.total_usd")
    root = ET.fromstring(svg)
    ns = "{http://www.w3.org/2000/svg}"
    circles = root.findall(f"{ns}circle")
    assert len(circles) == 3


def test_dominated_point_is_not_on_frontier():
    svg = build_pareto_svg(_RESULTS, metric_name="accuracy", x_field="cost.total_usd")
    # expensive-bad is dominated by cheap-ok (cheaper AND better) -> grey, not orange.
    assert "mock:expensive-bad: accuracy=0.4000" in svg
    bad_idx = svg.index("mock:expensive-bad")
    circle_start = svg.rfind("<circle", 0, bad_idx)
    assert 'fill="#888"' in svg[circle_start:bad_idx]


def test_empty_results_renders_placeholder_svg():
    svg = build_pareto_svg([], metric_name="accuracy", x_field="cost.total_usd")
    ET.fromstring(svg)
    assert "No comparable points" in svg


def test_missing_field_is_skipped_not_crashed():
    results = [{"model": "no-cost", "metrics": {"accuracy": 0.5}, "cost": {}}]
    svg = build_pareto_svg(results, metric_name="accuracy", x_field="cost.total_usd")
    ET.fromstring(svg)
    assert "No comparable points" in svg
