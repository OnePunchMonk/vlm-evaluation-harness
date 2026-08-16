"""Smoke tests for CLI commands that work fully offline (mock adapters)."""

from typer.testing import CliRunner

from vlm_harness.cli import app

runner = CliRunner()


def test_list_benchmarks():
    result = runner.invoke(app, ["list-benchmarks"])
    assert result.exit_code == 0
    assert "DemoMC" in result.stdout


def test_eval_with_mock_adapter(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))  # isolate ~/.vlm-harness/history.jsonl
    result = runner.invoke(
        app,
        ["eval", "--model", "mock:demo-v1", "--bench", "demo_mc", "--split", "validation"],
    )
    assert result.exit_code == 0
    assert "accuracy" in result.stdout


def test_gen_eval_with_mock_adapter(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    result = runner.invoke(
        app,
        ["gen-eval", "--model", "mock:t2i-v1", "--bench", "genjudge_mini", "--max-samples", "2"],
    )
    assert result.exit_code == 0
    assert "llm_judge" in result.stdout


def test_history_reports_tracked_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    runner.invoke(app, ["eval", "--model", "mock:demo-v1", "--bench", "demo_mc"])
    result = runner.invoke(app, ["history"])
    assert result.exit_code == 0
    assert "mock:demo-v1" in result.stdout


def test_regression_without_tracked_runs_exits_nonzero(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    result = runner.invoke(app, ["regression", "--baseline", "mock:a", "--current", "mock:b"])
    assert result.exit_code == 1


def test_validate_bench():
    result = runner.invoke(app, ["validate-bench", "--bench", "demo_mc"])
    assert result.exit_code == 0
    assert "valid" in result.stdout
