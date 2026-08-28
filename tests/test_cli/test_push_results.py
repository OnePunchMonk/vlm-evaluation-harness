"""Tests for the `push-results` CLI command (HF Hub upload, mocked)."""

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from vlm_evaluation_harness.cli import app

runner = CliRunner()


def test_push_results_requires_hf_token(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    results_file = tmp_path / "demomc_results.json"
    results_file.write_text("{}")

    result = runner.invoke(app, ["push-results", str(results_file), "--repo", "org/repo"])
    assert result.exit_code != 0
    assert "HF_TOKEN" in result.stdout


def test_push_results_requires_existing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    missing = tmp_path / "nope_results.json"

    result = runner.invoke(app, ["push-results", str(missing), "--repo", "org/repo"])
    assert result.exit_code != 0
    assert "No such file" in result.stdout


def test_push_results_calls_hf_api_upload_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    results_file = tmp_path / "demomc_results.json"
    results_file.write_text('{"model": "mock:v1"}')

    mock_api_instance = MagicMock()
    mock_api_cls = MagicMock(return_value=mock_api_instance)

    with patch("huggingface_hub.HfApi", mock_api_cls):
        result = runner.invoke(
            app, ["push-results", str(results_file), "--repo", "org/my-results"]
        )

    assert result.exit_code == 0, result.stdout
    mock_api_cls.assert_called_once_with(token="fake-token")
    mock_api_instance.upload_file.assert_called_once_with(
        path_or_fileobj=str(results_file),
        path_in_repo="demomc_results.json",
        repo_id="org/my-results",
        repo_type="dataset",
    )


def test_push_results_respects_custom_path_in_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    results_file = tmp_path / "demomc_results.json"
    results_file.write_text('{"model": "mock:v1"}')

    mock_api_instance = MagicMock()
    mock_api_cls = MagicMock(return_value=mock_api_instance)

    with patch("huggingface_hub.HfApi", mock_api_cls):
        result = runner.invoke(
            app,
            [
                "push-results",
                str(results_file),
                "--repo",
                "org/my-results",
                "--path-in-repo",
                "runs/v1.json",
            ],
        )

    assert result.exit_code == 0, result.stdout
    mock_api_instance.upload_file.assert_called_once_with(
        path_or_fileobj=str(results_file),
        path_in_repo="runs/v1.json",
        repo_id="org/my-results",
        repo_type="dataset",
    )
