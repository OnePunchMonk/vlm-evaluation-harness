"""Smoke tests for CLI commands that work fully offline (mock adapters)."""

from pathlib import Path

from typer.testing import CliRunner

from vlm_evaluation_harness.cli import app

runner = CliRunner()


def test_list_benchmarks():
    result = runner.invoke(app, ["list-benchmarks"])
    assert result.exit_code == 0
    assert "DemoMC" in result.stdout


def test_list_benchmarks_filtered_by_tag():
    result = runner.invoke(app, ["list-benchmarks", "--tags", "safety"])
    assert result.exit_code == 0
    assert "POPE" in result.stdout
    assert "DemoMC" not in result.stdout


def test_list_benchmarks_verbose_shows_tags_column():
    result = runner.invoke(app, ["list-benchmarks", "--tags", "safety", "--verbose"])
    assert result.exit_code == 0
    assert "safety" in result.stdout


def test_eval_with_mock_adapter(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))  # isolate ~/.vlm-evaluation-harness/history.jsonl
    result = runner.invoke(
        app,
        ["eval", "--model", "mock:demo-v1", "--bench", "demo_mc", "--split", "validation"],
    )
    assert result.exit_code == 0
    assert "accuracy" in result.stdout


def test_eval_records_seed_and_harness_provenance(tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("HOME", str(tmp_path))
    out_dir = tmp_path / "results"
    result = runner.invoke(
        app,
        [
            "eval",
            "--model",
            "mock:demo-v1",
            "--bench",
            "demo_mc",
            "--seed",
            "7",
            "--output-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0
    results_file = next(out_dir.glob("*.json"))
    data = json.loads(results_file.read_text())
    provenance = data["provenance"]
    assert provenance["decoding"]["seed"] == 7
    assert provenance["results_schema_version"] == "1.0"
    assert provenance["harness_version"]
    # harness_sha is best-effort (None outside a git checkout) but the key
    # must always be present so downstream tooling can rely on it.
    assert "harness_sha" in provenance


def test_eval_predict_only_skips_scoring(tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("HOME", str(tmp_path))
    out_dir = tmp_path / "results"
    result = runner.invoke(
        app,
        [
            "eval",
            "--model",
            "mock:demo-v1",
            "--bench",
            "demo_mc",
            "--predict-only",
            "--output-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0
    data = json.loads(next(out_dir.glob("*.json")).read_text())
    assert data["metrics"] == {}
    assert data["samples"][0]["prediction"]


def test_eval_no_log_samples_omits_samples_array(tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("HOME", str(tmp_path))
    out_dir = tmp_path / "results"
    result = runner.invoke(
        app,
        [
            "eval",
            "--model",
            "mock:demo-v1",
            "--bench",
            "demo_mc",
            "--no-log-samples",
            "--output-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0
    data = json.loads(next(out_dir.glob("*.json")).read_text())
    assert "samples" not in data


def test_eval_system_override_recorded_in_provenance(tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("HOME", str(tmp_path))
    out_dir = tmp_path / "results"
    result = runner.invoke(
        app,
        [
            "eval",
            "--model",
            "mock:demo-v1",
            "--bench",
            "demo_mc",
            "--system",
            "You are terse.",
            "--output-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0
    data = json.loads(next(out_dir.glob("*.json")).read_text())
    assert data["provenance"]["prompt"]["system"] == "You are terse."
    assert data["provenance"]["prompt"]["system_override"] is True


def test_include_path_makes_custom_manifest_evaluable(tmp_path, monkeypatch):
    """A benchmark manifest outside the built-in manifests/ dir is picked up
    via --include-path and is fully evaluable end to end."""
    import textwrap

    from vlm_evaluation_harness import benchmarks as _bench_pkg

    monkeypatch.setenv("HOME", str(tmp_path))
    fixture_dir = (
        Path(_bench_pkg.__file__).resolve().parent / "fixtures" / "demo_mc"
    )
    manifest_dir = tmp_path / "custom_manifests"
    manifest_dir.mkdir()
    (manifest_dir / "custom_mc.yaml").write_text(
        textwrap.dedent(
            f"""\
            name: CustomMC
            schema_version: "1.0"
            taxonomy_category: perception
            modality: 2d
            source:
              type: local
              path: {fixture_dir}
            splits:
              - name: validation
                scorable: true
            task_type: multiple_choice
            fields:
              question: question
              choices: choices
              answer: answer
              images: [image]
            prompt_template: |
              {{question}}

              Options:
              {{formatted_choices}}

              Answer with the option letter only.
            answer_extraction:
              strategy: first_letter
              normalize: uppercase
            metrics:
              - type: accuracy
            """
        )
    )

    list_result = runner.invoke(app, ["list-benchmarks", "--include-path", str(manifest_dir)])
    assert list_result.exit_code == 0
    assert "CustomMC" in list_result.stdout

    eval_result = runner.invoke(
        app,
        [
            "eval",
            "--model",
            "mock:demo-v1",
            "--bench",
            "custom_mc",
            "--include-path",
            str(manifest_dir),
        ],
    )
    assert eval_result.exit_code == 0
    assert "accuracy" in eval_result.stdout


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
