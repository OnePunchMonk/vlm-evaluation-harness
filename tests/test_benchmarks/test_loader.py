"""Tests for BenchmarkLoader's HuggingFace source path: revision pinning."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

from vlm_evaluation_harness.benchmarks.loader import BenchmarkLoader
from vlm_evaluation_harness.benchmarks.schema import (
    BenchmarkManifest,
    FieldsConfig,
    SourceConfig,
    SplitConfig,
)


def _hf_manifest(revision: str, revision_note: str | None = None) -> BenchmarkManifest:
    return BenchmarkManifest(
        name="TestHFBench",
        source=SourceConfig(
            type="huggingface", path="org/dataset", revision=revision, revision_note=revision_note
        ),
        splits=[SplitConfig(name="validation")],
        fields=FieldsConfig(question="question", answer="answer"),
    )


def _fake_dataset(rows: list[dict]):
    ds = MagicMock()
    ds.__iter__.return_value = iter(rows)
    ds.__len__.return_value = len(rows)
    ds.shuffle.return_value = ds
    ds.select.return_value = ds
    return ds


def test_revision_is_passed_to_load_dataset():
    manifest = _hf_manifest(revision="abc123def")
    rows = [{"question": "q", "answer": "a"}]
    fake_load_dataset = MagicMock(return_value=_fake_dataset(rows))

    with patch.dict("sys.modules", {"datasets": MagicMock(load_dataset=fake_load_dataset)}):
        loader = BenchmarkLoader()
        list(loader.load(manifest, split="validation"))

    assert fake_load_dataset.call_args.kwargs["revision"] == "abc123def"


def test_warns_when_revision_is_unpinned_main(caplog):
    manifest = _hf_manifest(revision="main")
    rows = [{"question": "q", "answer": "a"}]
    fake_load_dataset = MagicMock(return_value=_fake_dataset(rows))

    with patch.dict("sys.modules", {"datasets": MagicMock(load_dataset=fake_load_dataset)}):
        loader = BenchmarkLoader()
        with caplog.at_level(logging.WARNING, logger="vlm_evaluation_harness.benchmarks.loader"):
            list(loader.load(manifest, split="validation"))

    assert any("not a reproducible pin" in r.message for r in caplog.records)


def test_no_warning_when_revision_is_pinned(caplog):
    manifest = _hf_manifest(revision="deadbeef")
    rows = [{"question": "q", "answer": "a"}]
    fake_load_dataset = MagicMock(return_value=_fake_dataset(rows))

    with patch.dict("sys.modules", {"datasets": MagicMock(load_dataset=fake_load_dataset)}):
        loader = BenchmarkLoader()
        with caplog.at_level(logging.WARNING, logger="vlm_evaluation_harness.benchmarks.loader"):
            list(loader.load(manifest, split="validation"))

    assert not any("not a reproducible pin" in r.message for r in caplog.records)


def test_no_warning_when_main_has_a_documented_revision_note(caplog):
    manifest = _hf_manifest(revision="main", revision_note="gated dataset, pin after auth")
    rows = [{"question": "q", "answer": "a"}]
    fake_load_dataset = MagicMock(return_value=_fake_dataset(rows))

    with patch.dict("sys.modules", {"datasets": MagicMock(load_dataset=fake_load_dataset)}):
        loader = BenchmarkLoader()
        with caplog.at_level(logging.WARNING, logger="vlm_evaluation_harness.benchmarks.loader"):
            list(loader.load(manifest, split="validation"))

    assert not any("not a reproducible pin" in r.message for r in caplog.records)
