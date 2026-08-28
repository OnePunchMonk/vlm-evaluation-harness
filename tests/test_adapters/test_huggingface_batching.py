"""Tests for HuggingFaceAdapter's batching/OOM-backoff logic.

`transformers`/`torch` aren't installed in this environment (and shouldn't
be required just to test control flow), so these tests construct the
adapter via `object.__new__` and monkeypatch `_generate_forward` — the one
method that actually needs torch — with a fake, to exercise the batching,
chunking, and OOM-backoff logic in isolation.
"""

from __future__ import annotations

import pytest

from vlm_evaluation_harness.adapters.base import VLMResponse
from vlm_evaluation_harness.adapters.huggingface import (
    BatchGenerateRequest,
    HuggingFaceAdapter,
    _is_oom_error,
)


def _make_adapter(batch_size: int = 4) -> HuggingFaceAdapter:
    adapter = object.__new__(HuggingFaceAdapter)
    adapter._model_id = "fake-model"
    adapter._batch_size = batch_size
    return adapter


def _fake_response(req: BatchGenerateRequest) -> VLMResponse:
    return VLMResponse(text=f"echo:{req.prompt}", model_id="fake-model")


def test_is_oom_error_matches_runtime_error_message():
    assert _is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate 2 MiB"))
    assert not _is_oom_error(RuntimeError("some other failure"))
    assert not _is_oom_error(ValueError("not even a runtime error"))


def test_generate_batch_empty_returns_empty():
    adapter = _make_adapter()
    assert adapter.generate_batch([]) == []


def test_generate_batch_preserves_order_and_chunks_by_batch_size(monkeypatch):
    adapter = _make_adapter(batch_size=2)
    seen_chunk_sizes = []

    def fake_forward(chunk):
        seen_chunk_sizes.append(len(chunk))
        return [_fake_response(req) for req in chunk]

    monkeypatch.setattr(adapter, "_generate_forward", fake_forward)

    requests = [BatchGenerateRequest(images=[], prompt=f"q{i}") for i in range(5)]
    responses = adapter.generate_batch(requests)

    assert [r.text for r in responses] == [f"echo:q{i}" for i in range(5)]
    assert seen_chunk_sizes == [2, 2, 1]


def test_generate_batch_backs_off_on_oom_then_succeeds(monkeypatch):
    adapter = _make_adapter(batch_size=4)
    attempts = []

    def flaky_forward(chunk):
        attempts.append(len(chunk))
        if len(chunk) > 1:
            raise RuntimeError("CUDA out of memory.")
        return [_fake_response(req) for req in chunk]

    monkeypatch.setattr(adapter, "_generate_forward", flaky_forward)

    requests = [BatchGenerateRequest(images=[], prompt=f"q{i}") for i in range(3)]
    responses = adapter.generate_batch(requests)

    assert [r.text for r in responses] == ["echo:q0", "echo:q1", "echo:q2"]
    # First attempt at size 3 OOMs, backs off, and eventually every request
    # succeeds individually.
    assert 3 in attempts
    assert attempts.count(1) == 3


def test_generate_batch_reraises_non_oom_errors(monkeypatch):
    adapter = _make_adapter(batch_size=4)

    def always_fails(chunk):
        raise ValueError("not an OOM, don't retry this")

    monkeypatch.setattr(adapter, "_generate_forward", always_fails)

    requests = [BatchGenerateRequest(images=[], prompt="q")]
    with pytest.raises(ValueError, match="not an OOM"):
        adapter.generate_batch(requests)


def test_supports_batch_inference_flag():
    adapter = _make_adapter()
    assert adapter.supports_batch_inference is True
