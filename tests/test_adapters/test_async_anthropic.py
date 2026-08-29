"""Tests for AsyncAnthropicAdapter.

Groundwork for a future async runner (issue #23) -- NOT wired into
engine/runner.py's synchronous eval loop. Skips cleanly if the optional
`anthropic` package isn't installed, same pattern as the openai-extra tests.
"""

from __future__ import annotations

import pytest

pytest.importorskip("anthropic")

from vlm_evaluation_harness.adapters.anthropic import AsyncAnthropicAdapter  # noqa: E402


class _FakeUsage:
    input_tokens = 20
    output_tokens = 5


class _FakeTextBlock:
    type = "text"
    text = "B"


class _FakeResponse:
    content = [_FakeTextBlock()]
    usage = _FakeUsage()
    model = "claude-opus-4-6"


async def test_agenerate_sends_correct_request_and_parses_response(monkeypatch):
    adapter = AsyncAnthropicAdapter(model_id="claude-opus-4-6", api_key="sk-fake")

    captured = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return _FakeResponse()

    monkeypatch.setattr(adapter._client.messages, "create", fake_create)

    response = await adapter.agenerate(
        images=[], prompt="What color?", system="Be terse.", max_tokens=16
    )

    assert captured["model"] == "claude-opus-4-6"
    assert captured["system"] == "Be terse."
    assert captured["messages"][-1]["role"] == "user"
    assert response.text == "B"
    assert response.input_tokens == 20
    assert response.output_tokens == 5
    assert response.model_id == "claude-opus-4-6"
