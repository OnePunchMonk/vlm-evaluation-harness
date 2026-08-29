"""Tests for AsyncOpenAIAdapter/AsyncOpenAICompatibleAdapter.

Groundwork for a future async runner (issue #23) -- these are NOT wired
into engine/runner.py's synchronous eval loop. `pytest-asyncio` is
configured with `asyncio_mode = auto` (pyproject.toml), so plain `async def`
tests run without a decorator.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openai")

from vlm_evaluation_harness.adapters.openai import AsyncOpenAIAdapter  # noqa: E402
from vlm_evaluation_harness.adapters.openai_compatible import (  # noqa: E402
    AsyncOpenAICompatibleAdapter,
)


class _FakeUsage:
    prompt_tokens = 12
    completion_tokens = 3


class _FakeChoice:
    class message:
        content = "B"


class _FakeResponse:
    choices = [_FakeChoice()]
    usage = _FakeUsage()
    model = "my-model"


async def test_agenerate_sends_correct_request_and_parses_response(monkeypatch):
    adapter = AsyncOpenAIAdapter(model_id="my-model", api_key="sk-fake")

    captured = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return _FakeResponse()

    monkeypatch.setattr(adapter._client.chat.completions, "create", fake_create)

    response = await adapter.agenerate(
        images=[], prompt="What color?", system="Be terse.", max_tokens=16
    )

    assert captured["model"] == "my-model"
    assert captured["messages"][0] == {"role": "system", "content": "Be terse."}
    assert captured["messages"][-1]["role"] == "user"
    assert response.text == "B"
    assert response.input_tokens == 12
    assert response.output_tokens == 3
    assert response.model_id == "my-model"


async def test_async_openai_compatible_uses_custom_base_url(monkeypatch):
    monkeypatch.setenv("VLM_HARNESS_BASE_URL", "http://localhost:8000/v1")
    adapter = AsyncOpenAICompatibleAdapter(model_id="my-model")
    assert str(adapter._client.base_url) == "http://localhost:8000/v1/"
    assert adapter.cost_per_million_input_tokens is None

    async def fake_create(**kwargs):
        return _FakeResponse()

    monkeypatch.setattr(adapter._client.chat.completions, "create", fake_create)
    response = await adapter.agenerate(images=[], prompt="hi")
    assert response.text == "B"


def test_async_openai_compatible_requires_base_url(monkeypatch):
    monkeypatch.delenv("VLM_HARNESS_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="base_url"):
        AsyncOpenAICompatibleAdapter(model_id="my-model")
