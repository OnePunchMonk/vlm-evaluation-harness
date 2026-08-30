"""OpenAICompatibleAdapter: base_url resolution, and parity with OpenAIAdapter
via their shared ChatCompletionsAdapter base (see issue #22)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

pytest.importorskip("openai")

import openai  # noqa: E402

from vlm_harness.adapters.openai import OpenAIAdapter  # noqa: E402
from vlm_harness.adapters.openai_compatible import (  # noqa: E402
    _BASE_URL_ENV_VAR,
    OpenAICompatibleAdapter,
)


class _FakeChatCompletions:
    def __init__(self, response):
        self._response = response
        self.last_call = None

    def create(self, **kwargs):
        self.last_call = kwargs
        return self._response


class _FakeOpenAIClient:
    def __init__(self, api_key=None, base_url=None):
        self.api_key = api_key
        self.base_url = base_url
        self.chat = SimpleNamespace(completions=_FakeChatCompletions(_canned_response()))


def _canned_response():
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="a cat"))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3),
        model="served-model",
    )


@pytest.fixture(autouse=True)
def fake_openai_client(monkeypatch):
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAIClient)


def test_base_url_required_without_env_var(monkeypatch):
    monkeypatch.delenv(_BASE_URL_ENV_VAR, raising=False)
    with pytest.raises(ValueError, match="base_url"):
        OpenAICompatibleAdapter(model_id="llava")


def test_base_url_from_kwarg():
    adapter = OpenAICompatibleAdapter(model_id="llava", base_url="http://localhost:8000/v1")
    assert adapter._client.base_url == "http://localhost:8000/v1"


def test_base_url_from_env_var(monkeypatch):
    monkeypatch.setenv(_BASE_URL_ENV_VAR, "http://localhost:11434/v1")
    adapter = OpenAICompatibleAdapter(model_id="llava")
    assert adapter._client.base_url == "http://localhost:11434/v1"


def test_no_api_key_needed_for_local_server():
    adapter = OpenAICompatibleAdapter(model_id="llava", base_url="http://localhost:8000/v1")
    assert adapter._client.api_key  # SDK requires a non-empty string; we default one


def test_local_server_reports_no_cost():
    adapter = OpenAICompatibleAdapter(model_id="llava", base_url="http://localhost:8000/v1")
    assert adapter.cost_per_million_input_tokens is None
    assert adapter.cost_per_million_output_tokens is None


def test_generate_shares_request_shape_with_openai_adapter():
    """Both adapters build the same messages payload via the shared base --
    this is the point of factoring ChatCompletionsAdapter out."""
    img = Image.new("RGB", (4, 4), color="red")

    compat = OpenAICompatibleAdapter(model_id="llava", base_url="http://localhost:8000/v1")
    compat.generate(images=[img], prompt="what is this?", system="be terse", max_tokens=16)
    compat_call = compat._client.chat.completions.last_call

    openai_adapter = OpenAIAdapter(model_id="gpt-4o", api_key="sk-test")
    openai_adapter.generate(images=[img], prompt="what is this?", system="be terse", max_tokens=16)
    openai_call = openai_adapter._client.chat.completions.last_call

    assert compat_call["messages"] == openai_call["messages"]
    assert compat_call["max_tokens"] == openai_call["max_tokens"] == 16


def test_generate_returns_parsed_response():
    adapter = OpenAICompatibleAdapter(model_id="llava", base_url="http://localhost:8000/v1")
    img = Image.new("RGB", (4, 4), color="blue")

    response = adapter.generate(images=[img], prompt="describe")

    assert response.text == "a cat"
    assert response.input_tokens == 10
    assert response.output_tokens == 3
    assert response.model_id == "served-model"
