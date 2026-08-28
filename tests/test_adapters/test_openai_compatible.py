"""Tests for the OpenAI-compatible self-hosted endpoint adapter."""

from __future__ import annotations

import pytest

pytest.importorskip("openai")

from vlm_evaluation_harness.adapters.openai_compatible import OpenAICompatibleAdapter  # noqa: E402


def test_requires_a_base_url(monkeypatch):
    monkeypatch.delenv("VLM_HARNESS_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="base_url"):
        OpenAICompatibleAdapter(model_id="my-model")


def test_base_url_kwarg_wins_over_env(monkeypatch):
    monkeypatch.setenv("VLM_HARNESS_BASE_URL", "http://env-host:9999/v1")
    adapter = OpenAICompatibleAdapter(model_id="my-model", base_url="http://kwarg-host:8000/v1")
    assert str(adapter._client.base_url) == "http://kwarg-host:8000/v1/"


def test_base_url_from_env_var(monkeypatch):
    monkeypatch.setenv("VLM_HARNESS_BASE_URL", "http://localhost:8000/v1")
    adapter = OpenAICompatibleAdapter(model_id="my-model")
    assert str(adapter._client.base_url) == "http://localhost:8000/v1/"


def test_self_hosted_models_have_no_per_token_cost(monkeypatch):
    monkeypatch.setenv("VLM_HARNESS_BASE_URL", "http://localhost:8000/v1")
    adapter = OpenAICompatibleAdapter(model_id="my-model")
    assert adapter.cost_per_million_input_tokens is None
    assert adapter.cost_per_million_output_tokens is None


def test_generate_sends_request_to_custom_base_url(monkeypatch):
    monkeypatch.setenv("VLM_HARNESS_BASE_URL", "http://localhost:8000/v1")
    adapter = OpenAICompatibleAdapter(model_id="my-model")

    captured = {}

    class FakeUsage:
        prompt_tokens = 12
        completion_tokens = 3

    class FakeChoice:
        class message:
            content = "B"

    class FakeResponse:
        choices = [FakeChoice()]
        usage = FakeUsage()
        model = "my-model"

    def fake_create(**kwargs):
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(adapter._client.chat.completions, "create", fake_create)

    response = adapter.generate(images=[], prompt="What color?", max_tokens=16)

    assert captured["model"] == "my-model"
    assert captured["messages"][-1]["role"] == "user"
    assert response.text == "B"
    assert response.input_tokens == 12
    assert response.output_tokens == 3
    assert response.model_id == "my-model"
