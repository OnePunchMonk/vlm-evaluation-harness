"""Tests for the vLLM/SGLang OpenAI-compatible server adapter."""

from __future__ import annotations

from vlm_evaluation_harness.adapters.vllm import DEFAULT_BASE_URL, VLLMAdapter


def test_defaults_to_local_vllm_server(monkeypatch):
    monkeypatch.delenv("VLM_HARNESS_BASE_URL", raising=False)
    adapter = VLLMAdapter(model_id="Qwen/Qwen2-VL-7B-Instruct")
    assert str(adapter._client.base_url) == DEFAULT_BASE_URL + "/"


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("VLM_HARNESS_BASE_URL", "http://gpu-box:8001/v1")
    adapter = VLLMAdapter(model_id="Qwen/Qwen2-VL-7B-Instruct")
    assert str(adapter._client.base_url) == "http://gpu-box:8001/v1/"


def test_explicit_base_url_wins(monkeypatch):
    monkeypatch.setenv("VLM_HARNESS_BASE_URL", "http://gpu-box:8001/v1")
    adapter = VLLMAdapter(model_id="m", base_url="http://other:8002/v1")
    assert str(adapter._client.base_url) == "http://other:8002/v1/"


def test_registered_in_registry():
    from vlm_evaluation_harness.adapters.registry import list_adapters

    adapters = list_adapters()
    assert "vllm" in adapters
    assert "openai_compatible" in adapters
