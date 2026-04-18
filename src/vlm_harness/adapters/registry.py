"""Adapter registry: resolve 'provider:model_id' strings to adapter instances."""

from __future__ import annotations

from vlm_harness.adapters.base import VLMAdapter


_PROVIDERS: dict[str, str] = {
    "anthropic": "vlm_harness.adapters.anthropic.AnthropicAdapter",
    "openai": "vlm_harness.adapters.openai.OpenAIAdapter",
    "google": "vlm_harness.adapters.google.GoogleAdapter",
    "huggingface": "vlm_harness.adapters.huggingface.HuggingFaceAdapter",
    "hf": "vlm_harness.adapters.huggingface.HuggingFaceAdapter",
    "vllm": "vlm_harness.adapters.vllm.VLLMAdapter",
    "ollama": "vlm_harness.adapters.ollama.OllamaAdapter",
    "litellm": "vlm_harness.adapters.litellm.LiteLLMAdapter",
}


def get_adapter(model_spec: str, **kwargs) -> VLMAdapter:
    """
    Parse 'provider:model_id' and return an instantiated adapter.

    Examples:
        get_adapter("anthropic:claude-opus-4-6")
        get_adapter("openai:gpt-4o")
        get_adapter("huggingface:liuhaotian/llava-v1.6-34b", device="cuda:0")
    """
    if ":" not in model_spec:
        raise ValueError(
            f"Invalid model spec '{model_spec}'. "
            "Expected format: 'provider:model_id' (e.g. 'anthropic:claude-opus-4-6')"
        )

    provider, model_id = model_spec.split(":", 1)
    provider = provider.lower()

    if provider not in _PROVIDERS:
        available = ", ".join(sorted(_PROVIDERS))
        raise ValueError(
            f"Unknown provider '{provider}'. Available providers: {available}"
        )

    module_path, class_name = _PROVIDERS[provider].rsplit(".", 1)

    try:
        import importlib
        module = importlib.import_module(module_path)
        adapter_cls = getattr(module, class_name)
    except ImportError as e:
        raise ImportError(
            f"Could not import adapter for provider '{provider}'. "
            f"Install the required extras: pip install vlm-harness[{provider}]\n"
            f"Original error: {e}"
        ) from e

    return adapter_cls(model_id=model_id, **kwargs)


def list_adapters() -> dict[str, str]:
    """Return all registered provider names and their adapter class paths."""
    return dict(_PROVIDERS)
