"""Adapter registry: resolve 'provider:model_id' strings to adapter instances."""

from __future__ import annotations

from vlm_harness.adapters.base import VLMAdapter

# Only providers with an implementation in this package are listed. The
# registry previously advertised google/vllm/ollama/litellm, and pyproject
# shipped matching extras, for adapter modules that did not exist.
_PROVIDERS: dict[str, str] = {
    "mock": "vlm_harness.adapters.mock.MockAdapter",
    "anthropic": "vlm_harness.adapters.anthropic.AnthropicAdapter",
    "openai": "vlm_harness.adapters.openai.OpenAIAdapter",
    "huggingface": "vlm_harness.adapters.huggingface.HuggingFaceAdapter",
    "hf": "vlm_harness.adapters.huggingface.HuggingFaceAdapter",
}

# Extras name to install for each provider, where it differs from the provider key.
_EXTRAS: dict[str, str] = {"hf": "huggingface", "mock": ""}


def get_adapter(model_spec: str, **kwargs) -> VLMAdapter:
    """
    Parse 'provider:model_id' and return an instantiated adapter.

    Examples:
        get_adapter("anthropic:claude-opus-4-6")
        get_adapter("openai:gpt-4o")
        get_adapter("huggingface:llava-hf/llava-1.5-7b-hf", device="cuda:0")
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
        raise ValueError(f"Unknown provider '{provider}'. Available providers: {available}")

    module_path, class_name = _PROVIDERS[provider].rsplit(".", 1)

    try:
        import importlib

        module = importlib.import_module(module_path)
        adapter_cls = getattr(module, class_name)
    except ImportError as e:
        extra = _EXTRAS.get(provider, provider)
        hint = f"pip install vlm-harness[{extra}]" if extra else "check your installation"
        raise ImportError(
            f"Could not import adapter for provider '{provider}'. {hint}\n"
            f"Original error: {e}"
        ) from e

    return adapter_cls(model_id=model_id, **kwargs)


def list_adapters() -> dict[str, str]:
    """Return all registered provider names and their adapter class paths."""
    return dict(_PROVIDERS)
