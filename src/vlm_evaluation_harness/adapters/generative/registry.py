"""Adapter registry for text-to-image backends: resolve 'provider:model_id'."""

from __future__ import annotations

from vlm_evaluation_harness.adapters.generative.base import T2IAdapter

_PROVIDERS: dict[str, str] = {
    "mock": "vlm_evaluation_harness.adapters.generative.mock.MockT2IAdapter",
    "openai": "vlm_evaluation_harness.adapters.generative.openai_images.OpenAIImageAdapter",
    "diffusers": "vlm_evaluation_harness.adapters.generative.diffusers_local.DiffusersAdapter",
    "hf": "vlm_evaluation_harness.adapters.generative.diffusers_local.DiffusersAdapter",
}


def get_t2i_adapter(model_spec: str, **kwargs) -> T2IAdapter:
    """
    Parse 'provider:model_id' and return an instantiated T2I adapter.

    Examples:
        get_t2i_adapter("mock:demo-v1")
        get_t2i_adapter("openai:gpt-image-1")
        get_t2i_adapter("diffusers:stabilityai/stable-diffusion-2-1")
    """
    if ":" not in model_spec:
        raise ValueError(
            f"Invalid model spec '{model_spec}'. "
            "Expected format: 'provider:model_id' (e.g. 'openai:gpt-image-1')"
        )

    provider, model_id = model_spec.split(":", 1)
    provider = provider.lower()

    if provider not in _PROVIDERS:
        available = ", ".join(sorted(_PROVIDERS))
        raise ValueError(f"Unknown T2I provider '{provider}'. Available providers: {available}")

    module_path, class_name = _PROVIDERS[provider].rsplit(".", 1)

    try:
        import importlib
        module = importlib.import_module(module_path)
        adapter_cls = getattr(module, class_name)
    except ImportError as e:
        raise ImportError(
            f"Could not import T2I adapter for provider '{provider}'. "
            f"Install the required extras: pip install vlm-evaluation-harness[generative]\n"
            f"Original error: {e}"
        ) from e

    return adapter_cls(model_id=model_id, **kwargs)


def list_t2i_adapters() -> dict[str, str]:
    return dict(_PROVIDERS)
