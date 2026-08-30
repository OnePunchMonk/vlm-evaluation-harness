"""Adapter for any server exposing an OpenAI-compatible /chat/completions
endpoint: Ollama, TGI, LM Studio, vLLM's OpenAI-compat mode, etc.

Single adapter instead of one per server, since they all speak the same
wire format and differ only in base_url (and usually don't need a real
API key). See issue #22.
"""

from __future__ import annotations

import os

from vlm_harness.adapters._chat_completions import ChatCompletionsAdapter

_BASE_URL_ENV_VAR = "VLM_HARNESS_OPENAI_COMPATIBLE_BASE_URL"


class OpenAICompatibleAdapter(ChatCompletionsAdapter):
    """Adapter for a self-hosted OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        model_id: str,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        try:
            import openai as _openai
        except ImportError:
            raise ImportError("pip install vlm-harness[openai]")

        resolved_base_url = base_url or os.environ.get(_BASE_URL_ENV_VAR)
        if not resolved_base_url:
            raise ValueError(
                "OpenAICompatibleAdapter needs a base_url -- pass one explicitly "
                f"(CLI: --base-url) or set the {_BASE_URL_ENV_VAR} environment variable."
            )

        # Most self-hosted servers don't check the key, but the SDK requires
        # a non-empty string.
        self._client = _openai.OpenAI(base_url=resolved_base_url, api_key=api_key or "not-needed")
        self._model_id = model_id
        self._base_url = resolved_base_url

    @property
    def cost_per_million_input_tokens(self) -> float | None:
        return None

    @property
    def cost_per_million_output_tokens(self) -> float | None:
        return None
