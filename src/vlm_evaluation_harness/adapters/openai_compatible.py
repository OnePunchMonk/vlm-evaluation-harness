"""Adapter for self-hosted OpenAI-compatible chat completions endpoints.

Many self-hosted VLM servers (vLLM's `vllm serve`, TGI, LocalAI, ...) expose
the same `/v1/chat/completions` wire format as api.openai.com. This adapter
reuses `OpenAIAdapter` verbatim (the OpenAI Python SDK already supports
`base_url=`) and only overrides where the base URL comes from and the cost
model, since self-hosted models have no per-token API pricing.
"""

from __future__ import annotations

import os

from vlm_evaluation_harness.adapters.openai import OpenAIAdapter


class OpenAICompatibleAdapter(OpenAIAdapter):
    """OpenAIAdapter pointed at a self-hosted OpenAI-compatible server.

    The base URL is read, in order, from the `base_url` constructor kwarg,
    then the `VLM_HARNESS_BASE_URL` environment variable. `api_key` defaults
    to a placeholder ("EMPTY") since most self-hosted servers don't check it,
    but the OpenAI SDK requires a non-empty string.
    """

    def __init__(
        self,
        model_id: str,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        resolved_base_url = base_url or os.environ.get("VLM_HARNESS_BASE_URL")
        if not resolved_base_url:
            raise ValueError(
                "OpenAICompatibleAdapter needs a base URL: pass base_url=... or "
                "set the VLM_HARNESS_BASE_URL environment variable "
                "(e.g. http://localhost:8000/v1)."
            )

        try:
            import openai as _openai
        except ImportError:
            raise ImportError("pip install vlm-evaluation-harness[openai]")

        self._client = _openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
            base_url=resolved_base_url,
        )
        self._model_id = model_id

    @property
    def cost_per_million_input_tokens(self) -> float | None:
        return None

    @property
    def cost_per_million_output_tokens(self) -> float | None:
        return None
