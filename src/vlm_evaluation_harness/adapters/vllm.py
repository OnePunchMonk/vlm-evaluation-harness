"""Adapter for vLLM's OpenAI-compatible server mode.

Talks to `vllm serve <model> --port 8000` (vLLM's built-in OpenAI-compatible
HTTP server), not the in-process `vllm.LLM` Python API — this keeps the
harness installable without a GPU or the `vllm` package itself. SGLang's
`sglang.launch_server` exposes the same OpenAI-compatible wire format, so
this adapter works against either backend; the class name reflects the
common default port/setup, not a wire-format difference.
"""

from __future__ import annotations

import os

from vlm_evaluation_harness.adapters.openai_compatible import OpenAICompatibleAdapter

DEFAULT_BASE_URL = "http://localhost:8000/v1"


class VLLMAdapter(OpenAICompatibleAdapter):
    """OpenAICompatibleAdapter defaulting to a local vLLM/SGLang server.

    Start the server first: `vllm serve <model> --port 8000` (or the
    equivalent `python -m sglang.launch_server --port 8000`). Base URL
    resolution order: `base_url` kwarg, then `VLM_HARNESS_BASE_URL`, then
    `http://localhost:8000/v1`.
    """

    def __init__(self, model_id: str, base_url: str | None = None, api_key: str | None = None):
        resolved_base_url = (
            base_url or os.environ.get("VLM_HARNESS_BASE_URL") or DEFAULT_BASE_URL
        )
        super().__init__(model_id=model_id, base_url=resolved_base_url, api_key=api_key)
