"""OpenAI (GPT-4o / GPT-4-vision) adapter."""

from __future__ import annotations

from vlm_harness.adapters._chat_completions import ChatCompletionsAdapter


class OpenAIAdapter(ChatCompletionsAdapter):
    """Adapter for GPT-4o and other OpenAI vision models."""

    _PRICING: dict[str, tuple[float, float]] = {
        "gpt-4o": (5.0, 15.0),
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4-turbo": (10.0, 30.0),
    }

    def __init__(self, model_id: str = "gpt-4o", api_key: str | None = None):
        try:
            import openai as _openai
        except ImportError:
            raise ImportError("pip install vlm-harness[openai]")

        self._client = _openai.OpenAI(api_key=api_key)
        self._model_id = model_id

    @property
    def max_resolution(self) -> tuple[int, int] | None:
        return (2048, 2048)

    @property
    def cost_per_million_input_tokens(self) -> float | None:
        return self._PRICING.get(self._model_id, (0.0, 0.0))[0]

    @property
    def cost_per_million_output_tokens(self) -> float | None:
        return self._PRICING.get(self._model_id, (0.0, 0.0))[1]
