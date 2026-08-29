"""Anthropic (Claude) adapter."""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path

from PIL import Image

from vlm_evaluation_harness.adapters.base import ConversationTurn, VLMResponse
from vlm_evaluation_harness.pricing import get_pricing
from vlm_evaluation_harness.retry import with_retries


def _encode_image(image: Image.Image | str) -> dict:
    if isinstance(image, str):
        path = Path(image)
        if path.exists():
            image = Image.open(path)
        elif image.startswith("http"):
            return {"type": "image", "source": {"type": "url", "url": image}}
        else:
            raise ValueError(f"Cannot resolve image: {image}")

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    data = base64.standard_b64encode(buf.getvalue()).decode()
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": data},
    }


def _response_text(response) -> str:
    """Concatenate every text block in the response.

    Indexing `content[0]` breaks whenever the first block is not text --
    which is exactly what happens when extended thinking is enabled.
    """
    return "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )


def _build_messages(
    images: list, prompt: str, history: list[ConversationTurn] | None
) -> list[dict]:
    content = [_encode_image(img) for img in images]
    content.append({"type": "text", "text": prompt})

    messages = []
    for turn in history or []:
        turn_content = [_encode_image(img) for img in turn.images]
        turn_content.append({"type": "text", "text": turn.text})
        messages.append({"role": turn.role, "content": turn_content})
    messages.append({"role": "user", "content": content})
    return messages


class AnthropicAdapter:
    """Adapter for Claude models via the Anthropic API."""

    def __init__(self, model_id: str = "claude-opus-4-6", api_key: str | None = None):
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError("pip install vlm-evaluation-harness[anthropic]")

        self._client = _anthropic.Anthropic(api_key=api_key)
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def supports_multi_image(self) -> bool:
        return True

    @property
    def supports_video(self) -> bool:
        return False

    @property
    def max_resolution(self) -> tuple[int, int] | None:
        return (8096, 8096)

    @property
    def cost_per_million_input_tokens(self) -> float | None:
        base = self._model_id.split("-20")[0]  # strip date suffix
        return get_pricing("anthropic", base)[0]

    @property
    def cost_per_million_output_tokens(self) -> float | None:
        base = self._model_id.split("-20")[0]
        return get_pricing("anthropic", base)[1]

    def _request_kwargs(
        self,
        images: list,
        prompt: str,
        system: str | None,
        history: list[ConversationTurn] | None,
        max_tokens: int,
        temperature: float,
    ) -> dict:
        kwargs: dict = {
            "model": self._model_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": _build_messages(images, prompt, history),
        }
        if system:
            kwargs["system"] = system
        return kwargs

    def generate(
        self,
        images: list[Image.Image | str],
        prompt: str,
        system: str | None = None,
        history: list[ConversationTurn] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> VLMResponse:
        kwargs = self._request_kwargs(images, prompt, system, history, max_tokens, temperature)

        t0 = time.perf_counter()
        response = with_retries(lambda: self._client.messages.create(**kwargs))
        latency_ms = (time.perf_counter() - t0) * 1000

        return VLMResponse(
            text=_response_text(response),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
            model_id=response.model,
        )


class AsyncAnthropicAdapter:
    """Async variant of `AnthropicAdapter`, using `anthropic.AsyncAnthropic`.

    Exposes `agenerate()` (not `generate()` -- this does not implement the
    synchronous `VLMAdapter` protocol) for a future async runner to call.
    `engine/runner.py`'s evaluation loop is synchronous today and does not
    call this; wiring an async runner path is a separate, larger change
    outside this adapter's scope.
    """

    def __init__(self, model_id: str = "claude-opus-4-6", api_key: str | None = None):
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError("pip install vlm-evaluation-harness[anthropic]")

        self._client = _anthropic.AsyncAnthropic(api_key=api_key)
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def supports_multi_image(self) -> bool:
        return True

    @property
    def supports_video(self) -> bool:
        return False

    @property
    def max_resolution(self) -> tuple[int, int] | None:
        return (8096, 8096)

    @property
    def cost_per_million_input_tokens(self) -> float | None:
        base = self._model_id.split("-20")[0]
        return get_pricing("anthropic", base)[0]

    @property
    def cost_per_million_output_tokens(self) -> float | None:
        base = self._model_id.split("-20")[0]
        return get_pricing("anthropic", base)[1]

    async def agenerate(
        self,
        images: list[Image.Image | str],
        prompt: str,
        system: str | None = None,
        history: list[ConversationTurn] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> VLMResponse:
        kwargs: dict = {
            "model": self._model_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": _build_messages(images, prompt, history),
        }
        if system:
            kwargs["system"] = system

        t0 = time.perf_counter()
        response = await self._client.messages.create(**kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000

        return VLMResponse(
            text=_response_text(response),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
            model_id=response.model,
        )
