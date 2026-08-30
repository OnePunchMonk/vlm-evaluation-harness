"""OpenAI (GPT-4o / GPT-4-vision) adapter."""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path

from PIL import Image

from vlm_evaluation_harness.adapters.base import ConversationTurn, PromptPart, VLMResponse
from vlm_evaluation_harness.pricing import get_pricing
from vlm_evaluation_harness.retry import with_retries


def _encode_image(image: Image.Image | str) -> dict:
    if isinstance(image, str):
        if image.startswith("http"):
            return {"type": "image_url", "image_url": {"url": image}}
        path = Path(image)
        if path.exists():
            image = Image.open(path)
        else:
            raise ValueError(f"Cannot resolve image: {image}")

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    data = base64.standard_b64encode(buf.getvalue()).decode()
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{data}"},
    }


def _build_content(images: list, prompt: str, parts: list[PromptPart] | None) -> list[dict]:
    """Build one message's `content` array, honoring interleaved ordering
    when `parts` is given (see PromptPart) and falling back to today's
    images-then-text otherwise."""
    if parts:
        return [
            {"type": "text", "text": p.text} if p.kind == "text" else _encode_image(p.image)
            for p in parts
        ]
    content = [_encode_image(img) for img in images]
    content.append({"type": "text", "text": prompt})
    return content


def _build_messages(
    images: list,
    prompt: str,
    system: str | None,
    history: list[ConversationTurn] | None,
    parts: list[PromptPart] | None = None,
) -> list[dict]:
    content = _build_content(images, prompt, parts)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    for turn in history or []:
        turn_content = [_encode_image(img) for img in turn.images]
        turn_content.append({"type": "text", "text": turn.text})
        messages.append({"role": turn.role, "content": turn_content})
    messages.append({"role": "user", "content": content})
    return messages


class OpenAIAdapter:
    """Adapter for GPT-4o and other OpenAI vision models."""

    def __init__(self, model_id: str = "gpt-4o", api_key: str | None = None):
        try:
            import openai as _openai
        except ImportError:
            raise ImportError("pip install vlm-evaluation-harness[openai]")

        self._client = _openai.OpenAI(api_key=api_key)
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
        return (2048, 2048)

    @property
    def cost_per_million_input_tokens(self) -> float | None:
        return get_pricing("openai", self._model_id)[0]

    @property
    def cost_per_million_output_tokens(self) -> float | None:
        return get_pricing("openai", self._model_id)[1]

    def generate(
        self,
        images: list[Image.Image | str],
        prompt: str,
        system: str | None = None,
        history: list[ConversationTurn] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        parts: list[PromptPart] | None = None,
    ) -> VLMResponse:
        messages = _build_messages(images, prompt, system, history, parts)

        t0 = time.perf_counter()
        response = with_retries(
            lambda: self._client.chat.completions.create(
                model=self._model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        return VLMResponse(
            text=response.choices[0].message.content or "",
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            latency_ms=latency_ms,
            model_id=response.model,
        )


class AsyncOpenAIAdapter:
    """Async variant of `OpenAIAdapter`, using `openai.AsyncOpenAI`.

    Exposes `agenerate()` (not `generate()` -- this does not implement the
    synchronous `VLMAdapter` protocol) for a future async runner to call.
    `engine/runner.py`'s evaluation loop is synchronous today and does not
    call this.
    """

    def __init__(self, model_id: str = "gpt-4o", api_key: str | None = None):
        try:
            import openai as _openai
        except ImportError:
            raise ImportError("pip install vlm-evaluation-harness[openai]")

        self._client = _openai.AsyncOpenAI(api_key=api_key)
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
        return (2048, 2048)

    @property
    def cost_per_million_input_tokens(self) -> float | None:
        return get_pricing("openai", self._model_id)[0]

    @property
    def cost_per_million_output_tokens(self) -> float | None:
        return get_pricing("openai", self._model_id)[1]

    async def agenerate(
        self,
        images: list[Image.Image | str],
        prompt: str,
        system: str | None = None,
        history: list[ConversationTurn] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        parts: list[PromptPart] | None = None,
    ) -> VLMResponse:
        messages = _build_messages(images, prompt, system, history, parts)

        t0 = time.perf_counter()
        response = await self._client.chat.completions.create(
            model=self._model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        return VLMResponse(
            text=response.choices[0].message.content or "",
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            latency_ms=latency_ms,
            model_id=response.model,
        )
