"""Shared OpenAI chat-completions request/response handling.

Both OpenAIAdapter and OpenAICompatibleAdapter (Ollama, TGI, LM Studio,
vLLM's OpenAI-compat mode, ...) speak the same `/chat/completions` wire
format via the `openai` SDK -- only client construction and pricing differ.
Subclasses set self._client and self._model_id in their own __init__ and
this base class does the rest, so message/image encoding lives in one place.
"""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path

from PIL import Image

from vlm_harness.adapters.base import ConversationTurn, VLMResponse


class ChatCompletionsAdapter:
    """Base for any adapter backed by an OpenAI-compatible chat endpoint."""

    _client: object
    _model_id: str

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
        return None

    def generate(
        self,
        images: list[Image.Image | str],
        prompt: str,
        system: str | None = None,
        history: list[ConversationTurn] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> VLMResponse:
        content: list[dict] = [self._encode_image(img) for img in images]
        content.append({"type": "text", "text": prompt})

        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            for turn in history:
                turn_content = [self._encode_image(img) for img in turn.images]
                turn_content.append({"type": "text", "text": turn.text})
                messages.append({"role": turn.role, "content": turn_content})
        messages.append({"role": "user", "content": content})

        t0 = time.perf_counter()
        response = self._client.chat.completions.create(  # type: ignore[attr-defined]
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

    def _encode_image(self, image: Image.Image | str) -> dict:
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
