"""Anthropic (Claude) adapter."""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path

from PIL import Image

from vlm_harness.adapters.base import ConversationTurn, VLMResponse


class AnthropicAdapter:
    """Adapter for Claude models via the Anthropic API."""

    # Pricing as of 2026-04 (USD per 1M tokens)
    _PRICING: dict[str, tuple[float, float]] = {
        "claude-opus-4-6": (15.0, 75.0),
        "claude-sonnet-4-6": (3.0, 15.0),
        "claude-haiku-4-5": (0.80, 4.0),
        "claude-haiku-4-5-20251001": (0.80, 4.0),
    }

    def __init__(self, model_id: str = "claude-opus-4-6", api_key: str | None = None):
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError("pip install vlm-harness[anthropic]")

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
        return self._PRICING.get(base, (0.0, 0.0))[0]

    @property
    def cost_per_million_output_tokens(self) -> float | None:
        base = self._model_id.split("-20")[0]
        return self._PRICING.get(base, (0.0, 0.0))[1]

    def generate(
        self,
        images: list[Image.Image | str],
        prompt: str,
        system: str | None = None,
        history: list[ConversationTurn] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> VLMResponse:
        content = []
        for img in images:
            content.append(self._encode_image(img))
        content.append({"type": "text", "text": prompt})

        messages = []
        if history:
            for turn in history:
                turn_content = []
                for img in turn.images:
                    turn_content.append(self._encode_image(img))
                turn_content.append({"type": "text", "text": turn.text})
                messages.append({"role": turn.role, "content": turn_content})
        messages.append({"role": "user", "content": content})

        kwargs: dict = {
            "model": self._model_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        t0 = time.perf_counter()
        response = self._client.messages.create(**kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000

        text_blocks = [b.text for b in response.content if getattr(b, "type", None) == "text"]
        return VLMResponse(
            text=text_blocks[0] if text_blocks else "",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
            model_id=response.model,
        )

    def _encode_image(self, image: Image.Image | str) -> dict:
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
