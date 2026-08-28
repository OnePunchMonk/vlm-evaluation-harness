"""OpenAI Images API adapter (gpt-image-1, dall-e-3, dall-e-2)."""

from __future__ import annotations

import base64
import io
import time

from PIL import Image

from vlm_evaluation_harness.adapters.generative.base import T2IResponse

# USD per image at default quality, keyed by model id (as of 2026-04).
_PRICING: dict[str, float] = {
    "gpt-image-1": 0.04,
    "dall-e-3": 0.04,
    "dall-e-2": 0.02,
}


class OpenAIImageAdapter:
    """Adapter for text-to-image generation via the OpenAI Images API."""

    def __init__(self, model_id: str = "gpt-image-1", api_key: str | None = None):
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
    def cost_per_image_usd(self) -> float | None:
        return _PRICING.get(self._model_id)

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        seed: int | None = None,
        width: int = 1024,
        height: int = 1024,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 30,
    ) -> T2IResponse:
        # The OpenAI Images API has no negative_prompt/seed/steps knobs; fold
        # a negative prompt into the instruction text since that's the only
        # lever available.
        full_prompt = prompt
        if negative_prompt:
            full_prompt = f"{prompt}\n\nAvoid: {negative_prompt}"

        t0 = time.perf_counter()
        response = self._client.images.generate(
            model=self._model_id,
            prompt=full_prompt,
            size=f"{width}x{height}",
            n=1,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        item = response.data[0]
        if getattr(item, "b64_json", None):
            image = Image.open(io.BytesIO(base64.b64decode(item.b64_json)))
        else:
            import httpx
            image = Image.open(io.BytesIO(httpx.get(item.url).content))

        return T2IResponse(
            image=image.convert("RGB"),
            latency_ms=latency_ms,
            model_id=self._model_id,
            cost_usd=self.cost_per_image_usd or 0.0,
            seed=seed,
            metadata={"revised_prompt": getattr(item, "revised_prompt", None)},
        )
