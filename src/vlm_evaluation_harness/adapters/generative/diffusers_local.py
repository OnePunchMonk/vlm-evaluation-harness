"""Local text-to-image adapter backed by HuggingFace `diffusers`."""

from __future__ import annotations

import time

from vlm_evaluation_harness.adapters.generative.base import T2IResponse


class DiffusersAdapter:
    """Adapter for local diffusion models (Stable Diffusion, FLUX, ...) via diffusers."""

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-1",
        device: str = "auto",
        dtype: str = "float16",
    ):
        try:
            import torch
            from diffusers import DiffusionPipeline
        except ImportError:
            raise ImportError("pip install vlm-evaluation-harness[generative]")

        self._torch = torch
        resolved_dtype = getattr(torch, dtype, torch.float32)
        self._device = (
            device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=resolved_dtype)
        self._pipeline = self._pipeline.to(self._device)
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def cost_per_image_usd(self) -> float | None:
        return None  # local inference has no per-image API cost

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        seed: int | None = None,
        width: int = 512,
        height: int = 512,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 30,
    ) -> T2IResponse:
        generator = None
        if seed is not None:
            generator = self._torch.Generator(device=self._device).manual_seed(seed)

        t0 = time.perf_counter()
        result = self._pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        return T2IResponse(
            image=result.images[0].convert("RGB"),
            latency_ms=latency_ms,
            model_id=self._model_id,
            cost_usd=0.0,
            seed=seed,
        )
