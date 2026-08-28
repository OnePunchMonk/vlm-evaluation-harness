"""Base protocol for text-to-image (generative) adapters.

Discriminative VLM adapters map (images, text) -> text. Generative adapters
map (text prompt) -> image. The two are different enough (different inputs,
different cost model, different metrics) that they get a separate protocol
rather than being bolted onto VLMAdapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from PIL import Image


@dataclass
class T2IResponse:
    """Standardized response from any text-to-image backend."""

    image: Image.Image
    latency_ms: float = 0.0
    model_id: str = ""
    cost_usd: float = 0.0
    seed: int | None = None
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class T2IAdapter(Protocol):
    """Interface every text-to-image backend must implement."""

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
        """Generate one image from a text prompt."""
        ...

    @property
    def model_id(self) -> str:
        """Canonical model identifier."""
        ...

    @property
    def cost_per_image_usd(self) -> float | None:
        """Flat cost per generated image in USD, or None for local models."""
        ...
