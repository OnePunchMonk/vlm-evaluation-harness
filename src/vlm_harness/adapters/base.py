"""Base protocol and data models for VLM adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from PIL import Image


@dataclass
class VLMResponse:
    """Standardized response from any VLM backend."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    model_id: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class ConversationTurn:
    """A single turn in a multi-turn conversation."""

    role: str  # "user" or "assistant"
    text: str
    images: list[Image.Image | str] = field(default_factory=list)


@runtime_checkable
class VLMAdapter(Protocol):
    """Interface every model backend must implement."""

    def generate(
        self,
        images: list[Image.Image | str],
        prompt: str,
        system: str | None = None,
        history: list[ConversationTurn] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> VLMResponse:
        """Generate a response given images and a text prompt."""
        ...

    @property
    def model_id(self) -> str:
        """Canonical model identifier."""
        ...

    @property
    def supports_multi_image(self) -> bool:
        """Whether the model can process multiple images in one call."""
        ...

    @property
    def supports_video(self) -> bool:
        """Whether the model accepts video input."""
        ...

    @property
    def max_resolution(self) -> tuple[int, int] | None:
        """Maximum image resolution (width, height), or None if unconstrained."""
        ...

    @property
    def cost_per_million_input_tokens(self) -> float | None:
        """Cost in USD per 1M input tokens, or None for local models."""
        ...

    @property
    def cost_per_million_output_tokens(self) -> float | None:
        """Cost in USD per 1M output tokens, or None for local models."""
        ...
