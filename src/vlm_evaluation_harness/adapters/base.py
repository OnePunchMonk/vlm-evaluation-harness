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
    # True when this response was served from the on-disk cache rather than
    # by calling the model. Cached calls are excluded from latency stats.
    cached: bool = False
    metadata: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_cache_payload(self) -> dict:
        return {
            "text": self.text,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "model_id": self.model_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_cache_payload(cls, payload: dict) -> VLMResponse:
        return cls(
            text=payload["text"],
            input_tokens=payload.get("input_tokens", 0),
            output_tokens=payload.get("output_tokens", 0),
            latency_ms=payload.get("latency_ms", 0.0),
            model_id=payload.get("model_id", ""),
            cached=True,
            metadata=payload.get("metadata", {}),
        )


#: Normalization modes `ChoiceScores.argmax` understands.
KNOWN_LL_NORMALIZATIONS = {"none", "length", "char_length"}


@dataclass
class ChoiceScores:
    """Log-probabilities assigned to each candidate continuation."""

    logprobs: list[float]
    # Log-probability normalized by continuation token count. Reported
    # separately because unnormalized scores favour short options.
    logprobs_per_token: list[float]
    latency_ms: float = 0.0
    model_id: str = ""

    def argmax(
        self,
        length_normalized: bool = True,
        normalization: str | None = None,
        char_lengths: list[int] | None = None,
    ) -> int:
        """Pick the winning choice under a log-probability normalization.

        `normalization` (when given) takes priority over the legacy
        `length_normalized` bool, kept for backward compatibility:
          - "none": raw summed logprob. Favours short continuations.
          - "length": logprob / token count (this class's default prior to
            `normalization` existing, and the default here too).
          - "char_length": logprob / character count of the continuation.
            Useful when comparing choices whose tokenization diverges a lot
            from their surface length (e.g. one choice tokenizes to far
            fewer/more tokens per character than another). Requires
            `char_lengths` (one int per choice, same order as `logprobs`).
        """
        mode = normalization if normalization is not None else (
            "length" if length_normalized else "none"
        )
        if mode not in KNOWN_LL_NORMALIZATIONS:
            raise ValueError(
                f"unknown log-likelihood normalization {mode!r} "
                f"(known: {sorted(KNOWN_LL_NORMALIZATIONS)})"
            )
        if mode == "none":
            scores = self.logprobs
        elif mode == "length":
            scores = self.logprobs_per_token
        else:  # char_length
            if char_lengths is None or len(char_lengths) != len(self.logprobs):
                raise ValueError(
                    "normalization='char_length' requires char_lengths, one per choice"
                )
            scores = [
                lp / max(1, n_chars) for lp, n_chars in zip(self.logprobs, char_lengths)
            ]
        return max(range(len(scores)), key=scores.__getitem__)


@dataclass
class ConversationTurn:
    """A single turn in a multi-turn conversation."""

    role: str  # "user" or "assistant"
    text: str
    images: list[Image.Image | str] = field(default_factory=list)


@dataclass
class PromptPart:
    """One ordered piece of an interleaved prompt: either a text segment or
    an image, in the sequence they should appear to the model.

    Populated by PromptFormatter.format() only when the benchmark manifest
    sets image_config.placement == "interleaved" (see benchmarks/schema.py);
    empty otherwise, in which case adapters fall back to their existing
    images-then-text (or images-after-text) behavior using `images`/`prompt`
    directly. An adapter that doesn't understand `parts` can simply ignore
    it -- this is why it's an addition to generate()'s signature rather than
    a replacement for `images`/`prompt`.
    """

    kind: str  # "text" | "image"
    text: str | None = None
    image: Image.Image | str | None = None


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
        parts: list[PromptPart] | None = None,
    ) -> VLMResponse:
        """Generate a response given images and a text prompt.

        `parts`, when given, carries the same images/text in the exact
        interleaved order they should be sent to the model (see PromptPart).
        Adapters that support genuine interleaving should prefer it over
        `images`/`prompt` when present; adapters that don't can ignore it.
        """
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


@runtime_checkable
class ChoiceScoringAdapter(Protocol):
    """Optional capability: score candidate answers by log-likelihood.

    This is how open-weight models are compared on multiple-choice
    benchmarks — every option is scored under the model and the highest wins,
    with no free-text generation and therefore no answer-extraction step to
    fail. Numbers produced by generating text and regexing out a letter are
    not comparable to published leaderboard figures.

    Adapters that cannot do this (hosted APIs that expose no log-probs)
    simply do not implement it; the runner checks `supports_choice_scoring`.
    """

    @property
    def supports_choice_scoring(self) -> bool: ...

    def score_choices(
        self,
        images: list[Image.Image | str],
        prompt: str,
        choices: list[str],
        system: str | None = None,
    ) -> ChoiceScores:
        """Return the log-probability of each choice as a continuation of the prompt."""
        ...


def supports_choice_scoring(adapter: object) -> bool:
    """Whether `adapter` can score choices by log-likelihood."""
    return bool(getattr(adapter, "supports_choice_scoring", False)) and hasattr(
        adapter, "score_choices"
    )
