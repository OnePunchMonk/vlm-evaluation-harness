"""Common data structures and the compressor interface all methods implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CompressionResult:
    """Output of a single `TokenCompressor.compress()` call.

    `hidden_states` and `token_indices` are left as whatever array/tensor
    type was passed in (numpy or torch) -- this package does not force a
    dependency on either at the base-class level.
    """

    hidden_states: Any
    token_indices: Any
    metadata: dict = field(default_factory=dict)


class TokenCompressor(ABC):
    """Base class for a visual-token compression method.

    Implementations decide *which* visual tokens survive; `create_compressor`
    is the intended construction path so callers don't import method classes
    directly. Subclasses should accept `target_ratio` and a method-specific
    `config` dict.
    """

    name: str = "base"

    def __init__(self, target_ratio: float, config: dict | None = None) -> None:
        if not 0.0 < target_ratio <= 1.0:
            raise ValueError(f"target_ratio must be in (0, 1], got {target_ratio}")
        self.target_ratio = target_ratio
        self.config = config or {}

    @abstractmethod
    def compress(
        self,
        visual_hidden_states: Any,
        text_hidden_states: Any | None = None,
        attention_mask: Any | None = None,
        attention_weights: Any | None = None,
    ) -> CompressionResult:
        """Reduce `visual_hidden_states` (B, N_vis, D) to (B, N_kept, D).

        `text_hidden_states`, `attention_mask`, and `attention_weights` are
        optional signals a given method may use (e.g. FastV needs
        `attention_weights` from its pruning layer); methods that don't need
        a signal ignore it.
        """
        raise NotImplementedError

    def _n_keep(self, n_tokens: int) -> int:
        return max(1, round(n_tokens * self.target_ratio))
