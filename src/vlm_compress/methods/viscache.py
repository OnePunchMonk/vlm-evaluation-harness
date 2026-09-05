"""VisCache: KV-cache-level eviction for visual tokens, based on cumulative decode attention.

Operates at a different point in the pipeline than the other methods here:
instead of pruning visual tokens before/during prefill, it evicts visual
*KV cache entries* during decoding, based on attention accumulated over
recent decode steps. Orthogonal to prefill-time pruning -- can stack on top
of any other method in this package.

Reuses the `TokenCompressor` interface for consistency (`visual_hidden_states`
here is read as the visual KV entries -- keys or values -- to retain a
subset of; `attention_weights` is the cumulative attention each entry has
received since the last eviction, not a single layer's snapshot). Call
`compress()` periodically during decoding, not once at prefill time.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vlm_compress.base import CompressionResult, TokenCompressor
from vlm_compress.methods._tensor_utils import gather_tokens, to_numpy
from vlm_compress.registry import register_method


@register_method("viscache")
class VisCacheCompressor(TokenCompressor):
    """Evicts visual KV-cache entries with the lowest cumulative attention.

    Config:
      decay: float in [0, 1], optional exponential decay applied to
        `attention_weights` before ranking, for callers that pass raw
        per-step attention history of shape (B, N_vis, n_steps) rather than
        an already-accumulated (B, N_vis) score. Default: no decay (uniform
        average over steps).
    """

    def __init__(self, target_ratio: float, config: dict | None = None) -> None:
        super().__init__(target_ratio, config)
        self.decay = self.config.get("decay")

    def _accumulate(self, attn: np.ndarray) -> np.ndarray:
        if attn.ndim == 2:
            return attn
        if attn.ndim == 3:
            n_steps = attn.shape[-1]
            if self.decay is None:
                return attn.mean(axis=-1)
            weights = self.decay ** np.arange(n_steps - 1, -1, -1)
            weights = weights / weights.sum()
            return attn @ weights
        raise ValueError(f"unsupported attention_weights shape {attn.shape}")

    def compress(
        self,
        visual_hidden_states: Any,
        text_hidden_states: Any | None = None,
        attention_mask: Any | None = None,
        attention_weights: Any | None = None,
    ) -> CompressionResult:
        if attention_weights is None:
            raise ValueError("VisCache requires `attention_weights` (cumulative decode attention)")

        n_tokens = visual_hidden_states.shape[1]
        n_keep = self._n_keep(n_tokens)

        scores = self._accumulate(to_numpy(attention_weights))
        if scores.shape[-1] != n_tokens:
            raise ValueError(
                f"attention_weights last-but-one dim must match visual KV entry count ({n_tokens})"
            )

        indices = np.sort(np.argpartition(-scores, n_keep - 1, axis=-1)[:, :n_keep], axis=-1)
        kept = gather_tokens(visual_hidden_states, indices)

        return CompressionResult(
            hidden_states=kept,
            token_indices=indices,
            metadata={
                "method": self.name,
                "tokens_kept": n_keep,
                "tokens_total": n_tokens,
                "ratio": n_keep / n_tokens,
                "level": "kv_cache",
            },
        )
