"""FastV: prune visual tokens using attention from the last text token at an early LLM layer.

Chen et al., "An Image is Worth 1/2 Tokens After Layer 2," 2024
(arXiv:2403.06764). At a chosen layer K (default: 2), rank visual tokens by
the attention they receive from the last text token, keep the top
`target_ratio`, and drop the rest for all subsequent layers.

This module implements the *selection* logic only. Installing the forward
hook that captures layer-K attention and re-applies the resulting mask is a
model-family concern (see `vlm_compress.adapters`, not yet implemented) --
this compressor's `compress()` just needs the attention signal handed to it
as `attention_weights`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vlm_compress.base import CompressionResult, TokenCompressor
from vlm_compress.methods._tensor_utils import gather_tokens, to_numpy
from vlm_compress.registry import register_method


def _reduce_attention_to_scores(attention_weights: np.ndarray, n_vis: int) -> np.ndarray:
    """Normalize whatever attention shape was passed to (B, N_vis) importance scores.

    Accepted shapes, all assumed to already be sliced/aligned so the last
    axis indexes visual tokens:
      (B, N_vis)                       -- already-reduced per-token scores
      (B, n_heads, N_vis)              -- per-head, last text token's row
      (B, n_heads, seq_len, N_vis)     -- full attention map; last query row is used
    """
    arr = attention_weights
    if arr.ndim == 2:
        scores = arr
    elif arr.ndim == 3:
        scores = arr.mean(axis=1)
    elif arr.ndim == 4:
        scores = arr[:, :, -1, :].mean(axis=1)
    else:
        raise ValueError(f"unsupported attention_weights shape {arr.shape}")

    if scores.shape[-1] != n_vis:
        raise ValueError(
            f"attention_weights last dim ({scores.shape[-1]}) must match "
            f"visual token count ({n_vis})"
        )
    return scores


@register_method("fastv")
class FastVCompressor(TokenCompressor):
    """Keeps the `target_ratio` of visual tokens with highest last-text-token attention.

    Config:
      pruning_layer: int, LLM layer index the attention signal was captured
        at (default 2, matching the paper). Metadata only here -- enforcing
        *which* layer's attention was captured is the caller's job.
    """

    def __init__(self, target_ratio: float, config: dict | None = None) -> None:
        super().__init__(target_ratio, config)
        self.pruning_layer = self.config.get("pruning_layer", 2)

    def compress(
        self,
        visual_hidden_states: Any,
        text_hidden_states: Any | None = None,
        attention_mask: Any | None = None,
        attention_weights: Any | None = None,
    ) -> CompressionResult:
        if attention_weights is None:
            raise ValueError(
                "FastV requires `attention_weights` (last text token's attention "
                "over visual tokens at the pruning layer); got None"
            )

        n_tokens = visual_hidden_states.shape[1]
        n_keep = self._n_keep(n_tokens)

        scores = _reduce_attention_to_scores(to_numpy(attention_weights), n_tokens)
        # Keep original left-to-right token order among survivors (matches
        # the paper's implementation and keeps positional embeddings sane).
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
                "pruning_layer": self.pruning_layer,
            },
        )
