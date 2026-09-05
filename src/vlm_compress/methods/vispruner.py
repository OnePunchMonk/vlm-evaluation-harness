"""VisPruner / FasterVLM: prune using an early LLM layer's attention, averaged over text queries.

Zheng et al., "FasterVLM/VisPruner," 2025. Differs from FastV in the
aggregation of the attention signal: FastV uses only the *last* text
token's attention row; VisPruner averages attention from *all* text-token
query positions at the pruning layer, which the paper finds is a stronger
importance signal (less noisy than a single query position).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vlm_compress.base import CompressionResult, TokenCompressor
from vlm_compress.methods._tensor_utils import gather_tokens, to_numpy
from vlm_compress.registry import register_method


def _scores_from_text_query_attention(
    attn: np.ndarray, n_vis: int, n_text_queries: int | None
) -> np.ndarray:
    """Average attention to visual tokens over head dim and over the text-query positions.

    Accepted shapes (last axis = visual tokens, same convention as FastV):
      (B, N_vis)                                -- already reduced
      (B, n_heads, N_vis)                       -- single query position, per head
      (B, n_heads, seq_len, N_vis)               -- full map; averages the last
                                                    `n_text_queries` query rows
                                                    (default: all rows)
    """
    if attn.ndim == 2:
        scores = attn
    elif attn.ndim == 3:
        scores = attn.mean(axis=1)
    elif attn.ndim == 4:
        seq_len = attn.shape[2]
        k = seq_len if n_text_queries is None else min(n_text_queries, seq_len)
        scores = attn[:, :, seq_len - k :, :].mean(axis=(1, 2))
    else:
        raise ValueError(f"unsupported attention_weights shape {attn.shape}")

    if scores.shape[-1] != n_vis:
        raise ValueError(
            f"attention_weights last dim ({scores.shape[-1]}) must match "
            f"visual token count ({n_vis})"
        )
    return scores


@register_method("vispruner")
class VisPrunerCompressor(TokenCompressor):
    """Keeps the highest-scoring visual tokens by all-text-query attention at an early layer.

    Config:
      pruning_layer: int, default 3 (paper sweeps layers 2-3).
      n_text_queries: int | None, number of trailing query positions to
        average over when given a full (B, H, seq_len, N_vis) attention map.
        None (default) averages over every query position in the map.
    """

    def __init__(self, target_ratio: float, config: dict | None = None) -> None:
        super().__init__(target_ratio, config)
        self.pruning_layer = self.config.get("pruning_layer", 3)
        self.n_text_queries = self.config.get("n_text_queries")

    def compress(
        self,
        visual_hidden_states: Any,
        text_hidden_states: Any | None = None,
        attention_mask: Any | None = None,
        attention_weights: Any | None = None,
    ) -> CompressionResult:
        if attention_weights is None:
            raise ValueError(
                "VisPruner requires `attention_weights` from the pruning layer; got None"
            )

        n_tokens = visual_hidden_states.shape[1]
        n_keep = self._n_keep(n_tokens)

        scores = _scores_from_text_query_attention(
            to_numpy(attention_weights), n_tokens, self.n_text_queries
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
                "pruning_layer": self.pruning_layer,
            },
        )
