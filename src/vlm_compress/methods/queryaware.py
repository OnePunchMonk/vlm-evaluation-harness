"""Query-aware compression: keep visual tokens most relevant to the text query.

Unlike FastV/VisPruner/DART/TopV (which score tokens by *attention traffic*
observed inside the LLM) or PruMerge (which scores by [CLS] attention alone),
this method scores each visual token by its similarity to the text query
embedding directly -- "show me the red car" keeps tokens whose hidden state
is close to the query representation, independent of any attention map.
This lets it run before the LLM sees any visual tokens at all (encoder-side,
zero extra forward passes) and makes the compression ratio meaningfully
query-conditioned rather than query-agnostic.

Score for visual token i, given a pooled query vector q (D,):
    score_i = cosine_sim(hidden_i, q)
When `text_hidden_states` carries multiple query tokens (B, T, D), q is the
mean-pooled query vector per sample (mean over T), optionally restricted to
positions where `attention_mask` is 1.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vlm_compress.base import CompressionResult, TokenCompressor
from vlm_compress.methods._tensor_utils import gather_tokens, to_numpy
from vlm_compress.registry import register_method


def _pool_query(text_hidden: np.ndarray, attention_mask: np.ndarray | None) -> np.ndarray:
    """Mean-pool (B, T, D) text hidden states to (B, D), masking out padding."""
    if text_hidden.ndim == 2:
        return text_hidden
    if attention_mask is None:
        return text_hidden.mean(axis=1)
    mask = attention_mask.astype(text_hidden.dtype)[..., None]
    denom = np.clip(mask.sum(axis=1), 1e-6, None)
    return (text_hidden * mask).sum(axis=1) / denom


def _cosine_sim_to_query(hidden: np.ndarray, query: np.ndarray) -> np.ndarray:
    """hidden: (B, N, D), query: (B, D) -> (B, N) cosine similarity."""
    hidden_norm = hidden / (np.linalg.norm(hidden, axis=-1, keepdims=True) + 1e-12)
    query_norm = query / (np.linalg.norm(query, axis=-1, keepdims=True) + 1e-12)
    return np.einsum("bnd,bd->bn", hidden_norm, query_norm)


@register_method("queryaware")
class QueryAwareCompressor(TokenCompressor):
    """Keeps visual tokens with the highest cosine similarity to the pooled text query.

    Requires `text_hidden_states` of shape (B, D) (already pooled) or
    (B, T, D) (per-token query embeddings, mean-pooled here using
    `attention_mask` if given). Config:
      background_floor: float in [0, 1), default 0.0. Fraction of the token
        budget reserved for the globally highest-norm-attention-free tokens
        (a uniform sample) regardless of query similarity, so completely
        query-irrelevant regions (e.g. sky) aren't dropped to zero context
        when that hurts downstream grounding. 0.0 disables this (pure
        query-similarity ranking).
      seed: int, default 0. RNG seed for the background-floor sample.
    """

    def __init__(self, target_ratio: float, config: dict | None = None) -> None:
        super().__init__(target_ratio, config)
        self.background_floor = self.config.get("background_floor", 0.0)
        if not 0.0 <= self.background_floor < 1.0:
            raise ValueError("background_floor must be in [0, 1)")
        self.seed = self.config.get("seed", 0)

    def compress(
        self,
        visual_hidden_states: Any,
        text_hidden_states: Any | None = None,
        attention_mask: Any | None = None,
        attention_weights: Any | None = None,
    ) -> CompressionResult:
        if text_hidden_states is None:
            raise ValueError(
                "QueryAware requires `text_hidden_states` (the text query embedding)"
            )

        hidden = to_numpy(visual_hidden_states)
        batch_size, n_tokens, _dim = hidden.shape
        n_keep = self._n_keep(n_tokens)

        text_hidden = to_numpy(text_hidden_states)
        mask = to_numpy(attention_mask) if attention_mask is not None else None
        query = _pool_query(text_hidden, mask)
        if query.shape[-1] != hidden.shape[-1]:
            raise ValueError(
                f"text_hidden_states dim ({query.shape[-1]}) must match "
                f"visual hidden dim ({hidden.shape[-1]})"
            )

        scores = _cosine_sim_to_query(hidden, query)

        n_reserved = int(round(n_keep * self.background_floor))
        n_ranked = n_keep - n_reserved

        rng = np.random.default_rng(self.seed)
        indices_batch = []
        for b in range(batch_size):
            ranked_idx = np.argpartition(-scores[b], n_ranked - 1)[:n_ranked]
            if n_reserved > 0:
                remaining = np.setdiff1d(np.arange(n_tokens), ranked_idx)
                reserved_idx = rng.choice(remaining, size=n_reserved, replace=False)
                idx = np.concatenate([ranked_idx, reserved_idx])
            else:
                idx = ranked_idx
            indices_batch.append(np.sort(idx))

        indices = np.stack(indices_batch, axis=0)
        kept = gather_tokens(visual_hidden_states, indices)

        return CompressionResult(
            hidden_states=kept,
            token_indices=indices,
            metadata={
                "method": self.name,
                "tokens_kept": n_keep,
                "tokens_total": n_tokens,
                "ratio": n_keep / n_tokens,
                "query_conditioned": True,
                "background_floor": self.background_floor,
                "mean_query_similarity": scores[np.arange(batch_size)[:, None], indices].mean(
                    axis=1
                ).tolist(),
            },
        )
