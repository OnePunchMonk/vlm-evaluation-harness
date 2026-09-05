"""GlimpsePrune: query-aware region selection for dynamic-resolution models.

Given a text query and a low-resolution overview of the image, learns which
image *regions* (tiles) to process at full resolution. This module
implements the selection given the overview-to-tile relevance scores (the
"glimpse policy" output) plus a token-to-tile map -- training the policy
itself is out of scope, same caveat as LearnPruner. Particularly effective
on OCR/document tasks because it learns to zoom into text regions.

Because tiles carry different token counts (dynamic-res tiling) and
different images select different numbers of tiles to hit the same token
budget, output is a per-sample list of arrays rather than a stacked tensor.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vlm_compress.base import CompressionResult, TokenCompressor
from vlm_compress.methods._tensor_utils import to_numpy
from vlm_compress.registry import register_method


@register_method("glimpseprune")
class GlimpsePruneCompressor(TokenCompressor):
    """Selects tiles by query-relevance score until the token budget is met, keeps their tokens.

    Requires:
      attention_weights: (B, n_tiles) relevance score per tile (the glimpse
        policy's output; higher = more relevant to the query).
      config["tile_token_map"]: list of length n_tiles, where
        tile_token_map[t] is a list/array of token indices belonging to
        tile t. Must partition range(N_vis) (every visual token belongs to
        exactly one tile). Same map is assumed to apply to every sample in
        the batch (true for a fixed tiling grid; dynamic-res models that
        vary tile count across images should call this per-sample).
    """

    def __init__(self, target_ratio: float, config: dict | None = None) -> None:
        super().__init__(target_ratio, config)
        tile_map = self.config.get("tile_token_map")
        if tile_map is None:
            raise ValueError("GlimpsePrune requires config['tile_token_map']")
        self.tile_token_map = [np.asarray(t, dtype=int) for t in tile_map]

    def compress(
        self,
        visual_hidden_states: Any,
        text_hidden_states: Any | None = None,
        attention_mask: Any | None = None,
        attention_weights: Any | None = None,
    ) -> CompressionResult:
        if attention_weights is None:
            raise ValueError(
                "GlimpsePrune requires `attention_weights` (per-tile relevance scores)"
            )

        hidden = to_numpy(visual_hidden_states)
        batch_size, n_tokens, _dim = hidden.shape
        n_keep_budget = self._n_keep(n_tokens)

        tile_scores = to_numpy(attention_weights)
        if tile_scores.shape[-1] != len(self.tile_token_map):
            raise ValueError(
                f"attention_weights last dim ({tile_scores.shape[-1]}) must match "
                f"number of tiles ({len(self.tile_token_map)})"
            )

        per_sample_hidden = []
        per_sample_indices = []
        keep_counts = []
        for b in range(batch_size):
            order = np.argsort(-tile_scores[b])
            selected_tokens: list[int] = []
            for tile_idx in order:
                if len(selected_tokens) >= n_keep_budget:
                    break
                selected_tokens.extend(self.tile_token_map[tile_idx].tolist())
            idx = np.sort(np.array(selected_tokens, dtype=int))
            per_sample_indices.append(idx)
            per_sample_hidden.append(hidden[b, idx])
            keep_counts.append(len(idx))

        uniform = len(set(keep_counts)) == 1
        hidden_out: Any = np.stack(per_sample_hidden, axis=0) if uniform else per_sample_hidden
        indices_out: Any = np.stack(per_sample_indices, axis=0) if uniform else per_sample_indices

        return CompressionResult(
            hidden_states=hidden_out,
            token_indices=indices_out,
            metadata={
                "method": self.name,
                "tokens_total": n_tokens,
                "tokens_kept_per_sample": keep_counts,
                "token_budget": n_keep_budget,
                "uniform_batch": uniform,
            },
        )
