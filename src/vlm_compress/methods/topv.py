"""TopV: per-head top-k visual token selection, compatible with block-sparse attention.

Liu et al., "TopV," CVPR 2025. Instead of a single global top-k over an
averaged attention score, each attention head nominates its own top-k
visual tokens; the union is trimmed (or padded, if too small) to
`target_ratio` using the heads-averaged score as the tiebreaker. Framed this
way the pruning decision maps directly onto FlashAttention's block-sparse
mode -- restrict attention computation to the selected keys, don't mask.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vlm_compress.base import CompressionResult, TokenCompressor
from vlm_compress.methods._tensor_utils import gather_tokens, to_numpy
from vlm_compress.registry import register_method


@register_method("topv")
class TopVCompressor(TokenCompressor):
    """Per-head top-k visual token selection with union + score-based trimming to `target_ratio`.

    Requires `attention_weights` shaped (B, n_heads, N_vis) or
    (B, n_heads, seq_len, N_vis) (last query row is used in the 4D case).
    """

    def compress(
        self,
        visual_hidden_states: Any,
        text_hidden_states: Any | None = None,
        attention_mask: Any | None = None,
        attention_weights: Any | None = None,
    ) -> CompressionResult:
        if attention_weights is None:
            raise ValueError("TopV requires per-head `attention_weights`")

        n_tokens = visual_hidden_states.shape[1]
        n_keep = self._n_keep(n_tokens)

        attn = to_numpy(attention_weights)
        if attn.ndim == 4:
            attn = attn[:, :, -1, :]
        if attn.ndim != 3 or attn.shape[-1] != n_tokens:
            raise ValueError(
                f"attention_weights must be (B, n_heads, {n_tokens}) or "
                f"(B, n_heads, seq_len, {n_tokens}); got shape {attn.shape}"
            )

        batch_size, n_heads, _ = attn.shape
        per_head_k = max(1, n_keep // n_heads)
        avg_scores = attn.mean(axis=1)  # (B, N_vis), used as tiebreaker

        indices_batch = []
        for b in range(batch_size):
            union: set[int] = set()
            for h in range(n_heads):
                top_h = np.argpartition(-attn[b, h], per_head_k - 1)[:per_head_k]
                union.update(top_h.tolist())

            ranked = sorted(union, key=lambda i: -avg_scores[b, i])
            if len(ranked) >= n_keep:
                selected = ranked[:n_keep]
            else:
                remaining = [i for i in np.argsort(-avg_scores[b]) if i not in union]
                selected = ranked + remaining[: n_keep - len(ranked)]
            indices_batch.append(np.sort(np.array(selected, dtype=int)))

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
                "per_head_k": per_head_k,
            },
        )
