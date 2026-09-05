"""LLaVA-PruMerge: rank by [CLS] attention, merge (not drop) low-importance tokens.

Shang et al., "LLaVA-PruMerge," 2024 (arXiv:2403.15388). Vision-encoder-side:
uses the vision encoder's [CLS] token attention to pick a small set of
"anchor" tokens, then merges every other token into its nearest (by cosine
similarity) anchor via an attention-weighted average, rather than discarding
it outright. Preserves more information than hard pruning at the same
compression ratio.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vlm_compress.base import CompressionResult, TokenCompressor
from vlm_compress.methods._tensor_utils import to_numpy
from vlm_compress.registry import register_method


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


@register_method("prumerge")
class PruMergeCompressor(TokenCompressor):
    """Selects anchor tokens by CLS attention, merges remaining tokens into their nearest anchor.

    Requires `attention_weights` = the vision encoder's [CLS]-to-patch
    attention, shape (B, N_vis) or (B, n_heads, N_vis).
    """

    def compress(
        self,
        visual_hidden_states: Any,
        text_hidden_states: Any | None = None,
        attention_mask: Any | None = None,
        attention_weights: Any | None = None,
    ) -> CompressionResult:
        if attention_weights is None:
            raise ValueError("PruMerge requires `attention_weights` ([CLS]-to-patch attention)")

        hidden = to_numpy(visual_hidden_states)
        batch_size, n_tokens, _dim = hidden.shape
        n_keep = self._n_keep(n_tokens)

        cls_attn = to_numpy(attention_weights)
        if cls_attn.ndim == 3:
            cls_attn = cls_attn.mean(axis=1)
        if cls_attn.shape[-1] != n_tokens:
            raise ValueError(
                f"attention_weights last dim ({cls_attn.shape[-1]}) must match "
                f"visual token count ({n_tokens})"
            )

        merged_batch = []
        anchor_indices_batch = []
        for b in range(batch_size):
            anchor_idx = np.sort(np.argpartition(-cls_attn[b], n_keep - 1)[:n_keep])
            anchor_indices_batch.append(anchor_idx)

            anchors = hidden[b, anchor_idx]
            sims = _cosine_sim(hidden[b], anchors)  # (N_vis, n_keep)
            assignment = np.argmax(sims, axis=-1)  # nearest anchor per token
            assignment[anchor_idx] = np.arange(n_keep)  # anchors always assigned to themselves

            merged = np.zeros_like(anchors)
            for a in range(n_keep):
                members = np.where(assignment == a)[0]
                weights = np.clip(cls_attn[b, members], 1e-6, None)
                weights = weights / weights.sum()
                merged[a] = (hidden[b, members] * weights[:, None]).sum(axis=0)
            merged_batch.append(merged)

        return CompressionResult(
            hidden_states=np.stack(merged_batch, axis=0),
            token_indices=np.stack(anchor_indices_batch, axis=0),
            metadata={
                "method": self.name,
                "tokens_kept": n_keep,
                "tokens_total": n_tokens,
                "ratio": n_keep / n_tokens,
                "merged": True,
            },
        )
