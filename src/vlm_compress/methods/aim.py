"""AIM: two-stage hybrid compression -- spatial-coverage clustering, then attention pruning.

Stage 1 (encoder-side): cluster tokens by spatial position and keep the
highest-scoring token per cluster, guaranteeing coverage across the image
rather than letting one high-attention region dominate. Stage 2 (LLM-side):
further prune the stage-1 survivors using LLM attention, same top-k
selection as FastV/VisPruner. Combines spatial coverage with semantic
relevance.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vlm_compress.base import CompressionResult, TokenCompressor
from vlm_compress.methods._tensor_utils import to_numpy
from vlm_compress.registry import register_method


def _reduce_to_scores(attn: np.ndarray, n_vis: int) -> np.ndarray:
    if attn.ndim == 2:
        scores = attn
    elif attn.ndim == 3:
        scores = attn.mean(axis=1)
    elif attn.ndim == 4:
        scores = attn[:, :, -1, :].mean(axis=1)
    else:
        raise ValueError(f"unsupported attention shape {attn.shape}")
    if scores.shape[-1] != n_vis:
        raise ValueError(
            f"attention last dim ({scores.shape[-1]}) must match visual token count ({n_vis})"
        )
    return scores


@register_method("aim")
class AIMCompressor(TokenCompressor):
    """Stage 1: spatially-clustered coverage selection. Stage 2: attention-based pruning.

    `attention_weights` must be a dict: {"encoder": <stage-1 scores>, "llm": <stage-2 scores>},
    each accepted in the same shapes as FastV's `attention_weights`.

    Config:
      stage1_ratio: float, fraction of tokens kept after stage 1. Default:
        sqrt(target_ratio), so the two stages contribute roughly equally
        (in log-ratio terms) to the overall `target_ratio`.
    """

    def __init__(self, target_ratio: float, config: dict | None = None) -> None:
        super().__init__(target_ratio, config)
        self.stage1_ratio = self.config.get("stage1_ratio", target_ratio**0.5)
        if not target_ratio <= self.stage1_ratio <= 1.0:
            raise ValueError("stage1_ratio must be in [target_ratio, 1.0]")

    def compress(
        self,
        visual_hidden_states: Any,
        text_hidden_states: Any | None = None,
        attention_mask: Any | None = None,
        attention_weights: Any | None = None,
    ) -> CompressionResult:
        has_both_stages = (
            isinstance(attention_weights, dict)
            and "encoder" in attention_weights
            and "llm" in attention_weights
        )
        if not has_both_stages:
            raise ValueError('AIM requires attention_weights={"encoder": ..., "llm": ...}')

        hidden = to_numpy(visual_hidden_states)
        batch_size, n_tokens, _dim = hidden.shape
        n_stage1_keep = max(1, round(n_tokens * self.stage1_ratio))
        n_final_keep = self._n_keep(n_tokens)

        encoder_scores = _reduce_to_scores(to_numpy(attention_weights["encoder"]), n_tokens)
        grid = int(round(n_tokens**0.5))
        n_clusters = min(n_stage1_keep, n_tokens)

        stage1_indices_batch = []
        for b in range(batch_size):
            if grid * grid == n_tokens and n_clusters <= grid * grid:
                # Assign tokens to n_clusters spatial bands by row, keep the
                # top-scoring token within each band (approximate spatial coverage).
                bands = np.array_split(np.arange(n_tokens), n_clusters)
                picked = [band[np.argmax(encoder_scores[b, band])] for band in bands]
            else:
                picked = np.argpartition(-encoder_scores[b], n_stage1_keep - 1)[:n_stage1_keep]
            stage1_indices_batch.append(np.sort(np.array(picked, dtype=int)))

        llm_scores = _reduce_to_scores(to_numpy(attention_weights["llm"]), n_tokens)

        final_indices_batch = []
        final_hidden_batch = []
        for b in range(batch_size):
            candidates = stage1_indices_batch[b]
            candidate_scores = llm_scores[b, candidates]
            k2 = min(n_final_keep, len(candidates))
            top_local = np.argpartition(-candidate_scores, k2 - 1)[:k2]
            final_idx = np.sort(candidates[top_local])
            final_indices_batch.append(final_idx)
            final_hidden_batch.append(hidden[b, final_idx])

        indices = np.stack(final_indices_batch, axis=0)
        return CompressionResult(
            hidden_states=np.stack(final_hidden_batch, axis=0),
            token_indices=indices,
            metadata={
                "method": self.name,
                "tokens_total": n_tokens,
                "tokens_kept": indices.shape[1],
                "stage1_ratio": self.stage1_ratio,
                "ratio": indices.shape[1] / n_tokens,
            },
        )
