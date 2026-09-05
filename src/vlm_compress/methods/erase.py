"""ERASE: a learned "eraser" module that merges (not drops) low-importance tokens.

Differs from PruMerge in *how* anchors and merge weights are chosen:
PruMerge picks anchors from vision-encoder [CLS] attention and merges by
plain cosine-similarity nearest-anchor assignment (no learning involved).
ERASE instead trains a small module to (a) score token importance directly
from hidden states -- no attention signal needed -- and (b) predict how
much of each erased token's information to retain in its target anchor.
Training that module end-to-end is out of scope for this package (same
caveat as LearnPruner/FlashVLM): what's implemented here is the
inference-time scorer + merge, parameterized by weights you supply once
trained. Without supplied weights, falls back to fixed-seed random
weights and reports `metadata["trained"] = False`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vlm_compress.base import CompressionResult, TokenCompressor
from vlm_compress.methods._tensor_utils import to_numpy
from vlm_compress.registry import register_method


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@register_method("erase")
class ERASECompressor(TokenCompressor):
    """Learned anchor selection + learned soft merge of erased tokens into anchors.

    Config:
      importance_weights: array-like (D,), trained scorer for picking
        anchors (score = hidden @ importance_weights + importance_bias).
        If omitted, a fixed-seed random scorer is used.
      importance_bias: float, default 0.0.
      retention_weights: array-like (2*D,), trained scorer predicting how
        much of an erased token's info to retain when merging into its
        anchor: retention = sigmoid([anchor; erased_token] @ retention_weights).
        If omitted, a fixed-seed random scorer is used.
      seed: int, seed for fallback random scorers. Default 0.
    """

    def __init__(self, target_ratio: float, config: dict | None = None) -> None:
        super().__init__(target_ratio, config)
        self._importance_weights = self.config.get("importance_weights")
        self._importance_bias = self.config.get("importance_bias", 0.0)
        self._retention_weights = self.config.get("retention_weights")
        self._seed = self.config.get("seed", 0)
        self._trained = (
            self._importance_weights is not None and self._retention_weights is not None
        )

    def _importance_scorer(self, dim: int) -> np.ndarray:
        if self._importance_weights is not None:
            weights = np.asarray(self._importance_weights, dtype=float)
            if weights.shape != (dim,):
                raise ValueError(
                    f"config['importance_weights'] must have shape ({dim},), "
                    f"got {weights.shape}"
                )
            return weights
        return np.random.default_rng(self._seed).normal(size=dim)

    def _retention_scorer(self, dim: int) -> np.ndarray:
        if self._retention_weights is not None:
            weights = np.asarray(self._retention_weights, dtype=float)
            if weights.shape != (2 * dim,):
                raise ValueError(
                    f"config['retention_weights'] must have shape ({2 * dim},), "
                    f"got {weights.shape}"
                )
            return weights
        return np.random.default_rng(self._seed + 1).normal(size=2 * dim)

    def compress(
        self,
        visual_hidden_states: Any,
        text_hidden_states: Any | None = None,
        attention_mask: Any | None = None,
        attention_weights: Any | None = None,
    ) -> CompressionResult:
        hidden = to_numpy(visual_hidden_states)
        batch_size, n_tokens, dim = hidden.shape
        n_keep = self._n_keep(n_tokens)

        importance_weights = self._importance_scorer(dim)
        retention_weights = self._retention_scorer(dim)
        scores = hidden @ importance_weights + self._importance_bias  # (B, N_vis)

        merged_batch = []
        anchor_indices_batch = []
        for b in range(batch_size):
            anchor_idx = np.sort(np.argpartition(-scores[b], n_keep - 1)[:n_keep])
            anchor_indices_batch.append(anchor_idx)

            anchors = hidden[b, anchor_idx]  # (n_keep, D)
            sims = hidden[b] @ anchors.T  # (N_vis, n_keep), nearest-anchor by dot product
            assignment = np.argmax(sims, axis=-1)
            assignment[anchor_idx] = np.arange(n_keep)

            merged = np.zeros_like(anchors)
            for a in range(n_keep):
                members = np.where(assignment == a)[0]
                anchor_vec = anchors[a]
                pair_features = np.concatenate(
                    [np.tile(anchor_vec, (len(members), 1)), hidden[b, members]], axis=-1
                )
                retention = _sigmoid(pair_features @ retention_weights)  # (len(members),)
                retention = retention / (retention.sum() + 1e-12)
                merged[a] = (hidden[b, members] * retention[:, None]).sum(axis=0)
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
                "trained": self._trained,
            },
        )
