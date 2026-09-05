"""FlashVLM: a learned per-layer visual token gate, trained end-to-end with the VLM.

The paper trains a small gating function at each decoder layer that decides,
per visual token, whether the layer keeps attending to it -- gates are
binary at inference, relaxed with a straight-through estimator during
training so gradients flow. Training end-to-end against a real model's
loss is out of scope for this package (same caveat as LearnPruner): what's
implemented here is the *inference-time* gate -- a linear scorer per
"gate" (one call = one layer's gating decision) whose weights you supply
once trained elsewhere. Without supplied weights, falls back to a
fixed-seed random gate and reports `metadata["trained"] = False`.

Unlike LearnPruner (one global importance ranking), FlashVLM's gate is
meant to be applied independently at multiple layers with different
learned weights -- construct one `FlashVLMCompressor` per gated layer,
each with that layer's `config["weights"]`/`config["bias"]`, and call it
at the corresponding point in the forward pass.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vlm_compress.base import CompressionResult, TokenCompressor
from vlm_compress.methods._tensor_utils import gather_tokens, to_numpy
from vlm_compress.registry import register_method


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@register_method("flashvlm")
class FlashVLMCompressor(TokenCompressor):
    """Per-layer learned gate: gate_score = sigmoid(hidden @ weights + bias).

    Selection keeps the `target_ratio` fraction of tokens with the highest
    gate score (a hard top-k at inference, rather than the training-time
    straight-through relaxation, which only matters for gradient flow).

    Config:
      layer: int, which decoder layer this gate belongs to (for bookkeeping
        only -- this class scores whatever hidden states it's given).
        Default 0.
      weights: array-like (D,), trained gate weights. If omitted, a
        fixed-seed random gate is used and `metadata["trained"]` is False.
      bias: float, default 0.0.
      seed: int, seed for the fallback random gate. Default 0.
    """

    def __init__(self, target_ratio: float, config: dict | None = None) -> None:
        super().__init__(target_ratio, config)
        self.layer = self.config.get("layer", 0)
        self._weights = self.config.get("weights")
        self._bias = self.config.get("bias", 0.0)
        self._seed = self.config.get("seed", 0)
        self._trained = self._weights is not None

    def _gate_weights(self, dim: int) -> np.ndarray:
        if self._weights is not None:
            weights = np.asarray(self._weights, dtype=float)
            if weights.shape != (dim,):
                raise ValueError(f"config['weights'] must have shape ({dim},), got {weights.shape}")
            return weights
        return np.random.default_rng(self._seed).normal(size=dim)

    def compress(
        self,
        visual_hidden_states: Any,
        text_hidden_states: Any | None = None,
        attention_mask: Any | None = None,
        attention_weights: Any | None = None,
    ) -> CompressionResult:
        hidden = to_numpy(visual_hidden_states)
        _batch_size, n_tokens, dim = hidden.shape
        n_keep = self._n_keep(n_tokens)

        weights = self._gate_weights(dim)
        gate_scores = _sigmoid(hidden @ weights + self._bias)  # (B, N_vis)

        indices = np.sort(
            np.argpartition(-gate_scores, n_keep - 1, axis=-1)[:, :n_keep], axis=-1
        )
        kept = gather_tokens(visual_hidden_states, indices)

        return CompressionResult(
            hidden_states=kept,
            token_indices=indices,
            metadata={
                "method": self.name,
                "tokens_kept": n_keep,
                "tokens_total": n_tokens,
                "ratio": n_keep / n_tokens,
                "layer": self.layer,
                "trained": self._trained,
                "mean_gate_score": gate_scores.mean(axis=1).tolist(),
            },
        )
