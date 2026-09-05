"""LearnPruner: a tiny learned per-token importance predictor.

Trains a lightweight (paper: 0.53M-param) predictor over vision-encoder
output to score token importance, reaching high accuracy retention at
aggressive compression ratios. Training the predictor is out of scope for
this module (it needs labeled importance data and a training loop external
to this package) -- what's implemented here is the *inference-time*
scoring + selection, parameterized by a linear probe whose weights you
supply once trained.

Without supplied weights, this falls back to a randomly-initialized probe
and says so loudly in the result metadata: it will run, but its rankings
are meaningless until you plug in trained weights via `config["weights"]`
(and optionally `config["bias"]`).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vlm_compress.base import CompressionResult, TokenCompressor
from vlm_compress.methods._tensor_utils import gather_tokens, to_numpy
from vlm_compress.registry import register_method


@register_method("learnpruner")
class LearnPrunerCompressor(TokenCompressor):
    """Linear-probe token scorer: score = hidden_state @ weights + bias.

    Config:
      weights: array-like (D,), trained probe weights. If omitted, a
        fixed-seed random probe is used and `metadata["trained"]` is False.
      bias: float, default 0.0.
      seed: int, seed for the fallback random probe. Default 0.
    """

    def __init__(self, target_ratio: float, config: dict | None = None) -> None:
        super().__init__(target_ratio, config)
        self._weights = self.config.get("weights")
        self._bias = self.config.get("bias", 0.0)
        self._seed = self.config.get("seed", 0)
        self._trained = self._weights is not None

    def _probe_weights(self, dim: int) -> np.ndarray:
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

        weights = self._probe_weights(dim)
        scores = hidden @ weights + self._bias  # (B, N_vis)

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
                "trained": self._trained,
            },
        )
