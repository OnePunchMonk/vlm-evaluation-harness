"""DART: difficulty-aware, instance-adaptive visual token pruning.

Li et al., "DART," 2025. The compression ratio adapts per-instance based on
how concentrated the attention distribution over visual tokens is: diffuse
attention (hard example) retains more tokens, concentrated attention (easy
example) prunes more aggressively. No training required.

Because the keep-count varies per batch element, `compress()` returns a
*list* of per-sample arrays (rather than a single stacked tensor) whenever
ratios actually differ across the batch -- there is no valid fixed-width
tensor to return otherwise.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vlm_compress.base import CompressionResult, TokenCompressor
from vlm_compress.methods._tensor_utils import to_numpy
from vlm_compress.registry import register_method


def _entropy(scores: np.ndarray) -> np.ndarray:
    """Normalized Shannon entropy per row, in [0, 1]. 1 = diffuse, 0 = concentrated."""
    probs = scores / (scores.sum(axis=-1, keepdims=True) + 1e-12)
    probs = np.clip(probs, 1e-12, 1.0)
    raw_entropy = -(probs * np.log(probs)).sum(axis=-1)
    max_entropy = np.log(scores.shape[-1])
    return raw_entropy / max_entropy


@register_method("dart")
class DARTCompressor(TokenCompressor):
    """Instance-adaptive pruning: ratio scales with attention entropy, between min/max_ratio.

    `target_ratio` is used as the *midpoint* default for `max_ratio` when not
    otherwise configured. Config:
      min_ratio: float, ratio used for the most-concentrated (easiest) instances. Default 0.2.
      max_ratio: float, ratio used for the most-diffuse (hardest) instances. Default target_ratio.
    """

    def __init__(self, target_ratio: float, config: dict | None = None) -> None:
        super().__init__(target_ratio, config)
        self.min_ratio = self.config.get("min_ratio", 0.2)
        self.max_ratio = self.config.get("max_ratio", target_ratio)
        if not self.min_ratio <= self.max_ratio:
            raise ValueError("min_ratio must be <= max_ratio")

    def compress(
        self,
        visual_hidden_states: Any,
        text_hidden_states: Any | None = None,
        attention_mask: Any | None = None,
        attention_weights: Any | None = None,
    ) -> CompressionResult:
        if attention_weights is None:
            raise ValueError("DART requires `attention_weights` to score attention concentration")

        n_tokens = visual_hidden_states.shape[1]
        attn = to_numpy(attention_weights)
        if attn.ndim == 3:
            attn = attn.mean(axis=1)
        elif attn.ndim == 4:
            attn = attn[:, :, -1, :].mean(axis=1)
        if attn.shape[-1] != n_tokens:
            raise ValueError(
                f"attention_weights last dim ({attn.shape[-1]}) must match "
                f"visual token count ({n_tokens})"
            )

        entropy = _entropy(attn)
        per_sample_ratio = self.min_ratio + entropy * (self.max_ratio - self.min_ratio)
        keep_counts = np.maximum(1, np.round(per_sample_ratio * n_tokens)).astype(int)

        hidden_np = to_numpy(visual_hidden_states)
        per_sample_hidden = []
        per_sample_indices = []
        for b, k in enumerate(keep_counts):
            idx = np.sort(np.argpartition(-attn[b], k - 1)[:k])
            per_sample_indices.append(idx)
            per_sample_hidden.append(hidden_np[b, idx])

        uniform = len(set(keep_counts.tolist())) == 1
        hidden_out: Any = np.stack(per_sample_hidden, axis=0) if uniform else per_sample_hidden
        indices_out: Any = np.stack(per_sample_indices, axis=0) if uniform else per_sample_indices

        return CompressionResult(
            hidden_states=hidden_out,
            token_indices=indices_out,
            metadata={
                "method": self.name,
                "tokens_total": n_tokens,
                "tokens_kept_per_sample": keep_counts.tolist(),
                "ratio_per_sample": per_sample_ratio.tolist(),
                "uniform_batch": uniform,
            },
        )
