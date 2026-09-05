"""Random token pruning: the sanity-check baseline.

UniPruneBench (see design doc, section 5.3) found random pruning at
moderate-to-aggressive ratios is a surprisingly strong baseline -- any
method claiming an improvement should beat this, not just beat "full"
accuracy minus some hand-wavy tax.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vlm_compress.base import CompressionResult, TokenCompressor
from vlm_compress.methods._tensor_utils import gather_tokens, to_numpy
from vlm_compress.registry import register_method


@register_method("random")
class RandomCompressor(TokenCompressor):
    """Uniformly samples `target_ratio` of visual tokens per batch element, without replacement."""

    def __init__(self, target_ratio: float, config: dict | None = None) -> None:
        super().__init__(target_ratio, config)
        seed = self.config.get("seed")
        self._rng = np.random.default_rng(seed)

    def compress(
        self,
        visual_hidden_states: Any,
        text_hidden_states: Any | None = None,
        attention_mask: Any | None = None,
        attention_weights: Any | None = None,
    ) -> CompressionResult:
        shape = getattr(visual_hidden_states, "shape", None) or to_numpy(visual_hidden_states).shape
        batch_size, n_tokens = shape[0], shape[1]
        n_keep = self._n_keep(n_tokens)

        indices = np.stack(
            [
                np.sort(self._rng.choice(n_tokens, size=n_keep, replace=False))
                for _ in range(batch_size)
            ]
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
            },
        )
