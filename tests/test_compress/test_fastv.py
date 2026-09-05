from __future__ import annotations

import numpy as np
import pytest

from vlm_compress import create_compressor


def test_fastv_requires_attention_weights():
    compressor = create_compressor("fastv", target_ratio=0.5)
    hidden = np.random.rand(1, 10, 4)
    with pytest.raises(ValueError, match="requires `attention_weights`"):
        compressor.compress(hidden)


def test_fastv_keeps_highest_scoring_tokens_2d_scores():
    compressor = create_compressor("fastv", target_ratio=0.5)
    hidden = np.arange(1 * 8 * 2).reshape(1, 8, 2).astype(float)
    # tokens 0,2,4,6 have the highest scores -> should be the survivors
    scores = np.array([[9, 1, 8, 2, 7, 3, 6, 4]], dtype=float)

    result = compressor.compress(hidden, attention_weights=scores)

    assert result.token_indices.tolist() == [[0, 2, 4, 6]]
    assert result.metadata["pruning_layer"] == 2


def test_fastv_accepts_per_head_attention_and_averages():
    compressor = create_compressor("fastv", target_ratio=0.5, config={"pruning_layer": 3})
    hidden = np.random.rand(1, 4, 2)
    # two heads; averaged scores should rank tokens [1, 3] highest
    attn = np.array([[[1, 5, 1, 4], [1, 4, 1, 5]]], dtype=float)

    result = compressor.compress(hidden, attention_weights=attn)

    assert sorted(result.token_indices[0].tolist()) == [1, 3]
    assert result.metadata["pruning_layer"] == 3


def test_fastv_accepts_full_attention_map_uses_last_query_row():
    compressor = create_compressor("fastv", target_ratio=0.5)
    hidden = np.random.rand(1, 4, 2)
    # shape (B, heads=1, seq_len=3, n_vis=4); only last row (index -1) matters
    attn = np.zeros((1, 1, 3, 4))
    attn[0, 0, -1] = [1, 9, 1, 8]

    result = compressor.compress(hidden, attention_weights=attn)

    assert sorted(result.token_indices[0].tolist()) == [1, 3]


def test_fastv_mismatched_attention_shape_raises():
    compressor = create_compressor("fastv", target_ratio=0.5)
    hidden = np.random.rand(1, 4, 2)
    bad_scores = np.random.rand(1, 5)  # wrong last-dim size
    with pytest.raises(ValueError, match="must match"):
        compressor.compress(hidden, attention_weights=bad_scores)
