"""Tests for the query-aware compressor: relevance is driven by text-query similarity."""

from __future__ import annotations

import numpy as np
import pytest

from vlm_compress import create_compressor, list_methods


def test_queryaware_registered():
    assert "queryaware" in list_methods()


def test_queryaware_requires_text_hidden_states():
    compressor = create_compressor("queryaware", target_ratio=0.5)
    hidden = np.random.rand(1, 8, 4)
    with pytest.raises(ValueError, match="requires `text_hidden_states`"):
        compressor.compress(hidden)


def test_queryaware_keeps_tokens_similar_to_query():
    compressor = create_compressor("queryaware", target_ratio=0.5)

    # 4 tokens aligned with the query direction, 4 tokens orthogonal to it.
    query = np.array([[1.0, 0.0]])
    aligned = np.tile(np.array([1.0, 0.01]), (4, 1))
    orthogonal = np.tile(np.array([0.0, 1.0]), (4, 1))
    hidden = np.concatenate([aligned, orthogonal], axis=0)[None, ...]

    result = compressor.compress(hidden, text_hidden_states=query)

    assert result.hidden_states.shape == (1, 4, 2)
    assert set(result.token_indices[0].tolist()) == {0, 1, 2, 3}
    assert result.metadata["query_conditioned"] is True


def test_queryaware_pools_multi_token_query_with_attention_mask():
    compressor = create_compressor("queryaware", target_ratio=0.5)

    query_tokens = np.array([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]])  # padding token in the middle
    mask = np.array([[1, 0, 1]])  # mask out the [0, 1] padding token

    aligned = np.tile(np.array([1.0, 0.01]), (2, 1))
    orthogonal = np.tile(np.array([0.0, 1.0]), (2, 1))
    hidden = np.concatenate([aligned, orthogonal], axis=0)[None, ...]

    result = compressor.compress(hidden, text_hidden_states=query_tokens, attention_mask=mask)

    assert set(result.token_indices[0].tolist()) == {0, 1}


def test_queryaware_background_floor_reserves_budget_for_non_ranked_tokens():
    compressor = create_compressor(
        "queryaware", target_ratio=0.5, config={"background_floor": 0.5, "seed": 0}
    )
    query = np.array([[1.0, 0.0]])
    aligned = np.tile(np.array([1.0, 0.01]), (4, 1))
    orthogonal = np.tile(np.array([0.0, 1.0]), (4, 1))
    hidden = np.concatenate([aligned, orthogonal], axis=0)[None, ...]

    result = compressor.compress(hidden, text_hidden_states=query)

    kept = set(result.token_indices[0].tolist())
    assert len(kept) == 4
    # With a 50% background floor, at least one orthogonal (query-irrelevant) token survives.
    assert kept & {4, 5, 6, 7}


def test_queryaware_dim_mismatch_raises():
    compressor = create_compressor("queryaware", target_ratio=0.5)
    hidden = np.random.rand(1, 8, 4)
    query = np.random.rand(1, 3)
    with pytest.raises(ValueError, match="must match"):
        compressor.compress(hidden, text_hidden_states=query)
