from __future__ import annotations

import numpy as np

from vlm_compress import create_compressor


def test_random_keeps_expected_token_count():
    compressor = create_compressor("random", target_ratio=0.25, config={"seed": 0})
    hidden = np.random.rand(2, 100, 8)
    result = compressor.compress(hidden)

    assert result.hidden_states.shape == (2, 25, 8)
    assert result.token_indices.shape == (2, 25)
    assert result.metadata["tokens_kept"] == 25
    assert result.metadata["tokens_total"] == 100


def test_random_indices_are_unique_and_sorted_per_row():
    compressor = create_compressor("random", target_ratio=0.5, config={"seed": 1})
    hidden = np.random.rand(3, 40, 4)
    result = compressor.compress(hidden)

    for row in result.token_indices:
        assert len(set(row.tolist())) == len(row)
        assert list(row) == sorted(row)


def test_random_gathered_values_match_source():
    compressor = create_compressor("random", target_ratio=0.5, config={"seed": 2})
    hidden = np.arange(2 * 10 * 3).reshape(2, 10, 3).astype(float)
    result = compressor.compress(hidden)

    for b in range(2):
        for k, idx in enumerate(result.token_indices[b]):
            assert np.array_equal(result.hidden_states[b, k], hidden[b, idx])


def test_random_ratio_of_one_keeps_all_tokens():
    compressor = create_compressor("random", target_ratio=1.0)
    hidden = np.random.rand(1, 16, 4)
    result = compressor.compress(hidden)
    assert result.hidden_states.shape == (1, 16, 4)


def test_random_is_deterministic_given_seed():
    hidden = np.random.rand(1, 50, 4)
    a = create_compressor("random", target_ratio=0.3, config={"seed": 42}).compress(hidden)
    b = create_compressor("random", target_ratio=0.3, config={"seed": 42}).compress(hidden)
    assert np.array_equal(a.token_indices, b.token_indices)
