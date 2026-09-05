"""Tests for the two learned-compression methods: flashvlm and erase.

Both fall back to a fixed-seed random scorer when no trained weights are
supplied (mirroring learnpruner) -- these tests check the untrained/trained
metadata flag, shape correctness, and that supplied weights actually drive
selection rather than being ignored.
"""

from __future__ import annotations

import numpy as np
import pytest

from vlm_compress import create_compressor


def test_flashvlm_untrained_by_default_and_trained_when_weights_given():
    hidden = np.random.rand(1, 10, 4)

    untrained = create_compressor("flashvlm", target_ratio=0.5).compress(hidden)
    assert untrained.metadata["trained"] is False
    assert untrained.hidden_states.shape == (1, 5, 4)

    trained = create_compressor(
        "flashvlm", target_ratio=0.5, config={"weights": [1.0, 0.0, 0.0, 0.0], "layer": 2}
    ).compress(hidden)
    assert trained.metadata["trained"] is True
    assert trained.metadata["layer"] == 2


def test_flashvlm_keeps_highest_gate_score_tokens():
    compressor = create_compressor("flashvlm", target_ratio=0.5, config={"weights": [1.0, 0.0]})
    hidden = np.array([[[3.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 0.0]]])

    result = compressor.compress(hidden)

    assert set(result.token_indices[0].tolist()) == {0, 2}


def test_flashvlm_weight_dim_mismatch_raises():
    compressor = create_compressor("flashvlm", target_ratio=0.5, config={"weights": [1.0, 0.0]})
    hidden = np.random.rand(1, 8, 4)
    with pytest.raises(ValueError, match="must have shape"):
        compressor.compress(hidden)


def test_erase_untrained_by_default_and_trained_when_weights_given():
    hidden = np.random.rand(1, 8, 4)

    untrained = create_compressor("erase", target_ratio=0.5).compress(hidden)
    assert untrained.metadata["trained"] is False
    assert untrained.metadata["merged"] is True
    assert untrained.hidden_states.shape == (1, 4, 4)

    trained = create_compressor(
        "erase",
        target_ratio=0.5,
        config={
            "importance_weights": [1.0, 0.0, 0.0, 0.0],
            "retention_weights": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
    ).compress(hidden)
    assert trained.metadata["trained"] is True
    assert np.isfinite(trained.hidden_states).all()


def test_erase_importance_weight_dim_mismatch_raises():
    compressor = create_compressor(
        "erase", target_ratio=0.5, config={"importance_weights": [1.0, 0.0]}
    )
    hidden = np.random.rand(1, 8, 4)
    with pytest.raises(ValueError, match="importance_weights.*must have shape"):
        compressor.compress(hidden)
