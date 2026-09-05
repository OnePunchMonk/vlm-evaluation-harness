"""One focused correctness test per remaining method (vispruner, dart, prumerge,
topv, learnpruner, viscache, aim, glimpseprune), plus a smoke test that every
registered method is at least importable/constructible.
"""

from __future__ import annotations

import numpy as np
import pytest

from vlm_compress import create_compressor, list_methods


def test_all_ten_methods_registered():
    assert set(list_methods()) == {
        "random",
        "fastv",
        "vispruner",
        "dart",
        "prumerge",
        "topv",
        "learnpruner",
        "viscache",
        "aim",
        "glimpseprune",
    }


def test_vispruner_keeps_highest_scoring_tokens():
    compressor = create_compressor("vispruner", target_ratio=0.5, config={"pruning_layer": 3})
    hidden = np.random.rand(1, 8, 2)
    scores = np.array([[9, 1, 8, 2, 7, 3, 6, 4]], dtype=float)

    result = compressor.compress(hidden, attention_weights=scores)

    assert result.token_indices.tolist() == [[0, 2, 4, 6]]
    assert result.metadata["pruning_layer"] == 3


def test_dart_prunes_more_for_concentrated_attention():
    compressor = create_compressor(
        "dart", target_ratio=0.6, config={"min_ratio": 0.2, "max_ratio": 0.6}
    )
    hidden = np.random.rand(2, 20, 3)
    concentrated = np.zeros(20)
    concentrated[0] = 1.0
    diffuse = np.ones(20)
    attn = np.stack([concentrated, diffuse])

    result = compressor.compress(hidden, attention_weights=attn)

    kept_counts = result.metadata["tokens_kept_per_sample"]
    assert kept_counts[0] < kept_counts[1]


def test_prumerge_returns_anchor_count_and_valid_merge():
    compressor = create_compressor("prumerge", target_ratio=0.5)
    hidden = np.random.rand(1, 6, 4)
    cls_attn = np.array([[5, 1, 4, 1, 3, 1]], dtype=float)

    result = compressor.compress(hidden, attention_weights=cls_attn)

    assert result.hidden_states.shape == (1, 3, 4)
    assert result.metadata["merged"] is True
    assert np.isfinite(result.hidden_states).all()


def test_topv_returns_target_ratio_token_count():
    compressor = create_compressor("topv", target_ratio=0.5)
    hidden = np.random.rand(1, 8, 2)
    attn = np.random.rand(1, 4, 8)  # 4 heads

    result = compressor.compress(hidden, attention_weights=attn)

    assert result.hidden_states.shape == (1, 4, 2)
    assert len(set(result.token_indices[0].tolist())) == 4


def test_learnpruner_reports_untrained_by_default_and_trained_when_weights_given():
    hidden = np.random.rand(1, 10, 4)

    untrained = create_compressor("learnpruner", target_ratio=0.5).compress(hidden)
    assert untrained.metadata["trained"] is False

    trained = create_compressor(
        "learnpruner", target_ratio=0.5, config={"weights": [1.0, 0.0, 0.0, 0.0]}
    ).compress(hidden)
    assert trained.metadata["trained"] is True
    assert trained.hidden_states.shape == (1, 5, 4)


def test_viscache_evicts_low_cumulative_attention_entries():
    compressor = create_compressor("viscache", target_ratio=0.5)
    kv_entries = np.random.rand(1, 8, 4)
    cumulative_attn = np.array([[9, 1, 8, 2, 7, 3, 6, 4]], dtype=float)

    result = compressor.compress(kv_entries, attention_weights=cumulative_attn)

    assert result.token_indices.tolist() == [[0, 2, 4, 6]]
    assert result.metadata["level"] == "kv_cache"


def test_aim_two_stage_selection():
    compressor = create_compressor("aim", target_ratio=0.25, config={"stage1_ratio": 0.5})
    hidden = np.random.rand(1, 16, 2)
    encoder_scores = np.random.rand(1, 16)
    llm_scores = np.random.rand(1, 16)

    result = compressor.compress(
        hidden, attention_weights={"encoder": encoder_scores, "llm": llm_scores}
    )

    assert result.metadata["tokens_kept"] <= 4 + 1  # ~target_ratio of 16, rounding tolerance
    assert result.hidden_states.shape[0] == 1


def test_aim_requires_dict_attention_weights():
    compressor = create_compressor("aim", target_ratio=0.25)
    hidden = np.random.rand(1, 16, 2)
    with pytest.raises(ValueError, match="requires attention_weights"):
        compressor.compress(hidden, attention_weights=np.random.rand(1, 16))


def test_glimpseprune_keeps_full_tiles_and_respects_relevance_order():
    tile_token_map = [[0, 1], [2, 3], [4, 5], [6, 7]]
    compressor = create_compressor(
        "glimpseprune", target_ratio=0.5, config={"tile_token_map": tile_token_map}
    )
    hidden = np.random.rand(1, 8, 2)
    tile_scores = np.array([[1, 9, 2, 8]], dtype=float)  # tiles 1 and 3 most relevant

    result = compressor.compress(hidden, attention_weights=tile_scores)

    kept = result.token_indices[0]
    assert set(kept.tolist()) == {2, 3, 6, 7}
