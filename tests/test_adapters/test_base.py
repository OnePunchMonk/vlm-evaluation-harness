"""Tests for adapter base types."""

import pytest

from vlm_evaluation_harness.adapters.base import ChoiceScores, ConversationTurn, VLMResponse


def test_vlm_response_total_tokens():
    r = VLMResponse(text="hello", input_tokens=100, output_tokens=50)
    assert r.total_tokens == 150


def test_vlm_response_defaults():
    r = VLMResponse(text="hello")
    assert r.input_tokens == 0
    assert r.output_tokens == 0
    assert r.latency_ms == 0.0
    assert r.model_id == ""
    assert r.metadata == {}


def test_conversation_turn():
    turn = ConversationTurn(role="user", text="What is in the image?")
    assert turn.role == "user"
    assert turn.images == []


def test_conversation_turn_with_images():
    from PIL import Image
    img = Image.new("RGB", (100, 100))
    turn = ConversationTurn(role="user", text="Describe", images=[img])
    assert len(turn.images) == 1


class TestChoiceScoresArgmax:
    """Three choices constructed so 'none'/'length'/'char_length' each pick
    a different winner — proves the normalization modes are not equivalent
    and that each is wired up to the correct arithmetic."""

    def _scores(self) -> ChoiceScores:
        # A: short raw logprob winner. B: length-normalized winner.
        # C: char-length-normalized winner (few tokens, many characters).
        return ChoiceScores(
            logprobs=[-4.0, -6.0, -9.0],
            logprobs_per_token=[-4.0, -1.0, -3.0],
        )

    def test_default_is_length_normalized(self):
        scores = self._scores()
        assert scores.argmax() == 1  # B

    def test_none_normalization_picks_raw_winner(self):
        scores = self._scores()
        assert scores.argmax(normalization="none") == 0  # A

    def test_length_normalization_picks_per_token_winner(self):
        scores = self._scores()
        assert scores.argmax(normalization="length") == 1  # B

    def test_char_length_normalization_picks_per_char_winner(self):
        scores = self._scores()
        # A: -4/1, B: -6/2 = -3, C: -9/9 = -1 -> C wins
        assert scores.argmax(normalization="char_length", char_lengths=[1, 2, 9]) == 2

    def test_legacy_length_normalized_bool_still_works(self):
        scores = self._scores()
        assert scores.argmax(length_normalized=True) == 1
        assert scores.argmax(length_normalized=False) == 0

    def test_char_length_without_char_lengths_raises(self):
        scores = self._scores()
        with pytest.raises(ValueError, match="char_lengths"):
            scores.argmax(normalization="char_length")

    def test_char_length_with_wrong_length_list_raises(self):
        scores = self._scores()
        with pytest.raises(ValueError, match="char_lengths"):
            scores.argmax(normalization="char_length", char_lengths=[1, 2])

    def test_unknown_normalization_raises(self):
        scores = self._scores()
        with pytest.raises(ValueError, match="unknown log-likelihood normalization"):
            scores.argmax(normalization="pmi")
