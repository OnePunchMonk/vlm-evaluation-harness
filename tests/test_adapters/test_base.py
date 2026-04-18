"""Tests for adapter base types."""

import pytest
from vlm_harness.adapters.base import VLMResponse, ConversationTurn


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
