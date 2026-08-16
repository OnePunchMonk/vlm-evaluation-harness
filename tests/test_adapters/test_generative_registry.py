"""Tests for the T2I adapter registry."""

import pytest

from vlm_harness.adapters.generative.registry import get_t2i_adapter, list_t2i_adapters


def test_list_adapters_returns_dict():
    adapters = list_t2i_adapters()
    assert isinstance(adapters, dict)
    assert "mock" in adapters
    assert "openai" in adapters
    assert "diffusers" in adapters


def test_invalid_spec_raises():
    with pytest.raises(ValueError, match="Invalid model spec"):
        get_t2i_adapter("no-colon-here")


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown T2I provider"):
        get_t2i_adapter("unknownprovider:some-model")


def test_mock_adapter_resolves():
    adapter = get_t2i_adapter("mock:demo-v1")
    assert adapter.model_id == "demo-v1"
