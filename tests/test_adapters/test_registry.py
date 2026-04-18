"""Tests for adapter registry."""

import pytest
from vlm_harness.adapters.registry import list_adapters, get_adapter


def test_list_adapters_returns_dict():
    adapters = list_adapters()
    assert isinstance(adapters, dict)
    assert "anthropic" in adapters
    assert "openai" in adapters
    assert "huggingface" in adapters


def test_invalid_spec_raises():
    with pytest.raises(ValueError, match="Invalid model spec"):
        get_adapter("no-colon-here")


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        get_adapter("unknownprovider:some-model")


def test_missing_dependency_raises_import_error():
    # This will try to import the anthropic package.
    # In a test env without it, should raise ImportError with helpful message.
    try:
        adapter = get_adapter("anthropic:claude-opus-4-6")
        # If anthropic is installed, this should succeed
        assert adapter is not None
    except ImportError as e:
        assert "pip install" in str(e)
