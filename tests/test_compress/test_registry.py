from __future__ import annotations

import pytest

from vlm_compress import create_compressor, list_methods
from vlm_compress.base import TokenCompressor
from vlm_compress.registry import register_method


def test_list_methods_includes_builtins():
    methods = list_methods()
    assert "random" in methods
    assert "fastv" in methods
    assert methods == sorted(methods)


def test_create_compressor_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown compression method"):
        create_compressor("not-a-real-method", target_ratio=0.5)


def test_create_compressor_passes_model_family_into_config():
    compressor = create_compressor("random", target_ratio=0.5, model_family="qwen2.5-vl")
    assert compressor.config["model_family"] == "qwen2.5-vl"


def test_create_compressor_invalid_ratio_raises():
    with pytest.raises(ValueError, match="target_ratio"):
        create_compressor("random", target_ratio=0.0)
    with pytest.raises(ValueError, match="target_ratio"):
        create_compressor("random", target_ratio=1.5)


def test_register_method_conflicting_name_raises():
    class _Dummy(TokenCompressor):
        def compress(self, *args, **kwargs):
            raise NotImplementedError

    list_methods()  # ensure builtins (including "random") are already registered
    with pytest.raises(ValueError, match="already registered"):
        register_method("random")(_Dummy)
