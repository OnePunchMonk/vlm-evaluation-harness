"""Regression test: HuggingFaceAdapter must actually construct.

`transformers>=4.46` renamed `AutoModelForVision2Seq` to
`AutoModelForImageTextToText`; importing the old name unconditionally makes
the adapter's __init__ raise ImportError on any current transformers
install. Every other HuggingFaceAdapter test bypasses __init__ via
`object.__new__` (to avoid needing torch/transformers), which is exactly
why this broke silently -- this test is the one that actually calls it,
against a real tiny checkpoint.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from vlm_evaluation_harness.adapters.huggingface import HuggingFaceAdapter  # noqa: E402

_TINY_MODEL = "hf-internal-testing/tiny-random-BlipForConditionalGeneration"


def test_adapter_constructs_on_current_transformers():
    adapter = HuggingFaceAdapter(model_id=_TINY_MODEL, device="cpu")
    assert adapter.model_id == _TINY_MODEL
