"""Equivalence test: HuggingFaceAdapter.generate_batch vs. per-sample generate.

Uses a tiny real vision2seq checkpoint (not a mock) so left-padding and
attention-mask handling in the batched path are actually exercised.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

transformers = pytest.importorskip("transformers")
pytest.importorskip("torch")

from vlm_harness.adapters.huggingface import HuggingFaceAdapter  # noqa: E402

_TINY_MODEL = "hf-internal-testing/tiny-random-BlipForConditionalGeneration"


def _random_image(seed: int) -> Image.Image:
    rng = np.random.RandomState(seed)
    return Image.fromarray((rng.rand(32, 32, 3) * 255).astype("uint8"))


@pytest.fixture(scope="module")
def adapter():
    return HuggingFaceAdapter(model_id=_TINY_MODEL, device="cpu")


def test_batched_output_matches_unbatched_sample_for_sample(adapter):
    prompts = [
        "describe this image",
        "what is in the picture",
        "caption this",
    ]
    images = [_random_image(i) for i in range(len(prompts))]

    unbatched = [
        adapter.generate(images=[img], prompt=prompt, max_tokens=8)
        for img, prompt in zip(images, prompts)
    ]

    batched = adapter.generate_batch(
        [{"images": [img], "prompt": prompt} for img, prompt in zip(images, prompts)],
        max_tokens=8,
    )

    assert len(batched) == len(unbatched)
    for single, batch in zip(unbatched, batched):
        assert batch.text == single.text
        assert batch.output_tokens == single.output_tokens
        assert batch.input_tokens == single.input_tokens


def test_generate_batch_empty_requests_returns_empty_list(adapter):
    assert adapter.generate_batch([], max_tokens=8) == []
