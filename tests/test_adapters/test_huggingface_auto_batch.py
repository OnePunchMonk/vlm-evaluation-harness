"""generate_auto_batch: progressive halving on OOM until a chunk fits.

Uses the real tiny checkpoint (real generate_batch code path), but forces
OOMs deterministically by monkeypatching the model's .generate() to raise
torch.cuda.OutOfMemoryError whenever it's called with more sequences than
a fake "GPU capacity" -- CPU-only CI has no real CUDA OOM to trigger.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from vlm_harness.adapters.huggingface import HuggingFaceAdapter  # noqa: E402

_TINY_MODEL = "hf-internal-testing/tiny-random-BlipForConditionalGeneration"


def _random_image(seed: int) -> Image.Image:
    rng = np.random.RandomState(seed)
    return Image.fromarray((rng.rand(32, 32, 3) * 255).astype("uint8"))


@pytest.fixture
def adapter():
    return HuggingFaceAdapter(model_id=_TINY_MODEL, device="cpu", batch_size="auto")


def _fail_above(model, capacity: int):
    """Wrap model.generate so any call batched wider than `capacity` OOMs."""
    real_generate = model.generate

    def wrapped(**kwargs):
        batch_width = kwargs["input_ids"].shape[0]
        if batch_width > capacity:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
        return real_generate(**kwargs)

    model.generate = wrapped


def test_starts_at_auto_default(adapter):
    assert adapter._batch_size == 32
    assert adapter._auto_batch_size is True


def test_halves_on_simulated_oom_until_it_fits(adapter):
    _fail_above(adapter._model, capacity=2)

    requests = [{"images": [_random_image(i)], "prompt": "describe this"} for i in range(5)]
    results = adapter.generate_auto_batch(requests, max_tokens=4)

    assert len(results) == 5
    # 32 -> 16 -> 8 -> 4 -> 2 is the first size <= capacity=2.
    assert adapter._batch_size == 2


def test_never_grows_back_after_backing_off(adapter):
    _fail_above(adapter._model, capacity=2)
    requests = [{"images": [_random_image(i)], "prompt": "describe this"} for i in range(3)]
    adapter.generate_auto_batch(requests, max_tokens=4)
    assert adapter._batch_size == 2

    # A second, easier run must not probe back upward.
    more_requests = [{"images": [_random_image(i)], "prompt": "describe this"} for i in range(3)]
    adapter.generate_auto_batch(more_requests, max_tokens=4)
    assert adapter._batch_size == 2


def test_floor_of_one_propagates_oom(adapter):
    _fail_above(adapter._model, capacity=0)
    requests = [{"images": [_random_image(0)], "prompt": "describe this"}]
    with pytest.raises(torch.cuda.OutOfMemoryError):
        adapter.generate_auto_batch(requests, max_tokens=4)
