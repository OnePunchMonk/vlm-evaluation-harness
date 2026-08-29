"""Tests for global RNG seeding (seeding.py, issue #13).

seed_everything() is wired into engine/runner.py and engine/generative_runner.py
and its seed value round-trips through the CLI's provenance JSON (verified
manually), but until now nothing asserted that random.seed/np.random.seed/
torch.manual_seed are actually *called* with the right value -- and torch
isn't installed in this environment, so that branch had never executed even
once. These tests patch the underlying calls directly (and fake the `torch`
module via sys.modules when it isn't installed) so the seeding logic itself
is verified regardless of whether torch happens to be present.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np

from vlm_evaluation_harness.seeding import seed_everything


def test_none_seed_is_a_no_op():
    with patch("random.seed") as random_seed, patch.object(np.random, "seed") as np_seed:
        seed_everything(None)
    random_seed.assert_not_called()
    np_seed.assert_not_called()


def test_seeds_python_random_and_numpy():
    with patch("random.seed") as random_seed, patch.object(np.random, "seed") as np_seed:
        seed_everything(1234)
    random_seed.assert_called_once_with(1234)
    np_seed.assert_called_once_with(1234)


def test_missing_torch_does_not_raise():
    # In this environment torch genuinely isn't installed, so this exercises
    # the real ImportError path, not a simulated one.
    if "torch" in sys.modules:
        import pytest

        pytest.skip("torch is installed in this environment; see the faked-torch test instead")
    seed_everything(7)  # must not raise


def test_seeds_torch_when_importable():
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True

    with patch.dict(sys.modules, {"torch": fake_torch}):
        seed_everything(99)

    fake_torch.manual_seed.assert_called_once_with(99)
    fake_torch.cuda.is_available.assert_called_once()
    fake_torch.cuda.manual_seed_all.assert_called_once_with(99)


def test_skips_cuda_seeding_when_no_gpu_available():
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False

    with patch.dict(sys.modules, {"torch": fake_torch}):
        seed_everything(5)

    fake_torch.manual_seed.assert_called_once_with(5)
    fake_torch.cuda.manual_seed_all.assert_not_called()
