"""Global RNG seeding for reproducible runs.

Seeds Python's `random` and `numpy.random` unconditionally, and `torch` only
if it happens to be importable (the huggingface/generative extras pull it in;
the core package does not depend on it).
"""

from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
