"""Array-library-agnostic helpers shared by method implementations.

Methods accept either numpy arrays or torch tensors for hidden states (torch
is an optional dependency of this package, same as it is for the HF
adapter). These helpers dispatch on type rather than importing torch
unconditionally.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def is_torch_tensor(x: Any) -> bool:
    try:
        import torch
    except ImportError:
        return False
    return isinstance(x, torch.Tensor)


def gather_tokens(hidden_states: Any, indices: np.ndarray) -> Any:
    """Select tokens along the sequence dim: (B, N, D)[indices] -> (B, K, D).

    `indices` is a numpy int array of shape (B, K), one row of kept-token
    positions per batch element.
    """
    if is_torch_tensor(hidden_states):
        import torch

        idx = torch.as_tensor(indices, device=hidden_states.device, dtype=torch.long)
        idx = idx.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
        return torch.gather(hidden_states, dim=1, index=idx)

    hidden_states = np.asarray(hidden_states)
    batch_size = hidden_states.shape[0]
    return np.stack([hidden_states[b, indices[b]] for b in range(batch_size)], axis=0)


def to_numpy(x: Any) -> np.ndarray:
    if is_torch_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)
