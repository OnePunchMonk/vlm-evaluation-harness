"""Model pricing, loaded from YAML rather than hardcoded per adapter.

Default prices ship in ``pricing.yaml`` next to this module, keyed by
provider then model id, as ``[cost_per_million_input, cost_per_million_output]``
USD. Add a model or correct a price by editing that file -- no code change
or new release needed to update a number that goes stale the moment a
provider changes their price sheet.

To override or extend the defaults without touching the installed package
(e.g. a self-hosted deployment with negotiated pricing), set
``VLM_HARNESS_PRICING_FILE`` to a YAML file in the same shape; its entries
are merged over the built-in defaults, provider by provider.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

_DEFAULT_PRICING_FILE = Path(__file__).parent / "pricing.yaml"


@lru_cache(maxsize=1)
def _load_pricing() -> dict[str, dict[str, list[float]]]:
    data: dict[str, dict[str, list[float]]] = (
        yaml.safe_load(_DEFAULT_PRICING_FILE.read_text()) or {}
    )

    override_path = os.environ.get("VLM_HARNESS_PRICING_FILE")
    if override_path:
        override = yaml.safe_load(Path(override_path).read_text()) or {}
        for provider, models in override.items():
            data.setdefault(provider, {}).update(models)

    return data


def get_pricing(provider: str, model_id: str) -> tuple[float, float]:
    """Return (cost_per_million_input, cost_per_million_output) USD.

    Returns (0.0, 0.0) for a provider/model with no configured entry --
    matching this project's existing convention that an unpriced hosted
    model is treated as free/untracked rather than raising, same as before
    this was config-driven.
    """
    entry = _load_pricing().get(provider, {}).get(model_id)
    if entry is None:
        return (0.0, 0.0)
    return (float(entry[0]), float(entry[1]))


def clear_pricing_cache() -> None:
    """Drop the cached pricing table so a changed env var/file is re-read.

    Mainly for tests that monkeypatch VLM_HARNESS_PRICING_FILE.
    """
    _load_pricing.cache_clear()
