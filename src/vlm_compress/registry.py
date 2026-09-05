"""Method registry: name -> TokenCompressor class, and the `create_compressor` factory."""

from __future__ import annotations

from vlm_compress.base import TokenCompressor

_REGISTRY: dict[str, type[TokenCompressor]] = {}


def register_method(name: str):
    """Class decorator that adds a `TokenCompressor` subclass to the registry."""

    def _decorator(cls: type[TokenCompressor]) -> type[TokenCompressor]:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"compression method {name!r} is already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return _decorator


def list_methods() -> list[str]:
    """Names of all registered compression methods, sorted."""
    _ensure_builtins_loaded()
    return sorted(_REGISTRY)


def create_compressor(
    method: str,
    target_ratio: float,
    model_family: str | None = None,
    config: dict | None = None,
) -> TokenCompressor:
    """Construct a registered compressor by name.

    `model_family` (e.g. "qwen2.5-vl", "llava-1.5") is accepted for forward
    compatibility with methods that need model-specific defaults (dynamic-res
    tiling, projector dims); today it is passed through into `config` under
    `model_family` rather than changing behavior.
    """
    _ensure_builtins_loaded()
    try:
        cls = _REGISTRY[method]
    except KeyError:
        available = ", ".join(list_methods())
        raise ValueError(f"unknown compression method {method!r}; available: {available}") from None

    merged_config = dict(config or {})
    if model_family is not None:
        merged_config.setdefault("model_family", model_family)
    return cls(target_ratio=target_ratio, config=merged_config)


_builtins_loaded = False


def _ensure_builtins_loaded() -> None:
    """Import the built-in method modules so their `@register_method` decorators run.

    Deferred (rather than at package import time) to avoid import cycles
    between `vlm_compress.methods.*` and `vlm_compress.registry`.
    """
    global _builtins_loaded
    if _builtins_loaded:
        return
    from vlm_compress.methods import (  # noqa: F401
        aim,
        dart,
        fastv,
        glimpseprune,
        learnpruner,
        prumerge,
        queryaware,
        random,
        topv,
        viscache,
        vispruner,
    )

    _builtins_loaded = True
