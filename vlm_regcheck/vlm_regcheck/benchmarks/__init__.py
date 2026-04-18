"""
Benchmark registry — lazy imports so the package can be imported
without torch/transformers/datasets installed.
"""
from __future__ import annotations

from .base import Benchmark, BenchmarkResult, Sample

# Lazy-load benchmark classes to avoid requiring `datasets` at import time
def _get(cls_name: str, module_name: str):
    import importlib
    mod = importlib.import_module(f"vlm_regcheck.benchmarks.{module_name}")
    return getattr(mod, cls_name)


def __getattr__(name: str):
    _map = {
        "MMMUPro":    ("MMMUPro",    "mmmu_pro"),
        "MathVista":  ("MathVista",  "mathvista"),
        "MMStar":     ("MMStar",     "mmstar"),
        "POPE":       ("POPE",       "pope"),
        "Winoground": ("Winoground", "winoground"),
        "MMVP":       ("MMVP",       "mmvp"),
        "CVBench":    ("CVBench",    "cvbench"),
        "OCRBench":   ("OCRBench",   "ocrbench"),
        "BENCHMARK_REGISTRY": ("_registry", None),
        "DEFAULT_SUITE":      ("_default",  None),
        "FAST_SUITE":         ("_fast",     None),
    }
    if name in _map:
        cls_name, mod = _map[name]
        if mod:
            return _get(cls_name, mod)
        # Special: return the pre-built constants (need to materialise them once)
        if name == "BENCHMARK_REGISTRY":
            return _build_registry()
        if name == "DEFAULT_SUITE":
            return _build_default_suite()
        if name == "FAST_SUITE":
            return _build_fast_suite()
    raise AttributeError(f"module 'vlm_regcheck.benchmarks' has no attribute {name!r}")


def _build_registry():
    import importlib
    registry = {}
    _classes = {
        "mmmu_pro": ("MMMUPro", "mmmu_pro"),
        "mathvista": ("MathVista", "mathvista"),
        "mmstar": ("MMStar", "mmstar"),
        "pope": ("POPE", "pope"),
        "winoground": ("Winoground", "winoground"),
        "mmvp": ("MMVP", "mmvp"),
        "cvbench": ("CVBench", "cvbench"),
        "ocrbench": ("OCRBench", "ocrbench"),
    }
    for key, (cls_name, mod) in _classes.items():
        m = importlib.import_module(f"vlm_regcheck.benchmarks.{mod}")
        registry[key] = getattr(m, cls_name)
    return registry


def _build_default_suite():
    r = _build_registry()
    return [(k, r[k]) for k in [
        "mmstar", "mmvp", "mathvista", "pope",
        "cvbench", "ocrbench", "mmmu_pro", "winoground",
    ]]


def _build_fast_suite():
    r = _build_registry()
    return [(k, r[k]) for k in ["mmstar", "pope", "mmvp", "cvbench"]]


def get_benchmark(name: str, **kwargs) -> Benchmark:
    registry = _build_registry()
    if name not in registry:
        raise ValueError(
            f"Unknown benchmark {name!r}. Available: {list(registry)}"
        )
    return registry[name](**kwargs)


__all__ = [
    "Benchmark", "BenchmarkResult", "Sample",
    "get_benchmark",
    "MMMUPro", "MathVista", "MMStar", "POPE",
    "Winoground", "MMVP", "CVBench", "OCRBench",
    "BENCHMARK_REGISTRY", "DEFAULT_SUITE", "FAST_SUITE",
]
