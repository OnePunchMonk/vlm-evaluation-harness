"""
vlm-regcheck: VLM post-training regression evaluation suite.

Quick start:

    from vlm_regcheck import RegressionEvaluator, EvalConfig

    report = RegressionEvaluator(EvalConfig(
        base_model_id="llava-hf/llava-1.5-7b-hf",
        finetuned_model_id="./my-finetuned-vlm",
        benchmarks=["mmstar", "pope", "mmvp", "cvbench"],
        n_samples=200,
    )).run()

    report.print_summary()
    report.to_json("report.json")
    report.to_html("report.html")
"""

def __getattr__(name):
    """Lazy imports — avoids requiring torch/transformers/datasets at import time."""
    _lazy = {
        "EvalConfig":          ("vlm_regcheck.evaluator",  "EvalConfig"),
        "RegressionEvaluator": ("vlm_regcheck.evaluator",  "RegressionEvaluator"),
        "RegressionReport":    ("vlm_regcheck.report",     "RegressionReport"),
        "BenchmarkDelta":      ("vlm_regcheck.report",     "BenchmarkDelta"),
        "VLMWrapper":          ("vlm_regcheck.models",     "VLMWrapper"),
        "Benchmark":           ("vlm_regcheck.benchmarks", "Benchmark"),
        "BenchmarkResult":     ("vlm_regcheck.benchmarks", "BenchmarkResult"),
        "Sample":              ("vlm_regcheck.benchmarks", "Sample"),
        "MMMUPro":             ("vlm_regcheck.benchmarks", "MMMUPro"),
        "MathVista":           ("vlm_regcheck.benchmarks", "MathVista"),
        "MMStar":              ("vlm_regcheck.benchmarks", "MMStar"),
        "POPE":                ("vlm_regcheck.benchmarks", "POPE"),
        "Winoground":          ("vlm_regcheck.benchmarks", "Winoground"),
        "MMVP":                ("vlm_regcheck.benchmarks", "MMVP"),
        "CVBench":             ("vlm_regcheck.benchmarks", "CVBench"),
        "OCRBench":            ("vlm_regcheck.benchmarks", "OCRBench"),
        "BENCHMARK_REGISTRY":  ("vlm_regcheck.benchmarks", "BENCHMARK_REGISTRY"),
        "DEFAULT_SUITE":       ("vlm_regcheck.benchmarks", "DEFAULT_SUITE"),
        "FAST_SUITE":          ("vlm_regcheck.benchmarks", "FAST_SUITE"),
    }
    if name in _lazy:
        import importlib
        mod = importlib.import_module(_lazy[name][0])
        return getattr(mod, _lazy[name][1])
    raise AttributeError(f"module 'vlm_regcheck' has no attribute {name!r}")

__version__ = "0.1.0"

__all__ = [
    "EvalConfig", "RegressionEvaluator",
    "RegressionReport", "BenchmarkDelta",
    "VLMWrapper",
    "Benchmark", "BenchmarkResult", "Sample",
    "MMMUPro", "MathVista", "MMStar", "POPE",
    "Winoground", "MMVP", "CVBench", "OCRBench",
    "BENCHMARK_REGISTRY", "DEFAULT_SUITE", "FAST_SUITE",
]
