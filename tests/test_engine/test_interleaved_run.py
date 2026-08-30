"""End-to-end: image_config.placement == "interleaved" actually reaches the
adapter's generate() call with a `parts` kwarg carrying the right order,
and every other benchmark (placement != "interleaved") is completely
unaffected -- including adapters whose generate() doesn't accept `parts`
at all.
"""

from __future__ import annotations

from vlm_evaluation_harness.adapters.base import VLMResponse
from vlm_evaluation_harness.benchmarks.loader import BenchmarkSample
from vlm_evaluation_harness.benchmarks.schema import BenchmarkManifest
from vlm_evaluation_harness.cache import ResponseCache
from vlm_evaluation_harness.engine.runner import EvalConfig, EvalRunner


def _manifest(placement: str) -> BenchmarkManifest:
    return BenchmarkManifest.from_dict(
        {
            "name": "InterleaveRunTest",
            "source": {"type": "local", "path": "."},
            "splits": [{"name": "validation", "scorable": True}],
            "fields": {"question": "question", "answer": "answer"},
            "prompt_template": "Compare {image_1} to {image_2}. {question}",
            "image_config": {"max_images": 2, "placement": placement},
        }
    )


class _CapturingAdapter:
    """Records every kwarg generate() was called with."""

    def __init__(self):
        self.model_id = "capturing"
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return VLMResponse(text="ok", model_id=self.model_id)


class _NoPartsAdapter:
    """Old-style adapter with no `parts` param -- must never receive one."""

    def __init__(self):
        self.model_id = "no-parts"
        self.calls: list[dict] = []

    def generate(self, images, prompt, system=None, history=None, max_tokens=1024, temperature=0.0):
        self.calls.append({"images": images, "prompt": prompt})
        return VLMResponse(text="ok", model_id=self.model_id)


def _sample() -> BenchmarkSample:
    return BenchmarkSample(
        sample_id="s1",
        images=["a.png", "b.png"],
        text_fields={"question": "Which is bigger?"},
        references=["A"],
        metadata={},
    )


def test_interleaved_manifest_passes_parts_to_adapter():
    adapter = _CapturingAdapter()
    runner = EvalRunner(adapter)
    manifest = _manifest("interleaved")
    config = EvalConfig(model_spec="capturing:x", benchmark="InterleaveRunTest", use_cache=False)
    cache = ResponseCache(None, enabled=False)

    runner._eval_single(
        _sample(),
        manifest,
        config,
        cache,
        few_shot=[],
        images=["a.png", "b.png"],
        image_hashes=["ha", "hb"],
        template=None,
        sample_id="s1",
        references=["A"],
    )

    assert len(adapter.calls) == 1
    parts = adapter.calls[0]["parts"]
    assert [p.kind for p in parts] == ["text", "image", "text", "image", "text"]
    assert parts[1].image == "a.png"
    assert parts[3].image == "b.png"


def test_non_interleaved_manifest_never_sends_parts_even_to_a_legacy_adapter():
    """Regression guard: adapters without a `parts` param in their
    generate() signature must keep working for every benchmark that isn't
    explicitly interleaved -- this is what makes the change backward
    compatible without touching every adapter that exists."""
    adapter = _NoPartsAdapter()
    runner = EvalRunner(adapter)
    manifest = BenchmarkManifest.from_dict(
        {
            "name": "PlainTest",
            "source": {"type": "local", "path": "."},
            "splits": [{"name": "validation", "scorable": True}],
            "fields": {"question": "question", "answer": "answer"},
            "prompt_template": "{question}",
        }
    )
    config = EvalConfig(model_spec="no-parts:x", benchmark="PlainTest", use_cache=False)
    cache = ResponseCache(None, enabled=False)

    result = runner._eval_single(
        BenchmarkSample(
            sample_id="s1", images=[], text_fields={"question": "hi"}, references=["a"], metadata={}
        ),
        manifest,
        config,
        cache,
        few_shot=[],
        images=[],
        image_hashes=[],
        template=None,
        sample_id="s1",
        references=["a"],
    )

    assert result.error is None
    assert len(adapter.calls) == 1
