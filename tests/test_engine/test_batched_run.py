"""EvalRunner's batched path (_can_run_batched / _run_batched), wired up for
issues #18/#19: adapter.generate_batch() actually gets used end-to-end, not
just unit-tested in isolation on the adapter.

`FakeBatchAdapter` genuinely reads pixel colors (same trick as
test_self_consistency.py's PixelColorAdapter) so parity with the unbatched
path is a real assertion, not two mocks agreeing with themselves. It can
also be told to fail every Nth chunk, to test that a batch-level error
doesn't take down the whole run.
"""

from __future__ import annotations

from vlm_evaluation_harness.adapters.base import VLMResponse
from vlm_evaluation_harness.adapters.huggingface import BatchGenerateRequest
from vlm_evaluation_harness.engine.runner import EvalConfig, EvalRunner

_COLOR_TO_LETTER = {
    "red": "A",
    "blue": "B",
    "green": "C",
    "yellow": "D",
    "purple": "E",
    "orange": "F",
}
_NAMED_RGB = {
    "red": (220, 40, 40),
    "blue": (40, 90, 220),
    "green": (40, 180, 70),
    "yellow": (230, 200, 30),
    "purple": (140, 60, 190),
    "orange": (240, 130, 30),
}


def _answer_for(images, prompt) -> str:
    mean = tuple(int(x) for x in images[0].resize((1, 1)).getpixel((0, 0))[:3])
    true_color = min(
        _NAMED_RGB, key=lambda c: sum((a - b) ** 2 for a, b in zip(_NAMED_RGB[c], mean))
    )
    return _COLOR_TO_LETTER[true_color]


class FakeBatchAdapter:
    """Implements both generate() and generate_batch(); records how it was
    chunked so tests can assert on batching behavior, and can be told to
    fail a specific call for chunk-isolation tests."""

    def __init__(self, fail_on_call_index: int | None = None):
        self._model_id = "fake-batch"
        self.batch_call_sizes: list[int] = []
        self._fail_on_call_index = fail_on_call_index
        self._call_count = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def supports_multi_image(self) -> bool:
        return True

    @property
    def supports_video(self) -> bool:
        return False

    @property
    def max_resolution(self):
        return None

    @property
    def cost_per_million_input_tokens(self):
        return 0.0

    @property
    def cost_per_million_output_tokens(self):
        return 0.0

    @property
    def supports_batch_inference(self) -> bool:
        return True

    def generate(self, images, prompt, system=None, history=None, max_tokens=1024, temperature=0.0):
        letter = _answer_for(images, prompt)
        return VLMResponse(text=letter, input_tokens=1, output_tokens=1, model_id=self._model_id)

    def generate_batch(self, requests: list[BatchGenerateRequest]) -> list[VLMResponse]:
        self.batch_call_sizes.append(len(requests))
        self._call_count += 1
        if self._fail_on_call_index == self._call_count:
            raise RuntimeError("simulated non-OOM batch failure")
        return [
            VLMResponse(
                text=_answer_for(req.images, req.prompt),
                input_tokens=1,
                output_tokens=1,
                model_id=self._model_id,
            )
            for req in requests
        ]


def _run(adapter, **config_kwargs):
    runner = EvalRunner(adapter)
    config = EvalConfig(
        model_spec="fake:batch",
        benchmark="demo_mc",
        split="validation",
        use_cache=False,
        **config_kwargs,
    )
    return runner.run(config)


def test_batched_path_used_when_batch_size_greater_than_one():
    adapter = FakeBatchAdapter()
    result = _run(adapter, batch_size=4)
    accuracy = next(m for m in result.metrics if m.metric_name == "accuracy")
    assert accuracy.value == 1.0  # every sample answered correctly, real pixel read
    assert adapter.batch_call_sizes  # generate_batch was actually called
    assert max(adapter.batch_call_sizes) <= 4


def test_batch_size_one_never_touches_generate_batch():
    adapter = FakeBatchAdapter()
    _run(adapter, batch_size=1)
    assert adapter.batch_call_sizes == []


def test_batched_and_unbatched_paths_agree():
    batched = _run(FakeBatchAdapter(), batch_size=8)
    unbatched = _run(FakeBatchAdapter(), batch_size=1)
    batched_preds = {s.sample_id: s.prediction for s in batched.sample_results}
    unbatched_preds = {s.sample_id: s.prediction for s in unbatched.sample_results}
    assert batched_preds == unbatched_preds


def test_auto_batch_size_routes_through_batched_path():
    adapter = FakeBatchAdapter()
    _run(adapter, batch_size="auto")
    assert adapter.batch_call_sizes


def test_chunk_failure_is_isolated_not_fatal():
    """A non-OOM error in one generate_batch() call marks only that chunk's
    samples as errored -- it must not abort the whole run, and must not
    silently poison other chunks' results either."""
    adapter = FakeBatchAdapter(fail_on_call_index=1)
    result = _run(adapter, batch_size=4, max_error_rate=1.0)
    errored = [s for s in result.sample_results if s.error]
    ok = [s for s in result.sample_results if not s.error]
    assert 0 < len(errored) <= 4
    assert ok  # later chunks still succeeded
    assert all("simulated non-OOM batch failure" in s.error for s in errored)


def test_falls_back_to_unbatched_for_self_consistency():
    """self_consistency_n > 1 needs per-sample voting the batched path
    doesn't implement, so it must silently use _run_all instead."""
    adapter = FakeBatchAdapter()
    _run(adapter, batch_size=4, self_consistency_n=3, temperature=0.5)
    assert adapter.batch_call_sizes == []


def test_falls_back_to_unbatched_when_adapter_lacks_generate_batch():
    class NoBatchAdapter(FakeBatchAdapter):
        supports_batch_inference = False

    adapter = NoBatchAdapter()
    result = _run(adapter, batch_size=4)
    assert adapter.batch_call_sizes == []
    assert result.sample_results
