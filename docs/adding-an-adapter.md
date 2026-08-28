# Adding a model backend (adapter)

An adapter is one class implementing the `VLMAdapter` protocol
(`src/vlm_evaluation_harness/adapters/base.py`), registered in
`adapters/registry.py`. `MockAdapter`
(`src/vlm_evaluation_harness/adapters/mock.py`) is the smallest real
implementation and the easiest starting point — it is a normal adapter with
no network I/O, exercising the exact same `generate` / `score_choices`
interface a real backend would.

## Required interface

```python
class VLMAdapter(Protocol):
    def generate(
        self,
        images: list[Image.Image | str],
        prompt: str,
        system: str | None = None,
        history: list[ConversationTurn] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> VLMResponse: ...

    @property
    def model_id(self) -> str: ...
    @property
    def supports_multi_image(self) -> bool: ...
    @property
    def supports_video(self) -> bool: ...
    @property
    def max_resolution(self) -> tuple[int, int] | None: ...
    @property
    def cost_per_million_input_tokens(self) -> float | None: ...
    @property
    def cost_per_million_output_tokens(self) -> float | None: ...
```

`generate()` must return a `VLMResponse(text=..., input_tokens=...,
output_tokens=..., latency_ms=..., model_id=...)`. Token counts feed cost
estimation (`metrics/cost.py`); `latency_ms` feeds the p50/p95/p99 stats in
every report. Set the cost properties to `None` for backends with no
meaningful per-token pricing (local/open-weight models); the mock adapter
reports `0.0` for both since it's free but has a defined price.

## Optional: log-likelihood scoring

Implement `ChoiceScoringAdapter` as well (`supports_choice_scoring: bool`
property plus `score_choices(images, prompt, choices, system=None) ->
ChoiceScores`) if the backend exposes per-token log-probabilities. This lets
manifests set `scoring: loglikelihood`, which is what makes multiple-choice
numbers comparable to published leaderboards instead of depending on an
answer-extraction regex. `huggingface`/`hf` and `mock` implement this today;
hosted-API adapters generally cannot, since most chat completion APIs don't
expose token log-probs for arbitrary continuations.

## Registering the adapter

Add an entry to `_PROVIDERS` in `adapters/registry.py`:

```python
_PROVIDERS: dict[str, str] = {
    ...
    "myprovider": "vlm_evaluation_harness.adapters.myprovider.MyProviderAdapter",
}
```

The registry resolves `"myprovider:some-model-id"` on the CLI by importing
that module lazily and calling `MyProviderAdapter(model_id="some-model-id",
**kwargs)`. If the backend needs an extra dependency, add a
`[project.optional-dependencies]` entry in `pyproject.toml` and an entry in
`_EXTRAS` in `registry.py` so a missing dependency produces
`pip install vlm-evaluation-harness[myprovider]` in the error message
instead of a bare `ImportError`. Do **not** add a provider entry without a
matching adapter module — this repo used to advertise `google`/`vllm`/
`ollama`/`litellm` extras with no adapter behind them, which let
`pip install` succeed and using the model fail; the entries were removed
rather than fixed with a stub.

## Testing a new adapter

`MockAdapter` is registered as `mock:<any-id>` and needs no credentials, so
the fastest way to validate the discriminative and generative pipelines
plumb a new adapter's return values correctly is to write a unit test in
the same style as `tests/test_adapters/test_mock.py` — construct the
adapter directly, call `generate()`/`score_choices()`, and assert on the
returned `VLMResponse`/`ChoiceScores`. For an end-to-end check against a
real benchmark without live API calls, point `demo_mc` at your adapter and
inspect the cache/log output for correctness of the request shape (prompt
text, image encoding, `system`/`history` handling) before spending real API
credits on a full benchmark run.

## Generative (text-to-image) adapters

Text-to-image backends implement `T2IAdapter`
(`adapters/generative/base.py`, text prompt → image) instead of
`VLMAdapter`, registered the same way in
`adapters/generative/registry.py`. See `adapters/generative/mock.py` and
`adapters/generative/diffusers_local.py` for the offline and local-model
reference implementations respectively.
