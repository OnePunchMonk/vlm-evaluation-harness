# VLM-Evaluation-Harness

A unified evaluation framework for Vision Language Models — covering both
**discriminative** tasks (VQA, multiple-choice, hallucination probes: image+text in,
text out, scored against ground truth) and **generative** tasks on text-to-image
models (text in, image out, scored by CLIPScore / VQAScore / compositional checks /
FID / LLM-as-judge). See `idea.md` for the full design document.

Core pieces: a YAML benchmark registry with structural validation at load time,
a scoring engine with a content-addressed response cache, statistically-sound
regression tracking across runs (paired McNemar test + bootstrap confidence
intervals, not fixed thresholds on aggregate deltas), and an HTML/Markdown
report generator — all provider-agnostic and runnable fully offline via the
built-in mock adapters.

## Quick start (fully offline, no API keys)

```bash
pip install -e .

# Discriminative: image -> text, scored against ground truth
vlm-evaluation-harness eval --model mock:demo-v1 --bench demo_mc

# Generative: text -> image, scored by an LLM-judge (no torch needed)
vlm-evaluation-harness gen-eval --model mock:t2i-v1 --bench genjudge_mini

# Every tracked run is appended to ~/.vlm-evaluation-harness/history.jsonl, with
# per-sample scores stored alongside it for paired significance testing
vlm-evaluation-harness history

# Diff the latest tracked runs of two models. When both runs have per-sample
# scores this is a paired McNemar test + bootstrap CI, not a raw threshold.
vlm-evaluation-harness regression --baseline mock:demo-v1 --current mock:demo-v2

# Aggregate saved *_results.json files into one HTML/Markdown report
vlm-evaluation-harness eval --model mock:demo-v1 --bench demo_mc --output-dir results/
vlm-evaluation-harness report --results-dir results/ --output report.html
```

## Real backends

```bash
pip install -e ".[anthropic]"
vlm-evaluation-harness eval --model anthropic:claude-opus-4-6 --bench mmmu --split validation

pip install -e ".[openai]"
vlm-evaluation-harness gen-eval --model openai:gpt-image-1 --bench geneval_mini

pip install -e ".[generative]"   # torch + transformers + diffusers, for
                                  # local T2I inference and CLIP-based metrics
vlm-evaluation-harness gen-eval --model diffusers:stabilityai/stable-diffusion-2-1 --bench geneval_mini
```

Only providers with a real adapter implementation are registered:
`mock`, `anthropic`, `openai`, `huggingface`/`hf`. (An earlier version of this
README also advertised `google`, `vllm`, `ollama`, and `litellm` — those had no
adapter module behind them and were removed rather than fixed with a stub.)

## Benchmarks shipped

12 discriminative benchmarks (VQAv2, TextVQA, ChartQA, DocVQA, MMMU, POPE,
Winoground, ScanQA, and three 2026-research-inspired offline fixtures —
`comp_hardneg`, `hallu_fg`, `calib_deflect` — plus the `demo_mc` smoke test)
and 4 generative text-to-image benchmarks (CLIPScore, VQAScore,
GenEval-style compositional checks, LLM-as-judge). Run
`vlm-evaluation-harness list-benchmarks --verbose` for the live list, or see
[`docs/adding-a-benchmark.md`](docs/adding-a-benchmark.md) for the full
table and field reference.

Every built-in manifest is structurally validated at registry load time —
unresolvable prompt-template placeholders, a scorable split with no
reference field, an unknown metric type, and similar mistakes fail loudly
(`registry.errors()`) instead of silently scoring every sample as 0%.

## What a run actually measures

A metric with nothing to score is `NaN`, never `0.0`. Every result records
its full provenance (decoding params, prompt template, image-pipeline
config, applied corruptions, manifest content hash) and `reproduce
<results.json>` replays a run from that provenance. Responses are cached by
content hash, so a crashed run resumes for free. See
[`docs/reproducibility.md`](docs/reproducibility.md) for the complete list
of reproducibility guarantees, caching, self-consistency sampling, and
log-likelihood scoring.

`vlm-evaluation-harness regression --baseline A --current B` diffs the
latest tracked run of two models with a **paired McNemar test** plus a
bootstrap confidence interval — severity is driven by statistical
significance, not a fixed percentage threshold.

## Learn more

- [`docs/adding-a-benchmark.md`](docs/adding-a-benchmark.md) — manifest
  schema, offline fixtures, log-likelihood scoring.
- [`docs/adding-an-adapter.md`](docs/adding-an-adapter.md) — the
  `VLMAdapter`/`ChoiceScoringAdapter`/`T2IAdapter` protocols, registering a
  new provider.
- [`docs/reproducibility.md`](docs/reproducibility.md) — provenance,
  caching, seeding, self-consistency, corruptions.
- [`docs/results_schema.md`](docs/results_schema.md) — every field in a
  results JSON file.
- [`docs/comparison.md`](docs/comparison.md) — how this differs from
  lmms-eval and VLMEvalKit, and when to reach for which.
- [`idea.md`](idea.md) — the full original design document.
- `src/vlm_evaluation_harness/` — the package itself; module docstrings and
  the dataclasses in `benchmarks/schema.py` and `adapters/base.py` are the
  ground truth for anything the docs above summarize.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src/ tests/
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full contributor workflow.
