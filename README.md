# VLM-Harness

A unified evaluation framework for Vision Language Models — covering both
**discriminative** tasks (VQA, multiple-choice, hallucination probes: image+text in,
text out, scored against ground truth) and **generative** tasks on text-to-image
models (text in, image out, scored by CLIPScore / compositional checks / FID /
LLM-as-judge). See `idea.md` for the full design document.

Core pieces: a YAML benchmark registry, a scoring engine, local regression
tracking across runs, and an HTML/Markdown report generator — all provider-agnostic
and runnable fully offline via the built-in mock adapters.

## Quick start (fully offline, no API keys)

```bash
pip install -e .

# Discriminative: image -> text, scored against ground truth
vlm-harness eval --model mock:demo-v1 --bench demo_mc

# Generative: text -> image, scored by an LLM-judge (no torch needed)
vlm-harness gen-eval --model mock:t2i-v1 --bench genjudge_mini

# Every tracked run is appended to ~/.vlm-harness/history.jsonl
vlm-harness history

# Diff the latest tracked runs of two models and flag regressions
vlm-harness regression --baseline mock:demo-v1 --current mock:demo-v2

# Aggregate saved *_results.json files into one HTML/Markdown report
vlm-harness eval --model mock:demo-v1 --bench demo_mc --output-dir results/
vlm-harness report --results-dir results/ --output report.html
```

## Real backends

```bash
pip install -e ".[anthropic]"
vlm-harness eval --model anthropic:claude-opus-4-6 --bench mmmu --split validation

pip install -e ".[openai]"
vlm-harness gen-eval --model openai:gpt-image-1 --bench geneval_mini

pip install -e ".[generative]"   # torch + transformers + diffusers, for
                                  # local T2I inference and CLIP-based metrics
vlm-harness gen-eval --model diffusers:stabilityai/stable-diffusion-2-1 --bench geneval_mini
```

## Benchmarks shipped

**Discriminative** (image(s) + text → text, scored against ground truth):

| Benchmark | Category | Task type | Notes |
|---|---|---|---|
| `demo_mc` | perception | multiple_choice | Offline fixture — no network/API keys |
| `vqav2`, `textvqa` | perception | open_ended | |
| `chartqa`, `docvqa`, `mmmu` | reasoning | open_ended / multiple_choice | |
| `pope` | safety | yes_no | Object hallucination probing |
| `winoground` | cross_modal | multiple_choice | |
| `scanqa` | 3d_vision | open_ended | |

**Generative** (text prompt → image, scored on the image):

| Benchmark | Metrics | Notes |
|---|---|---|
| `genjudge_mini` | `llm_judge` | Offline fixture, zero extra dependencies (uses the mock VLM judge) |
| `geneval_mini` | `geneval_clip`, `clip_score` | Compositional (count/color/shape) prompts; needs `[generative]` |
| `clipscore_mini` | `clip_score`, `llm_judge` | Open-ended prompts; needs `[generative]` |

Run `vlm-harness list-benchmarks --verbose` for the live list, or
`vlm-harness validate-bench --bench <name>` to check a manifest.

## Architecture

- **Discriminative pipeline**: `benchmarks/registry.py` (YAML manifests) →
  `benchmarks/loader.py` (HF datasets or local JSONL) → `prompt/formatter.py` →
  `adapters/*` (`VLMAdapter` protocol: images+text → text) →
  `parsing/extractor.py` → `metrics/*` → `engine/runner.py`.
- **Generative pipeline**: same registry/loader, but `adapters/generative/*`
  (`T2IAdapter` protocol: text → image) and `metrics/generative/*`
  (CLIPScore, GenEval-style CLIP zero-shot checks, FID, LLM/VLM-as-judge),
  orchestrated by `engine/generative_runner.py`.
- **Tracking**: `tracking/history.py` appends every `eval`/`gen-eval` run
  (model, benchmark, metrics) to a local JSON-lines file. `tracking/regression.py`
  diffs two tracked runs and classifies drops by severity
  (CRITICAL/HIGH/MEDIUM/LOW), independent of whether the runs were
  discriminative or generative.
- **Reporting**: `reporting/terminal.py`, `reporting/html.py`,
  `reporting/markdown.py` all work off the same plain-dict result shape, so a
  report can mix discriminative and generative runs in one leaderboard.

Adding a benchmark is a YAML file (see `src/vlm_harness/benchmarks/manifests/`);
adding a model backend is one adapter class implementing `VLMAdapter` or
`T2IAdapter`.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src/ tests/
```

`vlm_regcheck/` is a separate, more specialized prototype for base-vs-finetuned
regression checks on local HuggingFace VLMs (own CLI, own benchmark set); it
predates and is independent of the tracking system described above.
