# VLM-Harness

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
vlm-harness eval --model mock:demo-v1 --bench demo_mc

# Generative: text -> image, scored by an LLM-judge (no torch needed)
vlm-harness gen-eval --model mock:t2i-v1 --bench genjudge_mini

# Every tracked run is appended to ~/.vlm-harness/history.jsonl, with
# per-sample scores stored alongside it for paired significance testing
vlm-harness history

# Diff the latest tracked runs of two models. When both runs have per-sample
# scores this is a paired McNemar test + bootstrap CI, not a raw threshold.
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

Only providers with a real adapter implementation are registered:
`mock`, `anthropic`, `openai`, `huggingface`/`hf`. (An earlier version of this
README also advertised `google`, `vllm`, `ollama`, and `litellm` — those had no
adapter module behind them and were removed rather than fixed with a stub.)

## Benchmarks shipped

**Discriminative** (image(s) + text → text, scored against ground truth):

| Benchmark | Category | Task type | Scoring | Notes |
|---|---|---|---|---|
| `demo_mc` | perception | multiple_choice | generate | Offline fixture — no network/API keys |
| `vqav2` | perception | open_ended | generate | `vqa_accuracy`: official 10-annotator consensus scoring |
| `textvqa` | perception | open_ended | generate | multi-reference `accuracy` + `f1` |
| `chartqa` | reasoning | open_ended | generate | `relaxed_accuracy` (5% numeric tolerance, the official metric) |
| `docvqa` | reasoning | open_ended | generate | `anls` (official metric) |
| `mmmu` | reasoning | multiple_choice | generate | |
| `pope` | safety | yes_no | generate | `pope` metric — reports `yes_rate`, the diagnostic POPE misses if scored as plain accuracy |
| `winoground` | cross_modal | pairwise_matching | generate | `pairwise_group`: both per-image prompts must be correct, per the original protocol |
| `scanqa` | 3d_vision | open_ended | generate | |
| `comp_hardneg` | reasoning | multiple_choice | generate | Offline fixture — ConMe/ARO-style hard negatives (attribute/relation/object swap); `accuracy_by_group` breaks out which swap type a model actually fails |
| `hallu_fg` | safety | yes_no | generate | Offline fixture — `fine_grained_hallucination` decomposes hallucination rate by object/attribute/relation probe, not one POPE-style aggregate |
| `calib_deflect` | safety | open_ended | generate | Offline fixture — `calibration` scores whether the model answers when it can and deflects when the image structurally can't answer, catching confident fabrication that accuracy alone rewards |

**Generative** (text prompt → image, scored on the image):

| Benchmark | Metrics | Notes |
|---|---|---|
| `genjudge_mini` | `llm_judge` | Offline fixture, zero extra dependencies (uses the mock VLM judge) |
| `vqascore_mini` | `vqa_score` | VQAScore: P(yes) under a VLM asked "does this image show X" — correlates with human judgement better than CLIPScore on compositional prompts |
| `geneval_mini` | `geneval_clip`, `clip_score` | Compositional (count/color/shape) prompts; needs `[generative]` |
| `clipscore_mini` | `clip_score`, `llm_judge` | Open-ended prompts; needs `[generative]` |

`comp_hardneg`, `hallu_fg`, and `calib_deflect` are 2026-research-inspired
additions: hard-negative compositional reasoning (ConMe, arXiv:2406.08164;
the axis SCRAMBLe and MultihopSpatial, arXiv:2603.18892, also target),
fine-grained hallucination decomposition (in the spirit of FIHA/FREAK,
arXiv:2603.19765), and refusal calibration (in the spirit of
VLM-DeflectionBench). All three ship as small hand-authored offline fixtures
so they run with zero network access or API keys, same as `demo_mc`/`pope`;
swap in a full dataset by changing `source:` in the manifest once you have
one to point at.

Run `vlm-harness list-benchmarks --verbose` for the live list, or
`vlm-harness validate-bench --bench <name>` to check a manifest.

Every built-in manifest is structurally validated at registry load time:
unresolvable prompt-template placeholders, a scorable split with no
reference field, an unknown metric type, and similar mistakes fail loudly
(`registry.errors()`) instead of silently scoring every sample as 0%. This is
the fix for a real bug this harness used to ship — the old Winoground
manifest's template referenced `{caption_0}`/`{caption_1}`, which the loader
never populated, so every model saw literal unfilled braces and scored 0%
with no error raised anywhere.

## What a run actually measures

A few decisions worth knowing about before you compare numbers across runs:

- **A metric with nothing to score is `NaN`, never `0.0`.** A sample with no
  ground truth is excluded from scoring, not compared against `""`. A
  benchmark that failed to load can no longer look identical to a model that
  got everything wrong.
- **`extraction_failure_rate` is reported on every discriminative run.** A
  jump here is the signature of output-format drift (e.g. the model stopped
  saying "The answer is B" and started saying something the extractor can't
  parse) — which otherwise masquerades as a capability regression.
- **Multi-reference scoring.** `BenchmarkSample.references` is a list; VQAv2's
  ten annotator answers, DocVQA/ScanQA/TextVQA's multiple accepted answers,
  and BLEU/ROUGE/ANLS/F1 all score against the best-matching reference,
  not a single canonical string.
- **Every result records its provenance**: decoding params, prompt template,
  image-pipeline config, applied corruptions, and a content hash of the
  manifest file used. `vlm-harness reproduce <results.json>` replays a run
  from that provenance rather than today's CLI defaults.
- **Responses are cached** by a hash of (model, rendered prompt, image
  hashes, decoding params) in `~/.vlm-harness/cache/responses.sqlite`. A run
  that crashes at sample 800 of 900 resumes for free; re-running the same
  config costs nothing. Disable with `--no-cache`.
- **`--max-concurrent N`** runs samples through a thread pool while keeping
  results in dataset order, so runs stay reproducible regardless of
  completion order.
- **`--corruptions gaussian_blur,jpeg_compression --corruption-severity 2`**
  actually applies the named corruptions before every image reaches the
  model (this flag existed before but was silently ignored).
- **Log-likelihood scoring** (`scoring: loglikelihood` in a manifest) scores
  each multiple-choice option by log-probability instead of generating free
  text and regexing a letter out of it — this is what makes numbers
  comparable to published open-weight leaderboards. Currently implemented by
  the `huggingface`/`hf` and `mock` adapters via `score_choices()`.
- **`--self-consistency N`** (with `--temperature > 0`) samples the model N
  times per question and majority-votes the extracted answers instead of
  trusting one generation (Wang et al. self-consistency). This is a real,
  testable accuracy lever, not just more evaluation surface — see
  `tests/test_engine/test_self_consistency.py`, which shows a synthetic noisy
  adapter recover accuracy under majority-of-5 vs. single-shot on the same
  fixture. `N=1` (default) is exactly today's single-call behavior. Each vote
  gets a distinct cache key, so a resumed run doesn't replay the same cached
  call N times:
  ```bash
  vlm-harness eval --model anthropic:claude-opus-4-6 --bench comp_hardneg \
    --temperature 0.7 --self-consistency 5
  ```

## Regression tracking

`vlm-harness regression --baseline A --current B` diffs the latest tracked
run of each model per benchmark. When both runs recorded per-sample scores
(the default since 0.2.0), each metric is compared with a **paired McNemar
test** — did the same samples flip from right to wrong — plus a bootstrap
confidence interval on the paired delta, and severity is driven by
statistical significance rather than a fixed percentage. A 4-point swing on
50 samples is not automatically "MEDIUM"; it's flagged only if it's unlikely
to be noise. Runs recorded before per-sample tracking existed fall back to
the previous magnitude-threshold behavior, and the CLI says so explicitly.

## Architecture

- **Discriminative pipeline**: `benchmarks/registry.py` (YAML manifests,
  validated at load) → `benchmarks/loader.py` (HF datasets or local JSONL) →
  `prompt/formatter.py` (fails loudly on unresolvable placeholders) →
  `adapters/*` (`VLMAdapter` protocol: images+text → text, or
  `ChoiceScoringAdapter` for loglikelihood scoring) → `parsing/extractor.py`
  → `metrics/*` → `engine/runner.py`.
- **Generative pipeline**: same registry/loader, but `adapters/generative/*`
  (`T2IAdapter` protocol: text → image) and `metrics/generative/*`
  (CLIPScore, VQAScore, GenEval-style CLIP zero-shot checks, FID,
  LLM/VLM-as-judge), orchestrated by `engine/generative_runner.py`.
- **Statistics**: `stats.py` — paired McNemar test, percentile bootstrap CIs
  (both plain and paired-delta), and a Wilson interval for proportions.
  Dependency-free (numpy only).
- **Caching & retries**: `cache.py` (SQLite response cache, keyed by content
  hash) and `retry.py` (exponential backoff with jitter for transient
  provider errors — rate limits, 5xx, timeouts).
- **Tracking**: `tracking/history.py` appends every `eval`/`gen-eval` run's
  aggregate metrics to a local JSON-lines file, and its per-sample scores to
  a companion file keyed by run id. `tracking/regression.py` diffs two
  tracked runs using `stats.py`.
- **Reporting**: `reporting/terminal.py`, `reporting/html.py`,
  `reporting/markdown.py` all work off the same plain-dict result shape, so a
  report can mix discriminative and generative runs in one leaderboard.

Adding a benchmark is a YAML file (see `src/vlm_harness/benchmarks/manifests/`);
adding a model backend is one adapter class implementing `VLMAdapter` (and
optionally `ChoiceScoringAdapter`) or `T2IAdapter`.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src/ tests/
```

`vlm_regcheck/` is a separate, more specialized prototype for base-vs-finetuned
regression checks on local HuggingFace VLMs (own CLI, own benchmark set); it
predates and is independent of the tracking system described above.
