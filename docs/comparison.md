# Comparison to lmms-eval and VLMEvalKit

Two established tools cover overlapping ground: [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)
(LMMs-Lab) and [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)
(OpenCompass). Both are mature, widely-used, and support far more models and
benchmarks out of the box than this project does — hundreds of benchmarks
and models between them, image/video/audio modalities, and large
contributor communities. If breadth of pre-integrated benchmarks and models
is what you need today, one of those two is very likely the better choice.

This project is not trying to out-cover them. It exists because of specific
gaps encountered while using them:

- **Statistically-sound regression tracking.** `vlm-evaluation-harness
  regression` diffs two tracked runs with a paired McNemar test over
  per-sample scores plus a bootstrap confidence interval, not a fixed
  percentage threshold on the aggregate delta. A 4-point swing on 50 samples
  is flagged only if it's unlikely to be noise. As far as we're aware,
  neither lmms-eval nor VLMEvalKit ships this kind of paired-significance
  run-to-run comparison out of the box (we haven't audited either codebase
  exhaustively, so treat this as our understanding rather than a verified
  claim about their internals) — in our own prior usage this was typically
  left to whatever spreadsheet or dashboard sits downstream of their result
  JSON/CSV output.
- **Manifests fail loudly, not silently.** Every built-in benchmark manifest
  is structurally validated at registry load time — an unresolvable prompt
  template placeholder, a scorable split with no reference field, or an
  unknown metric type raises an error immediately instead of quietly
  scoring every sample as 0%. This project's own Winoground manifest used
  to ship with exactly that bug (`{caption_0}`/`{caption_1}` placeholders
  the loader never populated), which is what motivated adding the
  validation layer.
- **Response caching + resumability by default.** Every call is cached by a
  content hash of (model, rendered prompt, image hashes, decoding params)
  in a local SQLite file. A run that crashes at sample 800 of 900 resumes
  for free.
- **Provenance-first reproduction.** Every result JSON records decoding
  params, the prompt template, image-pipeline config, applied corruptions,
  and a content hash of the manifest used. `vlm-evaluation-harness reproduce
  <results.json>` replays a run from that recorded provenance rather than
  today's CLI defaults, so a result from three months ago is actually
  reproducible rather than only approximately so.
- **Generative (text-to-image) evaluation as a first-class citizen**,
  scored by CLIPScore, VQAScore, a GenEval-style compositional CLIP check,
  FID, and LLM/VLM-as-judge — using the same manifest/registry/reporting
  machinery as discriminative VQA-style benchmarks, not a separate tool.
- **A small set of 2026-research-inspired offline fixture benchmarks**
  (`comp_hardneg`, `hallu_fg`, `calib_deflect`) that ship as hand-authored,
  zero-network fixtures rather than requiring a dataset download —
  compositional hard-negative reasoning, fine-grained hallucination
  decomposition by object/attribute/relation, and refusal calibration
  (does the model deflect when the image structurally can't answer, rather
  than confidently fabricating).

## What this project does not (yet) do

- No video, audio, or GUI-grounding benchmarks (see the open `W4-vision-depth`
  issues tracking this).
- No batched/multi-GPU local inference — the HuggingFace adapter runs one
  sample at a time today (see the open `W2-backends` issues).
- A fraction of the benchmark and model coverage of either lmms-eval or
  VLMEvalKit. Benchmarks here are added as validated YAML manifests (see
  [`docs/adding-a-benchmark.md`](adding-a-benchmark.md)), which keeps the
  bar for "does this benchmark actually score correctly" high, at the cost
  of a much shorter list than either established tool.

## When to reach for which

- Need to evaluate a new open-weight model against a huge published
  leaderboard surface, fast: lmms-eval or VLMEvalKit.
- Need to know, with statistical confidence, whether your last fine-tune
  regressed anything — and want that answer to survive a crashed run, a
  silently-broken manifest, or someone re-running the eval in three months
  and getting a different number: this project.
