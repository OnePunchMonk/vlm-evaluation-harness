# Adding a benchmark

A benchmark is one YAML manifest under `src/vlm_evaluation_harness/benchmarks/manifests/`.
There is no code to write for a standard multiple-choice, open-ended, yes/no,
or pairwise benchmark — the loader, prompt formatter, extractor, and metric
dispatcher are all driven off the manifest.

## Minimal example

`demo_mc.yaml` is the smallest real manifest in the repo and is a good
starting template:

```yaml
name: DemoMC
version: "1.0"
description: >
  Fully offline multiple-choice smoke-test benchmark: identify the color of
  a rendered square.
taxonomy_category: perception
modality: 2d

source:
  type: local
  path: ../fixtures/demo_mc

splits:
  - name: validation
    scorable: true

task_type: multiple_choice

fields:
  question: question
  choices: choices
  answer: answer
  images:
    - image
  subject: category

image_config:
  max_images: 1
  placement: before_text

prompt_template: |
  {question}

  Options:
  {formatted_choices}

  Answer with the option letter only.

answer_extraction:
  strategy: first_letter
  normalize: uppercase

metrics:
  - type: accuracy
  - type: accuracy_by_group
    group_field: subject

few_shot:
  count: 0
```

## Field reference

See `src/vlm_evaluation_harness/benchmarks/schema.py` for the authoritative
dataclasses — this is a summary, not a copy, and the schema is what actually
validates a manifest at load time.

- **`source`** — `type: huggingface` (with `path`, optional `revision`,
  `subset`) or `type: local` (a directory of a JSONL split file plus an
  `images/` folder, resolved relative to the manifest).
- **`splits`** — one or more named splits; `scorable: false` marks a split
  (e.g. a public test set with hidden labels) that should run inference but
  not be scored.
- **`task_type`** — one of `open_ended`, `multiple_choice`, `yes_no`,
  `captioning`, `pairwise_matching`, `text_to_image`.
- **`fields`** — maps manifest-level names (`question`, `answer`, `answers`,
  `choices`, `images`, `subject`, `difficulty`, `context`) to the actual
  dataset column names. `answers` (plural) is for datasets that carry a list
  of acceptable reference answers per sample (VQAv2-style); every value
  found there is merged into `BenchmarkSample.references`.
- **`prompt_template`** — a Python `.format()`-style string. Every
  placeholder must resolve against a `fields` entry or the manifest fails
  registry validation at load time — an unresolvable placeholder is a hard
  error, not a value silently left as literal `{braces}` in the prompt.
- **`answer_extraction`** — strategy the answer parser (`parsing/extractor.py`)
  uses to pull a scorable prediction out of free-text generation. Not used
  when `scoring: loglikelihood` (see below).
- **`metrics`** — one or more entries from `KNOWN_METRIC_TYPES` /
  `KNOWN_GENERATIVE_METRIC_TYPES` in `schema.py`. `accuracy_by_group` takes a
  `group_field` naming which `fields` entry to break the score out by.
- **`few_shot.count`** — number of in-context examples to prepend, drawn
  from a split other than the one being evaluated.

## Log-likelihood scoring

Set `scoring: loglikelihood` (default is `generate`) on a `multiple_choice`
manifest to score each option by log-probability instead of generating free
text and parsing it out — this is what makes numbers comparable to published
leaderboards for adapters that implement `ChoiceScoringAdapter`
(`huggingface`/`hf` and `mock` currently). No `answer_extraction` block is
needed in this mode.

## Offline fixtures

If you don't have a public dataset to point `source:` at yet, ship a small
hand-authored fixture instead (see `comp_hardneg`, `hallu_fg`, and
`calib_deflect` for examples, and `scripts/gen_fixtures.py` for how their
images/JSONL are generated). `source: {type: local, path: ...}` works
identically whether the directory is a real dataset export or an
eight-sample synthetic fixture — swap in a full dataset later by changing
only `source:`.

## Validating a new manifest

```bash
vlm-evaluation-harness validate-bench --bench <name>
vlm-evaluation-harness eval --model mock:demo-v1 --bench <name>
```

The first checks the manifest is structurally valid (resolvable
placeholders, a scorable split with a reference field, a known metric type,
etc. — see `registry.errors()`). The second actually runs it end to end
against the offline mock adapter, which is the fastest way to catch a wrong
field mapping before wiring up a real dataset or model.

To develop a manifest outside the package source tree entirely, use
`--include-path <dir>` on `list-benchmarks`, `eval`, `gen-eval`, and
`validate-bench` to point at an external directory of manifests without
copying it into `benchmarks/manifests/`.
