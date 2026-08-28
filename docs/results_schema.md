# Results JSON schema

Every `vlm-evaluation-harness eval`/`gen-eval --output-dir ...` run writes one
`<benchmark>_<timestamp>_results.json` file. `results_schema_version`
(currently `"1.0"`) is bumped whenever a field is renamed or removed (fields
are only ever added within a version). This document describes version
`1.0`, generated from the actual dict built in
`engine/runner.py::EvalResult.to_dict()`/`.save()` (discriminative runs) and
`engine/generative_runner.py::GenEvalResult.to_dict()`/`.save()` (generative
runs).

## Top-level fields (both eval and gen-eval)

| Field | Type | Notes |
|---|---|---|
| `harness_version` | string | Installed `vlm-evaluation-harness` package version. |
| `started_at` / `finished_at` | ISO 8601 string | UTC. |
| `model` | string | The `--model` spec, e.g. `mock:demo-v1`. |
| `benchmark` | string | Manifest name, e.g. `DemoMC`. |
| `split` | string | Dataset split evaluated. |
| `n_samples` | int | Samples attempted. |
| `metrics` | dict[str, float] | Metric name -> aggregate value. Empty (`{}`) if `--predict-only` was set, or if the split has no scorable ground truth. |
| `metric_n_scored` | dict[str, int] | How many samples each metric actually scored. |
| `metric_ci95` | dict[str, [float, float]] | Bootstrap 95% CI per metric, where per-sample scores exist. |
| `metric_breakdowns` | dict | Per-group breakdowns (e.g. accuracy by subject), where a metric produces one. |
| `provenance` | dict | See below — everything needed to reproduce or invalidate the run. |
| `cost` | dict | `total_usd`, `per_sample_usd` (discriminative also has `total_input_tokens`/`total_output_tokens`). |
| `latency` | dict | `p50_ms`, `p95_ms`, `p99_ms`, `throughput_per_min`. |
| `samples` | list[dict] | Present only when `--log-samples` (the default) is set. See below. |

Discriminative (`eval`) results also carry:

| Field | Type | Notes |
|---|---|---|
| `n_scored` | int | Max `n_scored` across metrics. |
| `n_errors` | int | Samples that errored out. |

## `provenance`

Everything needed to reproduce or invalidate a run — a results file that
doesn't record its decoding params, prompt version, and image preprocessing
can't be meaningfully compared to another run.

| Field | Notes |
|---|---|
| `results_schema_version` | This document's version, e.g. `"1.0"`. |
| `harness_version` | Same as top-level `harness_version`. |
| `harness_sha` | Git commit SHA of the harness install, or `null` if not run from a git checkout. |
| `python` | Python version. |
| `platform` | `platform.platform()`. |
| `model_spec` / `adapter_model_id` | The requested spec and the adapter's resolved model id. |
| `benchmark_version` | The benchmark manifest's own version field. |
| `manifest_hash` | Content hash of the manifest, so a manifest edit is detectable. |
| `scoring` / `task_type` | From the manifest. |
| `decoding.temperature` / `decoding.max_tokens` / `decoding.seed` | Discriminative runs only. |
| `prompt.template` / `.template_b` / `.system` / `.system_override` / `.few_shot_count` / `.answer_extraction` | `system` is the manifest's system prompt unless `--system` overrode it, in which case `system_override` is `true`. Discriminative runs only. |
| `images.max_resolution` / `.min_resolution` / `.color_space` / `.corruptions` / `.corruption_severity` | Discriminative runs only. |
| `cache.enabled` / `.hits` / `.misses` | Response cache stats. Discriminative runs only. |
| `generation.seed` / `.width` / `.height` / `.guidance_scale` / `.num_inference_steps` | Generative runs only. |

## `samples[]` (when `--log-samples`)

Discriminative (`eval`):

| Field | Notes |
|---|---|
| `id` | Sample id. |
| `prediction` | Extracted/parsed answer. |
| `references` | Ground-truth reference(s). |
| `raw_output` | Unparsed model text. |
| `scores` | Per-metric score for this sample, where the metric produces one. |
| `confident` | Whether the model answered rather than deflected (calibration benchmarks). |
| `cached` | Whether this response came from the response cache. |
| `latency_ms` | |
| `image_hashes` | Content hashes of the images sent, for cache/provenance debugging. |
| `metadata` | Sample metadata carried from the dataset. |
| `error` | Error string, or `null`. |

Generative (`gen-eval`):

| Field | Notes |
|---|---|
| `id` / `prompt` / `latency_ms` / `cost_usd` / `seed` / `metadata` | |
| `image_path` | Path to the saved PNG, or `null` (only the first 12 images per run are saved to disk). |
