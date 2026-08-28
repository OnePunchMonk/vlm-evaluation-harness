# VLM-Evaluation-Harness: A Unified Evaluation Framework for Vision Language Models

## 1. Problem Statement

The Vision Language Model (VLM) ecosystem is growing rapidly. Models like GPT-4o, Claude, Gemini, LLaVA, Qwen-VL, InternVL, and Phi-Vision ship on a near-weekly cadence. Yet evaluating them remains painful:

- **Fragmented benchmarks.** MMMU, MMBench, SEED-Bench, VQAv2, ChartQA, DocVQA, TextVQA, AI2D, MathVista, and dozens more each define their own data formats, prompt conventions, and scoring logic. Researchers rewrite hundreds of lines of boilerplate for every new benchmark they want to run.

- **Inconsistent methodology.** Different papers evaluate the same benchmark with different prompt templates, image resolutions, decoding strategies, and post-processing rules. This makes cross-model comparisons in the literature unreliable --- a 2-point accuracy delta can easily be an artifact of prompting rather than capability.

- **No standard model interface.** Some models are API-only (OpenAI, Anthropic, Google). Others run locally via HuggingFace Transformers, vLLM, or Ollama. Existing evaluation toolkits tend to be tightly coupled to one backend, forcing researchers to maintain adapter code themselves.

- **Hallucination is under-measured.** The most consequential failure mode of VLMs --- confidently describing objects, text, or spatial relationships that do not exist in the image --- has no standard, widely-adopted evaluation protocol. Most benchmarks measure what models get right; few systematically measure what models make up.

- **Safety and robustness are afterthoughts.** Bias in visual descriptions, susceptibility to adversarial visual prompts, and calibration of refusal behavior are rarely tested alongside capability benchmarks.

VLM-Evaluation-Harness is a framework designed to solve these problems.

---

## 2. Vision

**VLM-Evaluation-Harness is the pytest of VLM evaluation** --- minimal, composable, and extensible. It provides:

1. A universal model adapter interface that works with any VLM, whether it's a cloud API or a local checkpoint.
2. A declarative benchmark registry where adding a new benchmark is a YAML file, not a Python module.
3. A scoring engine with first-class support for hallucination detection, safety measurement, and cost/latency tracking alongside traditional accuracy metrics.
4. A CLI and reporting layer that makes single-model evaluation, head-to-head comparison, and leaderboard generation trivially easy.

The guiding design principle is **separation of concerns**: the benchmark definition, the prompt formatting, the model invocation, the response parsing, and the metric computation are all independent, pluggable components. This makes it possible to evaluate any model on any benchmark with a single config change.

---

## 3. Task Taxonomy

VLM capabilities span a wide range. The harness organizes benchmarks into a structured taxonomy so that evaluation results are interpretable at both the individual-benchmark and capability-cluster level.

### 3.1 Perception

Low-level visual understanding tasks that test whether the model can accurately perceive what is in an image.

| Sub-category | Description | Example Benchmarks |
|---|---|---|
| **OCR** | Recognizing and extracting text from images, including handwritten, rotated, and stylized text | TextVQA, OCRBench, SROIE |
| **Object Recognition** | Identifying and classifying objects present in a scene | COCO-val, OpenImages |
| **Spatial Understanding** | Reasoning about relative positions, sizes, and arrangements of objects | VSR, What'sUp, SpatialBench |
| **Counting** | Accurately counting instances of objects in a scene | CLEVR-Count, TallyQA |
| **Attribute Recognition** | Identifying colors, materials, textures, and states of objects | VAW, PACO |
| **Fine-Grained Recognition** | Distinguishing between visually similar sub-categories (bird species, car models) | CUB-200, Stanford Cars |

### 3.2 Reasoning

Higher-order inference tasks that require the model to go beyond surface-level perception.

| Sub-category | Description | Example Benchmarks |
|---|---|---|
| **Chart & Graph QA** | Extracting data and answering questions about charts, plots, and infographics | ChartQA, PlotQA, FigureQA |
| **Document Understanding** | Comprehending structured and semi-structured documents (forms, receipts, papers) | DocVQA, InfographicVQA, TabFact |
| **Diagram Reasoning** | Understanding scientific diagrams, flowcharts, and technical illustrations | AI2D, ScienceQA |
| **Mathematical Visual Reasoning** | Solving geometry problems, reading equations, and visual word problems | MathVista, GeoQA, MATH-Vision |
| **Multi-Image Reasoning** | Reasoning across two or more images simultaneously (differences, sequences, composites) | NLVR2, Spot-the-Diff, Mantis |
| **Temporal / Sequential** | Understanding before/after, cause/effect, or procedural sequences from image sets | STAR, Perception Test |
| **Commonsense Inference** | Applying world knowledge to visual scenes (what will happen next, what is unusual) | VCR, VisualCOMET |

### 3.3 Knowledge

Tasks that require grounding visual content in external or domain-specific knowledge.

| Sub-category | Description | Example Benchmarks |
|---|---|---|
| **Academic / Exam** | Multi-subject exam questions that combine images with domain knowledge | MMMU, ScienceQA, EXAMS-V |
| **Landmark & Place Recognition** | Identifying real-world locations, buildings, and landmarks | Google Landmarks, GeoGuessr-style |
| **Cultural Context** | Understanding culturally-specific visual content (symbols, customs, art styles) | MaRVL, CVQA |
| **Domain-Specific** | Medical imaging, satellite imagery, industrial inspection, etc. | PathVQA, RSVQA, SLAKE |

### 3.4 Generation Quality

Tasks where the output is open-ended and quality matters beyond simple correctness.

| Sub-category | Description | Example Benchmarks |
|---|---|---|
| **Image Captioning** | Generating accurate, detailed, and fluent descriptions of images | COCO Captions, NoCaps, Flickr30k |
| **Grounded QA** | Answering questions with evidence traceable to specific image regions | GQA, Grounding-DINO eval |
| **Long-Form Description** | Producing rich, paragraph-length descriptions of complex scenes | DetailCaps, ShareGPT4V-eval |
| **Creative / Narrative** | Visual storytelling, humor understanding, meme interpretation | MemeCap, VIST |
| **Instruction Following** | Following complex visual instructions (e.g., "describe only the background") | LLaVA-Bench, MM-Vet |

### 3.5 Safety & Robustness

Tasks that measure failure modes, biases, and adversarial vulnerability.

| Sub-category | Description | Example Benchmarks |
|---|---|---|
| **Object Hallucination** | Detecting when models describe objects not present in the image | POPE, CHAIR, HallusionBench |
| **Attribute Hallucination** | Detecting when models fabricate colors, counts, text, or spatial relationships | Custom probes, AMBER |
| **Demographic Bias** | Measuring disparities in description quality or assumptions across demographic groups | FairFace-eval, VisoGender |
| **Adversarial Visual Prompts** | Resistance to typographic attacks, visual prompt injection, and manipulated images | Custom adversarial suites |
| **Refusal Calibration** | Appropriateness of refusals --- does the model refuse when it should, and not when it shouldn't? | MM-SafetyBench, VLSafe |
| **Robustness to Corruption** | Performance stability under blur, compression, noise, rotation, and other image degradations | ImageNet-C adapted for VLMs |

### 3.6 3D Vision & Spatial Intelligence

A critical gap in most VLM evaluation frameworks is the complete absence of 3D understanding tasks. Real-world visual intelligence is fundamentally three-dimensional --- humans effortlessly infer depth, volume, occlusion relationships, and viewpoint transformations from 2D projections. VLMs must be tested on these capabilities.

| Sub-category | Description | Example Benchmarks |
|---|---|---|
| **Depth Estimation** | Monocular depth reasoning --- "which object is closer?", relative depth ordering from a single image | NYU Depth v2, KITTI Depth, SUN RGB-D |
| **3D Spatial Reasoning** | Reasoning about 3D relationships from 2D images --- "could you reach X from Y?", "what's behind the sofa?" | ScanQA, SQA3D, 3DMV-VQA |
| **Novel View Synthesis Understanding** | Given one view, reasoning about what another viewpoint would reveal --- testing mental rotation and 3D reconstruction | Objaverse-QA, MVBench-3D |
| **Volume & Scale Estimation** | Estimating real-world sizes, distances, and volumes from visual cues | Custom probes, CLEVR-3D |
| **Occlusion Reasoning** | Understanding what is hidden behind visible objects, predicting occluded geometry | CATER, Occlusion-QA |
| **Scene Geometry** | Understanding room layouts, floor plans from photos, architectural spatial reasoning | Structured3D, ScanNet-QA |
| **Point Cloud / Mesh Comprehension** | For models accepting 3D inputs --- understanding spatial structure from point clouds or rendered meshes | ShapeNet-QA, ModelNet-QA |
| **Multi-View Consistency** | Given multiple views of the same scene, answering questions that require fusing 3D information across viewpoints | Multi-view ScanQA, CO3D-QA |

**Why this matters:** Many high-value VLM applications --- robotics, autonomous driving, AR/VR, architectural design, medical imaging (CT/MRI slices) --- require 3D understanding. A harness that only tests 2D perception gives an incomplete and potentially misleading picture of model capability. A model that aces VQAv2 but cannot reason about depth or occlusion is unsuitable for embodied applications.

**Implementation notes:** 3D benchmarks introduce unique data requirements. Some provide point clouds or depth maps alongside RGB images; others provide multiple views. The benchmark manifest schema must support these additional modalities:

```yaml
# Example: 3D benchmark manifest extension
modality: 3d_scene
fields:
  images:
    - rgb_image
  depth_maps:
    - depth_image           # Optional depth map input
  point_clouds:
    - scene_pointcloud      # Optional point cloud input
  camera_params:
    - camera_intrinsics     # For multi-view tasks
    - camera_extrinsics
  multi_view:
    views: [view_1, view_2, view_3]
    strategy: all            # "all", "random_pair", "sequential"
```

### 3.7 Cross-Modal Reasoning

Most existing VLM benchmarks follow a simple pattern: image in, text question in, text answer out. This treats the visual modality as *context* for a language task rather than testing true multimodal integration. Cross-modal reasoning tasks require simultaneous, interleaved understanding of both modalities where neither alone is sufficient to arrive at the answer.

| Sub-category | Description | Example Benchmarks |
|---|---|---|
| **Visual Entailment** | Does the image support, contradict, or remain neutral to a text claim? Requires jointly parsing visual evidence and linguistic assertion | SNLI-VE, e-ViL |
| **Interleaved Document Reasoning** | Answering questions about documents mixing text, figures, tables, and equations where the answer requires synthesizing across modalities | ArXivQA, Docmatix, SlideVQA |
| **Figure-Reference Resolution** | Given text that says "as shown in Figure 3", correctly identifying and reasoning about the referenced figure | Custom probes, Paper-QA |
| **Cross-Modal Retrieval Verification** | Given a text description and multiple images (or vice versa), identifying which image matches and explaining why | Winoground, VL-CheckList |
| **Diagram-to-Structured-Output** | Understanding a visual representation (flowchart, circuit diagram, UML) and producing structured output (code, graph, state machine) that proves joint comprehension | Flow2Code, UML-QA, Circuit-QA |
| **Audio-Visual Reasoning** | For models supporting audio+vision --- answering questions requiring both modalities (e.g., "what instrument is the person in blue playing?") | MUSIC-AVQA, Pano-AVQA |
| **Text-in-Image + External Text** | Questions where the answer requires combining text visible in the image with text in the question or context | InfoVQA, ScreenQA |
| **Visual Grounding with Linguistic Ambiguity** | Resolving linguistically ambiguous references using visual context ("the bat" --- animal or sports equipment?) | Winoground, ARO |
| **Multimodal Chain-of-Thought** | Tasks requiring interleaved visual and textual reasoning steps, where intermediate conclusions from one modality feed into reasoning in the other | ScienceQA (with CoT), M-CoT |
| **Contradictory Modality Detection** | Identifying when visual content contradicts textual content (e.g., a caption that misrepresents the image) | FOIL, Caption-Contradiction |

**Why this matters:** The goal of VLMs is not to be a "language model that can also see" but a model that truly fuses information across modalities. A model that processes an image and a question independently --- encoding the image into a description and then answering the question from that description --- will score well on simple VQA but fail on tasks requiring genuine cross-modal binding. These benchmarks specifically target that fusion capability.

**The "modality ablation" test:** A powerful diagnostic that the harness supports is running a cross-modal benchmark in three configurations:
1. **Full multimodal**: Image + text (the normal setting).
2. **Text-only**: Question text + image caption (no actual image).
3. **Image-only**: Image + generic question template (no specific context).

If a model's accuracy in mode 2 approaches mode 1, the benchmark is not actually testing multimodal reasoning --- it's solvable from text alone. This ablation exposes "fake multimodal" benchmarks and helps researchers design better ones.

```yaml
# Cross-modal benchmark manifest extension
cross_modal:
  ablation_modes:
    - full                    # Normal: all modalities
    - text_only               # Replace images with captions
    - image_only              # Replace text with generic template
  requires_simultaneous: true  # Flag: answer requires both modalities
  modality_binding:            # Which text references which image regions
    strategy: explicit         # "explicit" (marked), "implicit" (inferred)
```

---

## 4. Relationship to lm-evaluation-harness

### 4.1 Design Philosophy: Extension, Not Fork

VLM-Evaluation-Harness does not exist in a vacuum. EleutherAI's `lm-evaluation-harness` is the de facto standard for text LLM evaluation, with a mature codebase, extensive benchmark coverage, and a large contributor community. Rather than rebuilding everything from scratch, VLM-Evaluation-Harness is designed as a **complementary extension** that reuses battle-tested components for the language dimension and adds purpose-built machinery for the visual dimension.

The relationship follows the principle: **reuse the text pipeline, extend with vision.**

### 4.2 Components to Reuse from lm-evaluation-harness

| Component | What it provides | How VLM-Evaluation-Harness uses it |
|---|---|---|
| **Text metrics** (exact_match, F1, BLEU, ROUGE, perplexity) | Well-tested, edge-case-hardened implementations of standard NLP metrics | Import directly as a dependency. No reason to rewrite string F1 or token-level BLEU. |
| **Answer extraction / normalization** | Regex-based extractors for multiple choice letters, numbers, yes/no, and free-form text. Handles edge cases like "The answer is B" vs. "B" vs. "(B)" | Reuse the extraction logic and extend it with vision-specific parsers (bounding box, region description, structured visual output). |
| **Few-shot example selection** | Strategies for selecting and formatting in-context examples (random, balanced, similar) | The sampling strategies are modality-agnostic. VLM-Evaluation-Harness adds image handling to the example rendering. |
| **Task grouping and suite management** | How individual benchmarks compose into suites, how results aggregate across groups | Adopt the same grouping semantics and extend with vision-specific taxonomy (perception, reasoning, 3D, cross-modal). |
| **Result serialization and logging** | Structured JSON output, per-sample predictions, aggregated metrics | Use the same output schema for text metrics. Extend with vision-specific fields (image hashes, cost, hallucination scores). |
| **Configuration system** | YAML-based task configuration, override system, CLI argument parsing | Adopt the config patterns and extend the schema with image pipeline, multi-modal, and 3D fields. |
| **Filter and sampling infrastructure** | Deterministic subsampling, stratified sampling, decontamination checks | Reuse directly --- sampling logic is modality-agnostic. |

### 4.3 Components That Must Be Built New

| Component | Why it can't be borrowed |
|---|---|
| **Image pipeline** (normalization, hashing, corruption probes) | No equivalent in text-only harness. |
| **VLM adapter interface** | Text harness uses `lm_eval.api.model.LM` which assumes text-in/text-out. VLM adapters must handle image encoding, multi-image ordering, and vision-specific parameters. |
| **Hallucination evaluation pipeline** | Claim extraction, visual verification, and hallucination-specific metrics are entirely novel. |
| **3D data loading** (point clouds, depth maps, multi-view) | No precedent in text evaluation. |
| **Cross-modal ablation framework** | The modality ablation test is a new concept. |
| **Cost tracking** | Text harness doesn't track API costs. |
| **Visual robustness probes** | Image corruption and augmentation testing has no text analogue. |

### 4.4 Integration Strategy

```python
# Concrete integration approach

# 1. Depend on lm-evaluation-harness as a library
# pyproject.toml:
# dependencies = ["lm-eval>=0.4.0", ...]

# 2. Import text metrics directly
from lm_eval.api.metrics import mean, weighted_perplexity
from lm_eval.filters.extraction import RegexFilter, MapFilter

# 3. Extend the task definition format
# lm-eval uses YAML task configs --- VLM-Evaluation-Harness adds image_config,
# cross_modal, modality, and 3d-specific fields to the same schema.

# 4. Wrap the LM interface with vision support
# The VLMAdapter Protocol extends the concept of lm_eval.api.model.LM
# by adding image parameters. For text-only benchmarks within VLM-Evaluation-Harness,
# the adapter delegates to the underlying LM interface.

# 5. Share result format
# VLM-Evaluation-Harness results are a superset of lm-eval results, so tools built
# for lm-eval output (leaderboards, dashboards) can consume VLM-Evaluation-Harness
# results for text metrics.
```

This approach gives VLM-Evaluation-Harness immediate access to the ~400 text-only tasks in lm-evaluation-harness. Researchers can run a combined evaluation: "evaluate this VLM on MMMU (vision), ScanQA (3D), Winoground (cross-modal), and MMLU (text-only) in one command" --- using lm-eval's text pipeline for the text benchmarks and VLM-Evaluation-Harness's vision pipeline for the rest.

### 4.5 What This Means for the Language Aspect of VLM Evaluation

VLMs don't just need vision evaluation --- they also need to maintain strong language capabilities. Regression testing on text benchmarks is essential. By inheriting from lm-evaluation-harness:

- **Language regression detection**: Run MMLU, HellaSwag, GSM8K alongside vision benchmarks to catch language degradation.
- **Vision-language tradeoff analysis**: Plot text benchmark scores vs. vision benchmark scores across model versions to understand capability tradeoffs.
- **Unified reporting**: A single report card that covers both modalities, not two separate evaluation runs with incompatible output formats.

---

## 5. Architecture

### 5.1 High-Level Data Flow

```
                   ┌──────────────────────────────────────────────────────────┐
                   │                      VLM-Evaluation-Harness                        │
                   │                                                          │
┌───────────┐     │  ┌───────────┐   ┌───────────┐   ┌──────────────────┐   │
│ Benchmark  │────▶│  │  Prompt   │──▶│   Model   │──▶│     Response     │   │
│  Registry  │     │  │ Formatter │   │  Adapter  │   │     Parser       │   │
└───────────┘     │  └───────────┘   └───────────┘   └────────┬─────────┘   │
                   │                                           │              │
                   │                                           ▼              │
┌───────────┐     │  ┌───────────┐   ┌───────────┐   ┌──────────────────┐   │
│  Config    │────▶│  │  Report   │◀──│  Metric   │◀──│   Answer         │   │
│  (YAML)   │     │  │ Generator │   │  Engine   │   │   Extractor      │   │
└───────────┘     │  └───────────┘   └───────────┘   └──────────────────┘   │
                   │                                                          │
                   └──────────────────────────────────────────────────────────┘
```

Each box in this diagram is a pluggable component with a well-defined interface. The harness orchestrates data flow between them but imposes no constraints on their internal implementation.

### 5.2 Component Responsibilities

**Benchmark Registry**
- Stores benchmark definitions as YAML manifests (see Section 5)
- Handles data downloading, caching, and versioning
- Provides an iterator interface that yields `(sample_id, images, metadata)` tuples
- Supports HuggingFace Datasets, local directories, and URLs as data sources

**Prompt Formatter**
- Converts a benchmark sample into the exact prompt string and image list that the model will receive
- Applies the prompt template defined in the benchmark YAML
- Handles few-shot example construction, chain-of-thought wrapping, and system prompt injection
- Logs the exact prompt sent (for reproducibility)

**Model Adapter**
- Provides a uniform `generate()` interface across all model backends
- Handles image encoding (base64, URL, file path) according to each backend's requirements
- Manages rate limiting, retries, and authentication
- Reports token usage and latency per call

**Response Parser**
- Extracts the structured answer from the model's free-form text output
- Handles common patterns: multiple-choice letter extraction, number extraction, yes/no normalization, bounding box parsing
- Configurable per benchmark via parsing rules in the YAML manifest

**Answer Extractor**
- Pairs parsed model responses with ground-truth answers from the benchmark
- Handles answer normalization (case, whitespace, synonyms, units)
- Produces `(predicted, ground_truth, metadata)` tuples for the metric engine

**Metric Engine**
- Computes all configured metrics for the benchmark
- Supports standard metrics (accuracy, F1, BLEU, CIDEr) and custom metrics (CHAIR, LLM-as-judge)
- Aggregates results by sub-category, difficulty, and other metadata dimensions
- Tracks cost (input/output tokens x price) and latency (p50, p95, p99)

**Report Generator**
- Produces human-readable reports in multiple formats (terminal table, Markdown, HTML, JSON)
- Supports model comparison tables and radar charts
- Generates a reproducibility manifest (exact prompts, image hashes, model config, harness version)

### 5.3 The Model Adapter Interface

```python
from typing import Protocol
from dataclasses import dataclass
from PIL import Image


@dataclass
class VLMResponse:
    """Standardized response from any VLM."""
    text: str                          # Raw model output
    input_tokens: int                  # Token count for billing/tracking
    output_tokens: int
    latency_ms: float                  # Wall-clock time for this call
    model_id: str                      # Exact model identifier used
    metadata: dict                     # Backend-specific extras


class VLMAdapter(Protocol):
    """Interface that every model backend must implement."""

    def generate(
        self,
        images: list[Image.Image | str],   # PIL images, file paths, or URLs
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> VLMResponse:
        """Generate a response given images and a text prompt."""
        ...

    @property
    def supports_multi_image(self) -> bool:
        """Whether the model can process multiple images in one call."""
        ...

    @property
    def supports_video(self) -> bool:
        """Whether the model accepts video input."""
        ...

    @property
    def max_resolution(self) -> tuple[int, int] | None:
        """Maximum image resolution, or None if unconstrained."""
        ...

    @property
    def cost_per_million_input_tokens(self) -> float | None:
        """Cost in USD, or None if not applicable (local models)."""
        ...

    @property
    def cost_per_million_output_tokens(self) -> float | None:
        """Cost in USD, or None if not applicable (local models)."""
        ...
```

**Planned adapters:**

| Adapter | Backend | Notes |
|---|---|---|
| `AnthropicAdapter` | Anthropic API | Claude models, supports multi-image, PDF input |
| `OpenAIAdapter` | OpenAI API | GPT-4o, GPT-4-turbo-vision |
| `GoogleAdapter` | Google GenAI API | Gemini models, supports video |
| `HuggingFaceAdapter` | Transformers / AutoModel | LLaVA, InternVL, Qwen-VL, Phi-Vision, etc. |
| `VLLMAdapter` | vLLM server | High-throughput local inference |
| `OllamaAdapter` | Ollama | Easy local model serving |
| `LiteLLMAdapter` | LiteLLM | Unified proxy to 100+ providers |
| `CustomAdapter` | HTTP endpoint | Any model behind a REST API |

---

## 6. Benchmark-as-Configuration

A core design choice: **benchmarks are declared in YAML, not implemented in Python.** This makes adding a new benchmark a data task, not a code task.

### 6.1 Benchmark Manifest Schema

```yaml
# benchmarks/mmmu.yaml

name: MMMU
version: "1.0"
description: >
  Massive Multi-discipline Multimodal Understanding benchmark.
  Tests expert-level perception and reasoning across 30 subjects.

source:
  type: huggingface
  path: MMMU/MMMU
  revision: main          # Pin to a specific commit for reproducibility

splits:
  - name: validation
    scorable: true        # Ground truth available
  - name: test
    scorable: false       # Requires server-side submission

task_type: multiple_choice

# Fields in the dataset
fields:
  question: question
  choices: options          # JSON-encoded list of options
  answer: answer            # Ground truth (letter)
  images:                   # List of columns that may contain images
    - image_1
    - image_2
    - image_3
    - image_4
    - image_5
    - image_6
    - image_7
  subject: subfield         # For per-category breakdown
  difficulty: null          # Not available in this dataset

# How to format the prompt
prompt_template: |
  {question}

  Options:
  {formatted_choices}

  Answer with the option letter only.

# How to handle images
image_config:
  max_images: 7
  placement: before_text    # "before_text", "after_text", "interleaved"
  missing_strategy: skip    # What to do if an image column is null

# How to extract the answer from model output
answer_extraction:
  strategy: first_letter    # "first_letter", "regex", "exact", "number", "json"
  normalize: uppercase

# What metrics to compute
metrics:
  - type: accuracy
  - type: accuracy_by_group
    group_field: subject

# Optional: few-shot examples
few_shot:
  count: 0                  # 0 = zero-shot
  source: validation        # Which split to draw examples from
  strategy: random          # "random", "similar", "fixed"
```

### 6.2 Prompt Template Variables

The prompt formatter resolves the following variables from the benchmark manifest and sample data:

| Variable | Description |
|---|---|
| `{question}` | The question text from the sample |
| `{choices}` | Raw choices data |
| `{formatted_choices}` | Choices formatted as "A. ... B. ... C. ..." |
| `{context}` | Additional context text, if any |
| `{image_description}` | Placeholder text indicating where images are |
| `{few_shot_examples}` | Rendered few-shot examples |
| Any field name | Direct access to any dataset column |

### 6.3 Adding a New Benchmark

The expected workflow for adding a new benchmark:

1. Create a YAML file in `benchmarks/`.
2. Optionally add a custom answer extractor if the default strategies don't work.
3. Run `vlm-evaluation-harness validate-bench --bench my_new_bench` to verify the manifest.
4. Submit a PR (or keep it local).

No Python code required for the common case. For benchmarks with unusual scoring logic (e.g., CIDEr for captioning), a small Python scoring plugin can be registered.

---

## 7. Scoring & Metrics

### 7.1 Standard Metrics

| Metric | Applicable To | Description |
|---|---|---|
| **Accuracy** | Multiple choice, yes/no, extraction | Fraction of exact matches |
| **Accuracy by Group** | Any with metadata | Accuracy broken down by subject, difficulty, etc. |
| **F1 Score** | Extraction tasks | Token-level overlap with ground truth |
| **BLEU / ROUGE / CIDEr** | Captioning, open-ended | N-gram overlap with reference text |
| **ANLS** | Document understanding | Average Normalized Levenshtein Similarity |
| **Relaxed Accuracy** | Numerical extraction | Correct if within a configurable tolerance |
| **IoU** | Bounding box / region tasks | Intersection over Union of predicted vs. ground-truth regions |

### 7.2 LLM-as-Judge

For open-ended tasks where ground truth is a reference answer rather than a definitive answer, the harness supports LLM-as-judge scoring:

```yaml
metrics:
  - type: llm_judge
    judge_model: anthropic:claude-sonnet-4-6
    rubric: |
      Score the response from 1-10 on the following criteria:
      - Accuracy: Does the response correctly describe what's in the image?
      - Completeness: Does it address all parts of the question?
      - Hallucination: Does it mention anything not present in the image?
    max_score: 10
    passes: 1              # Number of independent judgments to average
```

The judge model receives the original image(s), the question, the ground-truth reference, and the model's response. This allows the judge to verify visual claims against the actual image.

### 7.3 Hallucination Metrics

Hallucination detection is a first-class capability of the harness. Three complementary approaches:

**CHAIR (Caption Hallucination Assessment with Image Relevance)**
- For captioning tasks, computes the fraction of mentioned objects that are not present in the ground-truth object list.
- `CHAIR_i` = fraction of captions containing at least one hallucination.
- `CHAIR_s` = fraction of mentioned objects that are hallucinated.

**POPE (Polling-based Object Probing Evaluation)**
- Binary yes/no questions: "Is there a [object] in the image?"
- Objects are selected from three pools: random (easy), popular (medium), adversarial (hard --- objects frequently hallucinated by VLMs).
- Measures accuracy, precision, recall, and F1 on hallucination detection.

**Custom Hallucination Probes**
- A curated dataset of images with exhaustive object inventories.
- The model is asked to describe the image in detail.
- An automated pipeline (or judge model) checks every object/attribute claim against the inventory.
- Produces: object hallucination rate, attribute hallucination rate, spatial hallucination rate, and "I can't tell" calibration score.

### 7.4 Cost & Latency Tracking

For API models, every evaluation run automatically tracks:

| Metric | Description |
|---|---|
| **Total cost (USD)** | Input tokens x rate + output tokens x rate |
| **Cost per sample** | Average cost for one benchmark question |
| **Latency p50 / p95 / p99** | Response time distribution |
| **Tokens per sample** | Average input and output token counts |
| **Throughput** | Samples evaluated per minute |

This makes it possible to answer questions like "Model A is 2% more accurate than Model B, but costs 3x more per sample --- is it worth it?"

---

## 8. Image Pipeline

Image handling is one of the most error-prone aspects of VLM evaluation. Different models have wildly different expectations, and subtle differences in image preprocessing can meaningfully affect results.

### 8.1 Image Normalization

The harness provides a configurable image pipeline:

```yaml
image_pipeline:
  max_resolution: [2048, 2048]    # Downscale if larger
  min_resolution: [64, 64]        # Upscale if smaller (rare)
  format: png                      # Normalize to a single format
  color_space: RGB                 # Convert grayscale to RGB if needed
  hash_algorithm: sha256           # For reproducibility logging
```

The pipeline logs the exact image dimensions and hash sent to each model, so discrepancies can be diagnosed.

### 8.2 Multi-Image Handling

Models handle multiple images differently:

- **Interleaved**: Images are placed inline with text at marked positions.
- **Prefix**: All images are sent before the text prompt.
- **Suffix**: All images are sent after the text prompt.

The harness supports all three strategies and can adapt per model:

```yaml
image_config:
  placement: interleaved
  # For models that don't support interleaving, fall back to:
  fallback_placement: before_text
```

### 8.3 Robustness Probes

The harness can automatically generate corrupted variants of benchmark images to test model robustness:

| Corruption | Parameters |
|---|---|
| **JPEG compression** | Quality: 10, 30, 50, 70 |
| **Gaussian blur** | Sigma: 1, 2, 4, 8 |
| **Gaussian noise** | Sigma: 10, 25, 50 |
| **Rotation** | Degrees: 90, 180, 270 |
| **Center crop** | Fraction: 0.5, 0.75 |
| **Resolution downscale** | Factor: 2x, 4x, 8x |
| **Color jitter** | Brightness, contrast, saturation shifts |
| **Watermark overlay** | Simulated watermark text |

Running a benchmark with `--robustness-probe` generates a performance-vs-corruption plot that reveals which models degrade gracefully and which are brittle.

---

## 9. CLI Design

### 9.1 Core Commands

```bash
# ── Evaluation ──────────────────────────────────────────────────────
# Evaluate a single model on a single benchmark
vlm-evaluation-harness eval \
  --model anthropic:claude-opus-4-6 \
  --bench mmmu \
  --split validation

# Evaluate on multiple benchmarks
vlm-evaluation-harness eval \
  --model openai:gpt-4o \
  --bench mmmu,chartqa,docvqa,textvqa

# Evaluate on a predefined suite
vlm-evaluation-harness eval \
  --model google:gemini-2.5-pro \
  --suite perception          # Runs all perception-category benchmarks

# Evaluate a local model
vlm-evaluation-harness eval \
  --model huggingface:liuhaotian/llava-v1.6-34b \
  --bench vqav2 \
  --device cuda:0 \
  --batch-size 8

# ── Comparison ──────────────────────────────────────────────────────
# Compare multiple models on one benchmark
vlm-evaluation-harness compare \
  --models anthropic:claude-opus-4-6,openai:gpt-4o,google:gemini-2.5-pro \
  --bench mmmu \
  --output comparison.html

# ── Hallucination ───────────────────────────────────────────────────
# Run the dedicated hallucination evaluation suite
vlm-evaluation-harness eval \
  --model anthropic:claude-opus-4-6 \
  --suite hallucination

# ── Safety ──────────────────────────────────────────────────────────
# Run the safety suite
vlm-evaluation-harness eval \
  --model local:llava-next \
  --suite safety

# ── Robustness ──────────────────────────────────────────────────────
# Run with automatic image corruption probes
vlm-evaluation-harness eval \
  --model openai:gpt-4o \
  --bench vqav2 \
  --robustness-probe \
  --corruptions blur,noise,jpeg,rotation

# ── Benchmark Management ────────────────────────────────────────────
# List all available benchmarks
vlm-evaluation-harness list-benchmarks

# Validate a benchmark manifest
vlm-evaluation-harness validate-bench --bench my_custom_bench

# Add a benchmark from HuggingFace
vlm-evaluation-harness add-bench --from hf://lmms-lab/SEED-Bench-2

# Download benchmark data
vlm-evaluation-harness download --bench mmmu --split validation

# ── Reporting ───────────────────────────────────────────────────────
# Generate a detailed report from saved results
vlm-evaluation-harness report --results-dir ./results --format html

# Serve an interactive leaderboard
vlm-evaluation-harness serve --results-dir ./results --port 8080

# ── Utilities ───────────────────────────────────────────────────────
# Estimate the cost of an evaluation run before executing it
vlm-evaluation-harness estimate-cost \
  --model anthropic:claude-opus-4-6 \
  --bench mmmu \
  --split validation

# Reproduce a prior run from its manifest
vlm-evaluation-harness reproduce --manifest results/mmmu_claude_20260413/manifest.json
```

### 9.2 Configuration File

Global defaults can be set in `~/.vlm-evaluation-harness/config.yaml`:

```yaml
default_output_dir: ./results
cache_dir: ~/.vlm-evaluation-harness/cache

# API keys (or use environment variables)
api_keys:
  anthropic: ${ANTHROPIC_API_KEY}
  openai: ${OPENAI_API_KEY}
  google: ${GOOGLE_API_KEY}

# Default evaluation settings
defaults:
  temperature: 0.0
  max_tokens: 1024
  max_concurrent: 10       # Parallel API calls
  retry_attempts: 3
  retry_delay: 1.0

# Cost tracking
cost_tracking:
  enabled: true
  warn_above_usd: 50.0    # Warn before starting expensive runs
```

---

## 10. Hallucination Detection Deep Dive

Hallucination is arguably the most important failure mode of VLMs for production use cases. The harness treats it as a first-class evaluation dimension, not an afterthought.

### 10.1 The Hallucination Evaluation Pipeline

```
┌─────────────┐     ┌───────────────┐     ┌────────────────┐
│   Image +   │────▶│    VLM Under  │────▶│   Claim        │
│   Prompt    │     │    Test       │     │   Extractor    │
└─────────────┘     └───────────────┘     └───────┬────────┘
                                                   │
                                                   ▼
┌─────────────┐     ┌───────────────┐     ┌────────────────┐
│ Hallucin.   │◀────│   Claim       │◀────│   Structured   │
│ Report      │     │   Verifier    │     │   Claims List  │
└─────────────┘     └───────────────┘     └────────────────┘
```

**Step 1: Elicit a description.** The model is prompted to describe the image in detail. The prompt is carefully designed to encourage comprehensive output without leading the model toward specific objects.

**Step 2: Extract claims.** A claim extractor (rule-based or LLM-based) decomposes the model's output into atomic, verifiable claims:
- Object claims: "There is a red bicycle."
- Attribute claims: "The car is blue."
- Spatial claims: "The cat is on top of the table."
- Text claims: "The sign reads 'EXIT'."
- Count claims: "There are three people."

**Step 3: Verify claims.** Each claim is verified against the ground-truth image annotation. For benchmarks with exhaustive annotations (COCO with segmentation masks), this is automated. For others, an LLM judge examines the original image and the claim.

**Step 4: Report.** The pipeline produces:
- **Object hallucination rate**: % of claimed objects not in the image.
- **Attribute hallucination rate**: % of claimed attributes that are wrong.
- **Spatial hallucination rate**: % of spatial relationships that are wrong.
- **Fabrication severity score**: Weighted by how "creative" the hallucination is (claiming a fire truck in a bedroom is worse than confusing a sofa for a loveseat).
- **Calibration score**: How well-calibrated the model's hedging language ("I think", "it appears") correlates with actual uncertainty.

### 10.2 The Hallucination Probe Dataset

The harness ships with a curated hallucination probe dataset:

- 1,000 diverse images spanning indoor scenes, outdoor scenes, documents, charts, medical images, and edge cases.
- Each image has an exhaustive object inventory (every object, its attributes, and spatial relationships annotated by multiple human annotators).
- Three difficulty tiers:
  - **Easy**: Clear, well-lit, unambiguous scenes.
  - **Medium**: Cluttered scenes, partially occluded objects, ambiguous content.
  - **Hard**: Adversarial images designed to trigger common hallucination patterns (e.g., a kitchen with no stove, a parking lot with no cars).

---

## 11. Safety Evaluation

### 11.1 Bias Measurement

The harness measures demographic bias in VLM outputs along several axes:

- **Description disparity**: Do models describe people of different demographics with different levels of detail, different vocabulary, or different assumed activities?
- **Assumption injection**: When asked about ambiguous scenarios, do models inject assumptions correlated with demographic attributes? (e.g., assuming gender based on profession)
- **Performance disparity**: Does accuracy on visual questions vary systematically across images featuring different demographic groups?

### 11.2 Adversarial Visual Prompts

A growing concern is "visual prompt injection" --- embedding instructions in images that override the text prompt. The harness tests:

- **Typographic attacks**: Text overlaid on images instructing the model to ignore its system prompt.
- **Steganographic payloads**: Subtle image modifications not visible to humans but potentially parsed by models.
- **Misleading context**: Images designed to create false impressions that contradict the text prompt.

### 11.3 Refusal Calibration

A well-calibrated model should:
- Refuse to answer when the image is too ambiguous, low-quality, or the question is unanswerable from the visual content.
- Not refuse when the image clearly supports an answer, even if the question is unusual.

The harness measures both false refusals (unnecessary "I can't determine..." responses) and missed refusals (confident answers to genuinely unanswerable questions).

### 11.4 Implemented subset (2026 update)

Sections 10 and 11 above describe the full envisioned pipeline (claim
extraction, an exhaustive 1,000-image probe dataset, bias/adversarial-prompt
testing). What's actually shipped so far, in `metrics/hallucination.py` and
`metrics/calibration.py`:

- **Fine-grained hallucination** (`fine_grained_hallucination` metric, the
  `hallu_fg` benchmark) — a decomposed version of §10's object/attribute/
  spatial hallucination rates, using presence/absence probes tagged by
  category rather than full claim extraction. Same spirit as 2026's
  FIHA/FREAK-style benchmarks.
- **Refusal calibration** (`calibration` metric, the `calib_deflect`
  benchmark) — a first cut at §11.3: answerable vs. unanswerable questions,
  scored on whether the model answers/deflects correctly, reporting
  `overconfidence_rate` and `underconfidence_rate` separately. Same spirit as
  VLM-DeflectionBench.
- **Compositional hard negatives** (`accuracy` on the `comp_hardneg`
  benchmark) — not previously in this doc at all. In the spirit of 2026's
  ConMe/SCRAMBLe line of work: attribute/relation/object-swap hard negatives
  against a correct caption, which plain accuracy on natural captions cannot
  distinguish from bag-of-words matching.

Not yet implemented: LLM-based claim extraction/verification, the curated
1,000-image tiered probe dataset, bias measurement (§11.1), and adversarial
visual prompts (§11.2) — all three fixture benchmarks above are small,
hand-authored, and offline, standing in for what full datasets would cover.

---

## 12. Reproducibility

Reproducibility is a core design goal. Every evaluation run produces a manifest file:

```json
{
  "harness_version": "0.3.1",
  "timestamp": "2026-04-13T14:30:00Z",
  "model": {
    "adapter": "anthropic",
    "model_id": "claude-opus-4-6",
    "temperature": 0.0,
    "max_tokens": 1024
  },
  "benchmark": {
    "name": "MMMU",
    "version": "1.0",
    "split": "validation",
    "manifest_hash": "sha256:abc123...",
    "data_hash": "sha256:def456..."
  },
  "prompt_template_hash": "sha256:789ghi...",
  "image_pipeline": {
    "max_resolution": [2048, 2048],
    "format": "png"
  },
  "results": {
    "accuracy": 0.621,
    "samples_evaluated": 900,
    "total_cost_usd": 12.34,
    "total_latency_seconds": 1847.2
  },
  "sample_hashes": "sha256:jkl012..."
}
```

The `vlm-evaluation-harness reproduce` command re-runs an evaluation from a manifest file, verifying that all inputs (images, prompts, model config) match the original run.

---

## 13. Differentiation from Existing Tools

| Tool | Strengths | Gaps that VLM-Evaluation-Harness fills |
|---|---|---|
| **lm-evaluation-harness** (EleutherAI) | De facto standard for text LLM eval | Image support is minimal and bolted-on; no VLM-specific metrics |
| **lmms-eval** | Good benchmark coverage for VLMs | Tightly coupled to HuggingFace Transformers; weak API model support; no cost tracking; complex codebase |
| **VLMEvalKit** (OpenCompass) | Broad benchmark coverage | Hard to extend; monolithic architecture; limited hallucination testing |
| **OpenCompass** | Comprehensive Chinese + English eval | Heavy infrastructure requirements; opinionated on benchmark selection |
| **HELM** (Stanford) | Rigorous methodology | Slow to add new benchmarks; limited VLM support |
| **simple-evals** (OpenAI) | Clean, minimal | Very limited benchmark set; single-provider focus |

**VLM-Evaluation-Harness differentiators:**

1. **Benchmark-as-YAML**: Adding a new benchmark should take minutes, not hours.
2. **Provider-agnostic**: First-class support for all major API providers and local inference engines.
3. **Hallucination-first**: Dedicated hallucination evaluation pipeline, not just accuracy metrics.
4. **Cost-aware**: Automatic cost and latency tracking makes practical model selection possible.
5. **Reproducibility by default**: Every run produces a manifest that can recreate the exact evaluation.
6. **Safety built-in**: Bias, adversarial robustness, and refusal calibration are standard suites, not afterthoughts.

---

## 14. Open Questions & Future Directions

### 14.1 Video Support

Many frontier VLMs now accept video input (Gemini, GPT-4o). Video benchmarks like Video-MME, ActivityNet-QA, and MVBench test temporal understanding that still images cannot. The question is whether to scope video into v1 or defer to v2.

**Recommendation**: Include video support in the adapter interface from day one (the `supports_video` property) but defer video benchmarks to v1.1. This avoids breaking changes later while keeping the initial scope manageable.

### 14.2 Interactive & Agentic Tasks

Some VLM evaluations are multi-turn or interactive: VisualWebArena (navigate a website by looking at screenshots), OSWorld (operate a desktop), or SWE-bench-multimodal (fix bugs from screenshots). These require an environment loop, not a single prompt-response pair.

**Recommendation**: Design the core loop to support multi-turn interactions (the adapter already takes a conversation history), but defer agentic benchmarks with environment simulation to v2. The harness architecture should not preclude this extension.

### 14.3 Closed Test Set Submission

Benchmarks like MMMU-test, MMBench-test, and others require submitting predictions to a server for scoring. The harness should support generating submission files in each benchmark's expected format.

**Recommendation**: Include a `--generate-submission` flag that produces the correctly-formatted output file without attempting local scoring.

### 14.4 Leaderboard Hosting

A natural extension is a public or private leaderboard that aggregates results across models and benchmarks.

**Recommendation**: Ship a local `vlm-evaluation-harness serve` command for private leaderboards in v1. A hosted public leaderboard is a v2 feature that requires community infrastructure.

### 14.5 Continuous Evaluation

For model providers, running evaluations on every model update is essential. Integration with CI/CD pipelines (GitHub Actions, etc.) would enable automated regression detection.

**Recommendation**: Provide a `vlm-evaluation-harness ci` command that runs a configurable eval suite and exits with a non-zero code if any metric drops below a threshold. This enables simple CI integration without building a full CI framework.

---

## 15. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-4)

- Core architecture: adapter interface, benchmark registry, prompt formatter, metric engine
- Adapters: Anthropic, OpenAI, HuggingFace Transformers
- Benchmarks: MMMU, VQAv2, ChartQA, TextVQA, DocVQA (5 benchmarks covering the core task types)
- Metrics: accuracy, accuracy-by-group, F1, ANLS
- CLI: `eval`, `list-benchmarks`, `download`
- Output: terminal tables, JSON results, reproducibility manifest

### Phase 2: Hallucination & Safety (Weeks 5-8)

- Hallucination evaluation pipeline (claim extraction, verification, reporting)
- POPE and CHAIR metric implementations
- Custom hallucination probe dataset (v1: 250 images)
- Safety suite: bias measurement, adversarial visual prompts, refusal calibration
- LLM-as-judge scoring
- Adapters: Google GenAI, vLLM, Ollama

### Phase 3: 3D Vision & Cross-Modal (Weeks 9-12)

- 3D data loading infrastructure (point clouds, depth maps, multi-view images)
- 3D benchmark manifests: ScanQA, SQA3D, Structured3D
- Cross-modal reasoning benchmarks: Winoground, SNLI-VE, interleaved document QA
- Modality ablation framework (full / text-only / image-only comparison)
- Cross-modal metric implementations (binding accuracy, ablation delta)
- lm-evaluation-harness integration for text-only regression benchmarks (MMLU, HellaSwag, GSM8K)
- Unified reporting across text + vision + 3D + cross-modal

### Phase 4: Scale & Polish (Weeks 13-16)

- 30+ benchmarks covering all taxonomy categories including 3D and cross-modal
- Image robustness probes
- Cost tracking and estimation
- Model comparison reports (HTML with charts)
- Local leaderboard server
- Comprehensive documentation and contributor guide
- LiteLLM adapter for broad provider coverage

### Phase 5: Community & Extensions (Ongoing)

- Video benchmark support
- Multi-turn / agentic evaluation support
- CI integration (`vlm-evaluation-harness ci`)
- Community benchmark contributions
- Public leaderboard (if demand warrants)
- Plugin system for custom metrics and adapters
- Audio-visual benchmarks (MUSIC-AVQA) for models supporting audio+vision
- Embodied 3D evaluation (interaction with 3D environments)

---

## 16. Project Structure

```
vlm-evaluation-harness/
├── README.md
├── pyproject.toml
├── LICENSE                        # Apache 2.0
│
├── src/
│   └── vlm_evaluation_harness/
│       ├── __init__.py
│       ├── cli.py                 # CLI entry point (click/typer)
│       ├── config.py              # Configuration loading and validation
│       │
│       ├── adapters/              # Model adapters
│       │   ├── __init__.py
│       │   ├── base.py            # VLMAdapter protocol
│       │   ├── anthropic.py
│       │   ├── openai.py
│       │   ├── google.py
│       │   ├── huggingface.py
│       │   ├── vllm.py
│       │   ├── ollama.py
│       │   └── litellm.py
│       │
│       ├── benchmarks/            # Benchmark registry and loading
│       │   ├── __init__.py
│       │   ├── registry.py        # Discovers and loads YAML manifests
│       │   ├── loader.py          # Downloads and caches benchmark data
│       │   └── manifests/         # YAML benchmark definitions
│       │       ├── mmmu.yaml
│       │       ├── vqav2.yaml
│       │       ├── chartqa.yaml
│       │       └── ...
│       │
│       ├── prompt/                # Prompt formatting
│       │   ├── __init__.py
│       │   ├── formatter.py       # Template resolution
│       │   ├── few_shot.py        # Few-shot example selection
│       │   └── templates.py       # Common prompt patterns
│       │
│       ├── parsing/               # Response parsing
│       │   ├── __init__.py
│       │   ├── extractor.py       # Answer extraction strategies
│       │   └── normalizer.py      # Answer normalization
│       │
│       ├── metrics/               # Scoring
│       │   ├── __init__.py
│       │   ├── accuracy.py
│       │   ├── nlp_metrics.py     # BLEU, ROUGE, CIDEr
│       │   ├── hallucination.py   # CHAIR, POPE, custom probes
│       │   ├── llm_judge.py       # LLM-as-judge scoring
│       │   ├── cost.py            # Cost and latency tracking
│       │   └── safety.py          # Bias, adversarial, refusal metrics
│       │
│       ├── images/                # Image pipeline
│       │   ├── __init__.py
│       │   ├── pipeline.py        # Normalization, resizing, hashing
│       │   ├── corruption.py      # Robustness probe corruptions
│       │   └── multi_image.py     # Multi-image ordering strategies
│       │
│       ├── three_d/               # 3D vision support
│       │   ├── __init__.py
│       │   ├── loaders.py         # Point cloud, depth map, multi-view loading
│       │   ├── transforms.py      # 3D data normalization and augmentation
│       │   └── metrics.py         # 3D-specific metrics (depth error, IoU-3D)
│       │
│       ├── cross_modal/           # Cross-modal reasoning support
│       │   ├── __init__.py
│       │   ├── ablation.py        # Modality ablation framework
│       │   ├── binding.py         # Cross-modal reference resolution
│       │   └── metrics.py         # Cross-modal specific metrics
│       │
│       ├── compat/                # lm-evaluation-harness integration
│       │   ├── __init__.py
│       │   ├── lm_eval_bridge.py  # Bridge to lm-eval text tasks
│       │   ├── metric_import.py   # Import text metrics from lm-eval
│       │   └── task_adapter.py    # Adapt lm-eval tasks for unified reporting
│       │
│       ├── reporting/             # Output and visualization
│       │   ├── __init__.py
│       │   ├── terminal.py        # Rich terminal tables
│       │   ├── markdown.py        # Markdown reports
│       │   ├── html.py            # HTML reports with charts
│       │   ├── json_report.py     # Machine-readable output
│       │   └── leaderboard.py     # Local leaderboard server
│       │
│       └── engine/                # Orchestration
│           ├── __init__.py
│           ├── runner.py          # Main evaluation loop
│           ├── parallel.py        # Concurrent evaluation
│           └── reproduce.py       # Reproducibility from manifests
│
├── data/
│   └── hallucination_probes/      # Curated hallucination test images
│       ├── images/
│       ├── annotations/
│       └── README.md
│
├── tests/
│   ├── test_adapters/
│   ├── test_benchmarks/
│   ├── test_metrics/
│   ├── test_parsing/
│   └── test_cli/
│
├── docs/
│   ├── getting_started.md
│   ├── adding_benchmarks.md
│   ├── adding_adapters.md
│   ├── hallucination_eval.md
│   └── api_reference.md
│
└── examples/
    ├── basic_eval.py
    ├── custom_benchmark.py
    ├── model_comparison.py
    └── hallucination_analysis.py
```

---

## 17. Summary

VLM-Evaluation-Harness aims to become the standard evaluation framework for Vision Language Models by solving the biggest pain points in the current ecosystem:

1. **Fragmentation** --- one framework, any model, any benchmark.
2. **Inconsistency** --- declarative benchmarks with reproducibility manifests eliminate methodology variance.
3. **Incomplete measurement** --- hallucination detection, safety evaluation, and cost tracking are built in, not bolted on.
4. **Missing dimensions** --- 3D vision and true cross-modal reasoning are first-class evaluation categories, not afterthoughts.
5. **Reinventing the wheel** --- the language evaluation backbone is inherited from lm-evaluation-harness, not rebuilt from scratch.

The design prioritizes extensibility (YAML benchmarks, adapter protocol), practicality (cost tracking, CLI-first), rigor (reproducibility manifests, robustness probes), and completeness (2D + 3D + cross-modal + text regression). By building on the shoulders of lm-evaluation-harness and extending into the visual domain with purpose-built machinery, VLM-Evaluation-Harness shifts researcher time from evaluation infrastructure to actual research.
