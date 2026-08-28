"""Pydantic-free dataclass schema for benchmark manifests."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import Any

# Metric types the discriminative dispatcher knows how to compute.
KNOWN_METRIC_TYPES = {
    "accuracy",
    "accuracy_by_group",
    "vqa_accuracy",
    "relaxed_accuracy",
    "f1",
    "anls",
    "bleu",
    "rouge",
    "pope",
    "chair",
    "pairwise_group",
    "fine_grained_hallucination",
    "calibration",
}

# Metric types the generative dispatcher knows how to compute.
KNOWN_GENERATIVE_METRIC_TYPES = {
    "clip_score",
    "geneval_clip",
    "fid",
    "llm_judge",
    "vqa_score",
}

KNOWN_TASK_TYPES = {
    "open_ended",
    "multiple_choice",
    "yes_no",
    "captioning",
    "pairwise_matching",
    "text_to_image",
}

KNOWN_SCORING_MODES = {"generate", "loglikelihood"}

# Manifest schema (format) versions this loader knows how to read.
SUPPORTED_SCHEMA_VERSIONS = {"1.0"}


class ManifestError(ValueError):
    """Raised when a benchmark manifest is structurally invalid."""


@dataclass
class SplitConfig:
    name: str
    scorable: bool = True


@dataclass
class SourceConfig:
    type: str  # "huggingface" | "local"
    path: str
    revision: str = "main"
    subset: str | None = None


@dataclass
class FieldsConfig:
    question: str | None = "question"
    answer: str | None = "answer"
    # Additional reference columns. Datasets like VQAv2 carry a list of
    # annotator answers alongside the single canonical one; every value
    # found here is merged into the sample's reference list.
    answers: str | None = None
    choices: str | None = None
    images: list[str] = field(default_factory=list)
    subject: str | None = None
    difficulty: str | None = None
    context: str | None = None
    # Arbitrary dataset columns exposed to the prompt template by name.
    # This is what lets a manifest write `{caption_0}` and have it resolved.
    text_fields: dict[str, str] = field(default_factory=dict)
    # Arbitrary dataset columns copied verbatim into sample.metadata.
    metadata_fields: list[str] = field(default_factory=list)


@dataclass
class ImageConfig:
    max_images: int = 1
    placement: str = "before_text"  # "before_text" | "after_text"
    missing_strategy: str = "skip"  # "skip" | "error"


@dataclass
class AnswerExtractionConfig:
    strategy: str = "exact"  # "first_letter" | "regex" | "exact" | "number" | "yes_no" | "json"
    normalize: str = "strip"  # "strip" | "uppercase" | "lowercase" | "vqa" | "none"
    regex_pattern: str | None = None
    # Ordered post-normalization filters (see parsing/filters.py), applied
    # after `normalize` for composable cleanup a single normalize mode
    # doesn't cover. Empty by default — fully backward compatible.
    filters: list[str] = field(default_factory=list)


@dataclass
class FewShotConfig:
    count: int = 0
    source: str = "train"
    strategy: str = "fixed"  # "fixed" | "random"
    seed: int = 42
    # "concatenated": examples flattened into the prompt text (default,
    #   original behavior). "multi_turn": examples rendered as alternating
    #   user/assistant ConversationTurns, for adapters whose chat template
    #   expects real turns rather than a single long user message.
    mode: str = "concatenated"


@dataclass
class MetricConfig:
    type: str
    group_field: str | None = None
    judge_model: str | None = None
    rubric: str | None = None
    max_score: float = 10.0
    # Relative tolerance for `relaxed_accuracy` (ChartQA's official metric
    # accepts numeric answers within 5% of the reference).
    tolerance: float = 0.05
    # Generative-metric knobs.
    clip_model_id: str | None = None
    checks_field: str | None = None
    reference_dir: str | None = None
    # For `chair`: the metadata column holding ground-truth object labels.
    objects_field: str | None = None
    # For `fine_grained_hallucination`: metadata column holding the probe
    # category (object/attribute/relation). For `calibration`: metadata
    # column holding whether the sample is answerable from the image.
    field_name: str | None = None


@dataclass
class BenchmarkManifest:
    name: str
    source: SourceConfig
    splits: list[SplitConfig]

    # Manifest *format* version — distinct from `version` below, which is
    # the benchmark's own content version. Bumped only when a manifest field
    # is renamed/removed in a way older loaders can't parse; the loader
    # rejects any schema_version it doesn't know how to read, rather than
    # silently misinterpreting a manifest written for a newer format.
    schema_version: str = "1.0"
    version: str = "1.0"
    description: str = ""
    task_type: str = "open_ended"
    modality: str = "2d"  # "2d" | "3d" | "text_only"
    taxonomy_category: str = "perception"
    # Free-form labels for cross-cutting grouping (e.g. "safety",
    # "compositional") that don't fit the single taxonomy_category axis.
    tags: list[str] = field(default_factory=list)
    # "generate": sample free text, then extract an answer.
    # "loglikelihood": score each choice by log-probability and pick the best.
    #   Requires an adapter implementing `score_choices`; MC only.
    scoring: str = "generate"

    fields: FieldsConfig = field(default_factory=FieldsConfig)
    image_config: ImageConfig = field(default_factory=ImageConfig)
    answer_extraction: AnswerExtractionConfig = field(default_factory=AnswerExtractionConfig)
    few_shot: FewShotConfig = field(default_factory=FewShotConfig)
    metrics: list[MetricConfig] = field(default_factory=lambda: [MetricConfig(type="accuracy")])
    prompt_template: str = "{question}"
    system_prompt: str | None = None
    # For `pairwise_matching` (Winoground-style): the second prompt, asking
    # about the other image. Both are scored per sample.
    prompt_template_b: str | None = None
    # Correct answer for `prompt_template` and `prompt_template_b` respectively.
    # Winoground-style pairs are structural (caption_0 belongs to image_0), so
    # the ground truth lives in the manifest rather than a dataset column.
    pairwise_answers: list[str] = field(default_factory=lambda: ["A", "B"])

    def template_variables(self) -> set[str]:
        """Every `{placeholder}` referenced by this manifest's templates."""
        names: set[str] = set()
        for template in (self.prompt_template, self.prompt_template_b):
            if not template:
                continue
            for _, name, _, _ in string.Formatter().parse(template):
                if name:
                    names.add(name.split(".")[0].split("[")[0])
        return names

    def available_variables(self) -> set[str]:
        """Every placeholder name the loader can actually populate."""
        names = {"few_shot_examples"}
        if self.fields.question:
            names.add("question")
        if self.fields.choices:
            names.update({"choices", "formatted_choices"})
        if self.fields.context:
            names.add("context")
        names.update(self.fields.text_fields)
        return names

    def validate(self) -> None:
        """Structural validation. Raises ManifestError on anything unusable.

        This is deliberately strict: a manifest that scores against a field
        the dataset never provides, or renders a template placeholder the
        loader cannot fill, produces silent zeros rather than an error at
        run time. Catching it here is the whole point.
        """
        errors: list[str] = []

        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            errors.append(
                f"unsupported manifest schema_version {self.schema_version!r} "
                f"(this harness supports: {sorted(SUPPORTED_SCHEMA_VERSIONS)})"
            )
        if self.source.type not in {"huggingface", "local"}:
            errors.append(f"unsupported source.type {self.source.type!r}")
        if not self.splits:
            errors.append("no splits declared")
        if self.task_type not in KNOWN_TASK_TYPES:
            errors.append(
                f"unknown task_type {self.task_type!r} (known: {sorted(KNOWN_TASK_TYPES)})"
            )
        if self.scoring not in KNOWN_SCORING_MODES:
            errors.append(
                f"unknown scoring {self.scoring!r} (known: {sorted(KNOWN_SCORING_MODES)})"
            )
        if self.scoring == "loglikelihood" and self.task_type != "multiple_choice":
            errors.append("scoring 'loglikelihood' requires task_type 'multiple_choice'")
        if self.scoring == "loglikelihood" and not self.fields.choices:
            errors.append("scoring 'loglikelihood' requires fields.choices")

        # Unresolvable template placeholders are the single most common way a
        # manifest silently sends garbage to the model.
        missing = self.template_variables() - self.available_variables()
        if missing:
            errors.append(
                f"prompt template references {sorted(missing)}, which the loader cannot "
                "populate — declare them under fields.text_fields (or fields.question / "
                "fields.choices / fields.context)"
            )

        pairwise = self.task_type == "pairwise_matching"
        if pairwise and not self.prompt_template_b:
            errors.append("task_type 'pairwise_matching' requires prompt_template_b")
        if pairwise and len(self.pairwise_answers) != 2:
            errors.append("pairwise_answers must contain exactly two entries")

        # A scorable split with no reference field can never produce a score.
        generative = self.task_type == "text_to_image"
        scorable = [s.name for s in self.splits if s.scorable]
        if (
            scorable
            and not pairwise
            and not generative
            and not (self.fields.answer or self.fields.answers)
        ):
            errors.append(
                f"splits {scorable} are marked scorable but no fields.answer/fields.answers "
                "is declared — scoring would compare against empty references"
            )

        if not self.metrics:
            errors.append("no metrics declared")
        known = KNOWN_GENERATIVE_METRIC_TYPES if generative else KNOWN_METRIC_TYPES
        for metric in self.metrics:
            if metric.type not in known:
                errors.append(
                    f"unknown metric type {metric.type!r} for task_type {self.task_type!r} "
                    f"(known: {sorted(known)})"
                )
            if metric.type == "accuracy_by_group" and not metric.group_field:
                errors.append("metric 'accuracy_by_group' requires group_field")
            if metric.type == "chair" and not metric.objects_field:
                errors.append("metric 'chair' requires objects_field")
            field_name_required = {"fine_grained_hallucination", "calibration"}
            if metric.type in field_name_required and not metric.field_name:
                errors.append(f"metric {metric.type!r} requires field_name")
            if metric.type == "llm_judge" and not metric.judge_model:
                errors.append("metric 'llm_judge' requires judge_model")
            if metric.type == "vqa_score" and not metric.judge_model:
                errors.append("metric 'vqa_score' requires judge_model")
            if metric.type == "fid" and not metric.reference_dir:
                errors.append("metric 'fid' requires reference_dir")
            if metric.group_field and metric.group_field not in self._metadata_keys():
                errors.append(
                    f"metric group_field {metric.group_field!r} is not produced by the loader — "
                    "declare it via fields.subject, fields.difficulty, or fields.metadata_fields"
                )
            if metric.field_name and metric.field_name not in self._metadata_keys():
                errors.append(
                    f"metric field_name {metric.field_name!r} is not produced by the loader — "
                    "declare it via fields.metadata_fields"
                )

        if self.answer_extraction.strategy == "regex" and not self.answer_extraction.regex_pattern:
            errors.append("answer_extraction strategy 'regex' requires regex_pattern")
        if self.answer_extraction.regex_pattern:
            try:
                re.compile(self.answer_extraction.regex_pattern)
            except re.error as exc:
                errors.append(f"invalid regex_pattern: {exc}")
        if self.answer_extraction.filters:
            from vlm_evaluation_harness.parsing.filters import FILTERS as _KNOWN_FILTERS

            unknown = [f for f in self.answer_extraction.filters if f not in _KNOWN_FILTERS]
            if unknown:
                errors.append(
                    f"unknown answer_extraction.filters {unknown} "
                    f"(known: {sorted(_KNOWN_FILTERS)})"
                )

        if self.few_shot.count and self.few_shot.source not in {s.name for s in self.splits}:
            errors.append(
                f"few_shot.source {self.few_shot.source!r} is not one of this benchmark's splits"
            )
        if self.few_shot.mode not in {"concatenated", "multi_turn"}:
            errors.append(
                f"unknown few_shot.mode {self.few_shot.mode!r} "
                "(known: 'concatenated', 'multi_turn')"
            )

        if errors:
            raise ManifestError(
                f"benchmark '{self.name}' is invalid:\n  - " + "\n  - ".join(errors)
            )

    def _metadata_keys(self) -> set[str]:
        keys = set(self.fields.metadata_fields)
        if self.fields.subject:
            keys.add("subject")
        if self.fields.difficulty:
            keys.add("difficulty")
        return keys

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkManifest:
        source = SourceConfig(**data["source"])
        splits = [
            SplitConfig(**s) if isinstance(s, dict) else SplitConfig(name=s)
            for s in data["splits"]
        ]

        fields_data = data.get("fields", {})
        fields = FieldsConfig(
            question=fields_data.get("question", "question"),
            answer=fields_data.get("answer", "answer"),
            answers=fields_data.get("answers"),
            choices=fields_data.get("choices"),
            images=fields_data.get("images", []),
            subject=fields_data.get("subject"),
            difficulty=fields_data.get("difficulty"),
            context=fields_data.get("context"),
            text_fields=fields_data.get("text_fields", {}) or {},
            metadata_fields=fields_data.get("metadata_fields", []),
        )

        img_data = data.get("image_config", {})
        image_config = ImageConfig(
            max_images=img_data.get("max_images", 1),
            placement=img_data.get("placement", "before_text"),
            missing_strategy=img_data.get("missing_strategy", "skip"),
        )

        ae_data = data.get("answer_extraction", {})
        answer_extraction = AnswerExtractionConfig(
            strategy=ae_data.get("strategy", "exact"),
            normalize=ae_data.get("normalize", "strip"),
            regex_pattern=ae_data.get("regex_pattern"),
            filters=list(ae_data.get("filters", [])),
        )

        fs_data = data.get("few_shot", {})
        few_shot = FewShotConfig(
            count=fs_data.get("count", 0),
            source=fs_data.get("source", "train"),
            strategy=fs_data.get("strategy", "fixed"),
            seed=fs_data.get("seed", 42),
            mode=fs_data.get("mode", "concatenated"),
        )

        metrics = [
            MetricConfig(
                type=m["type"],
                group_field=m.get("group_field"),
                judge_model=m.get("judge_model"),
                rubric=m.get("rubric"),
                max_score=m.get("max_score", 10.0),
                tolerance=m.get("tolerance", 0.05),
                clip_model_id=m.get("clip_model_id"),
                checks_field=m.get("checks_field"),
                reference_dir=m.get("reference_dir"),
                objects_field=m.get("objects_field"),
                field_name=m.get("field_name"),
            )
            for m in data.get("metrics", [{"type": "accuracy"}])
        ]

        return cls(
            name=data["name"],
            source=source,
            splits=splits,
            schema_version=str(data.get("schema_version", "1.0")),
            version=str(data.get("version", "1.0")),
            description=data.get("description", ""),
            task_type=data.get("task_type", "open_ended"),
            modality=data.get("modality", "2d"),
            taxonomy_category=data.get("taxonomy_category", "perception"),
            tags=list(data.get("tags", [])),
            scoring=data.get("scoring", "generate"),
            fields=fields,
            image_config=image_config,
            answer_extraction=answer_extraction,
            few_shot=few_shot,
            metrics=metrics,
            prompt_template=data.get("prompt_template", "{question}"),
            prompt_template_b=data.get("prompt_template_b"),
            pairwise_answers=data.get("pairwise_answers", ["A", "B"]),
            system_prompt=data.get("system_prompt"),
        )
