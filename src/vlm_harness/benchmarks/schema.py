"""Pydantic-free dataclass schema for benchmark manifests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SplitConfig:
    name: str
    scorable: bool = True


@dataclass
class SourceConfig:
    type: str  # "huggingface", "local", "url"
    path: str
    revision: str = "main"
    subset: str | None = None
    # Required whenever revision is a mutable ref (e.g. "main") instead of a
    # pinned commit sha — explains why it can't be pinned yet (e.g. the
    # dataset is gated and needs manual acceptance before its sha can be
    # fetched). Enforced by scripts/check_pinned_revisions.py.
    revision_note: str | None = None


@dataclass
class FieldsConfig:
    question: str = "question"
    answer: str = "answer"
    choices: str | None = None
    images: list[str] = field(default_factory=list)
    subject: str | None = None
    difficulty: str | None = None
    context: str | None = None
    # Generic passthrough: arbitrary dataset columns copied verbatim into
    # sample.metadata, keyed by their own column name. Used by generative
    # benchmarks to carry structured compositional checks (count/color/shape)
    # alongside the prompt.
    metadata_fields: list[str] = field(default_factory=list)


@dataclass
class ImageConfig:
    max_images: int = 1
    placement: str = "before_text"  # "before_text" | "after_text" | "interleaved"
    missing_strategy: str = "skip"  # "skip" | "error"
    fallback_placement: str = "before_text"


@dataclass
class AnswerExtractionConfig:
    strategy: str = "exact"  # "first_letter" | "regex" | "exact" | "number" | "json"
    normalize: str = "strip"  # "strip" | "uppercase" | "lowercase" | "none"
    regex_pattern: str | None = None


@dataclass
class FewShotConfig:
    count: int = 0
    source: str = "validation"
    strategy: str = "random"  # "random" | "fixed"
    seed: int = 42


@dataclass
class MetricConfig:
    type: str
    group_field: str | None = None
    judge_model: str | None = None
    rubric: str | None = None
    max_score: float = 10.0
    tolerance: float = 0.05
    # Generative-metric knobs (clip_score, geneval_clip, fid, llm_judge over images)
    clip_model_id: str | None = None
    checks_field: str | None = None
    reference_dir: str | None = None


@dataclass
class CrossModalConfig:
    ablation_modes: list[str] = field(default_factory=lambda: ["full"])
    requires_simultaneous: bool = False


@dataclass
class ThreeDConfig:
    depth_maps: list[str] = field(default_factory=list)
    point_clouds: list[str] = field(default_factory=list)
    camera_params: list[str] = field(default_factory=list)
    multi_view: list[str] = field(default_factory=list)


@dataclass
class BenchmarkManifest:
    name: str
    source: SourceConfig
    splits: list[SplitConfig]

    version: str = "1.0"
    description: str = ""
    task_type: str = "open_ended"
    modality: str = "2d"  # "2d" | "3d" | "cross_modal" | "text_only"
    taxonomy_category: str = "perception"

    fields: FieldsConfig = field(default_factory=FieldsConfig)
    image_config: ImageConfig = field(default_factory=ImageConfig)
    answer_extraction: AnswerExtractionConfig = field(default_factory=AnswerExtractionConfig)
    few_shot: FewShotConfig = field(default_factory=FewShotConfig)
    metrics: list[MetricConfig] = field(default_factory=lambda: [MetricConfig(type="accuracy")])
    prompt_template: str = "{question}"
    system_prompt: str | None = None

    cross_modal: CrossModalConfig | None = None
    three_d: ThreeDConfig | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkManifest:
        source = SourceConfig(**data["source"])
        splits = [SplitConfig(**s) if isinstance(s, dict) else SplitConfig(name=s) for s in data["splits"]]

        fields_data = data.get("fields", {})
        fields = FieldsConfig(
            question=fields_data.get("question", "question"),
            answer=fields_data.get("answer", "answer"),
            choices=fields_data.get("choices"),
            images=fields_data.get("images", []),
            subject=fields_data.get("subject"),
            difficulty=fields_data.get("difficulty"),
            context=fields_data.get("context"),
            metadata_fields=fields_data.get("metadata_fields", []),
        )

        img_data = data.get("image_config", {})
        image_config = ImageConfig(
            max_images=img_data.get("max_images", 1),
            placement=img_data.get("placement", "before_text"),
            missing_strategy=img_data.get("missing_strategy", "skip"),
            fallback_placement=img_data.get("fallback_placement", "before_text"),
        )

        ae_data = data.get("answer_extraction", {})
        answer_extraction = AnswerExtractionConfig(
            strategy=ae_data.get("strategy", "exact"),
            normalize=ae_data.get("normalize", "strip"),
            regex_pattern=ae_data.get("regex_pattern"),
        )

        fs_data = data.get("few_shot", {})
        few_shot = FewShotConfig(
            count=fs_data.get("count", 0),
            source=fs_data.get("source", "validation"),
            strategy=fs_data.get("strategy", "random"),
            seed=fs_data.get("seed", 42),
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
            )
            for m in data.get("metrics", [{"type": "accuracy"}])
        ]

        cross_modal = None
        if "cross_modal" in data:
            cross_modal = CrossModalConfig(**data["cross_modal"])

        three_d = None
        if "three_d" in data:
            three_d = ThreeDConfig(**data["three_d"])

        return cls(
            name=data["name"],
            source=source,
            splits=splits,
            version=str(data.get("version", "1.0")),
            description=data.get("description", ""),
            task_type=data.get("task_type", "open_ended"),
            modality=data.get("modality", "2d"),
            taxonomy_category=data.get("taxonomy_category", "perception"),
            fields=fields,
            image_config=image_config,
            answer_extraction=answer_extraction,
            few_shot=few_shot,
            metrics=metrics,
            prompt_template=data.get("prompt_template", "{question}"),
            system_prompt=data.get("system_prompt"),
            cross_modal=cross_modal,
            three_d=three_d,
        )
