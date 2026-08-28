"""Dataset loading and caching for benchmarks."""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from vlm_evaluation_harness.benchmarks.schema import BenchmarkManifest

_CACHE_DIR = Path.home() / ".vlm-evaluation-harness" / "cache"

logger = logging.getLogger(__name__)


class BenchmarkSample:
    """A single sample from a benchmark dataset.

    `references` is a list because most open-ended benchmarks are scored
    against several acceptable answers (VQAv2 ships ten annotator responses
    per question). An empty list means this sample has no ground truth and
    must be excluded from scoring rather than compared against "".
    """

    __slots__ = ("sample_id", "images", "text_fields", "references", "metadata")

    def __init__(
        self,
        sample_id: str,
        images: list,
        text_fields: dict[str, Any],
        references: list[str],
        metadata: dict[str, Any],
    ):
        self.sample_id = sample_id
        self.images = images
        self.text_fields = text_fields
        self.references = references
        self.metadata = metadata

    @property
    def has_reference(self) -> bool:
        return bool(self.references)


def _coerce_references(value: Any) -> list[str]:
    """Normalize a dataset's answer column into a list of reference strings."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            # VQAv2-style: [{"answer": "cat", "answer_confidence": "yes"}, ...]
            if isinstance(item, dict):
                for key in ("answer", "text", "label"):
                    if key in item:
                        out.append(str(item[key]))
                        break
            elif item is not None:
                out.append(str(item))
        return out
    if isinstance(value, dict):
        for key in ("answer", "text", "label"):
            if key in value:
                return [str(value[key])]
        return []
    text = str(value)
    return [text] if text != "" else []


class BenchmarkLoader:
    """Loads benchmark data from various sources."""

    def __init__(self, cache_dir: Path | None = None):
        self._cache_dir = cache_dir or _CACHE_DIR

    def load(
        self,
        manifest: BenchmarkManifest,
        split: str = "validation",
        max_samples: int | None = None,
        shuffle: bool = False,
        seed: int = 42,
    ) -> Iterator[BenchmarkSample]:
        src = manifest.source
        if src.type == "huggingface":
            yield from self._load_hf(manifest, split, max_samples, shuffle, seed)
        elif src.type == "local":
            yield from self._load_local(manifest, split, max_samples, shuffle, seed)
        else:
            raise ValueError(f"Unsupported source type: {src.type}")

    def load_few_shot(self, manifest: BenchmarkManifest) -> list[BenchmarkSample]:
        """Load the few-shot example pool declared by the manifest."""
        cfg = manifest.few_shot
        if not cfg.count:
            return []
        pool = list(self.load(manifest, cfg.source, max_samples=max(cfg.count * 4, cfg.count)))
        if not pool:
            raise ValueError(
                f"few_shot.count={cfg.count} but split '{cfg.source}' yielded no samples"
            )
        if cfg.strategy == "random":
            random.Random(cfg.seed).shuffle(pool)
        return pool[: cfg.count]

    def _load_hf(
        self,
        manifest: BenchmarkManifest,
        split: str,
        max_samples: int | None,
        shuffle: bool,
        seed: int,
    ) -> Iterator[BenchmarkSample]:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets")

        src = manifest.source
        if src.revision == "main":
            logger.warning(
                "benchmark '%s' loads huggingface dataset '%s' at revision 'main' — "
                "a floating pointer, not a reproducible pin. Set source.revision to a "
                "commit SHA or tag in the manifest so results stay reproducible if the "
                "dataset is updated upstream.",
                manifest.name,
                src.path,
            )
        dataset = load_dataset(
            src.path,
            src.subset,
            split=split,
            cache_dir=str(self._cache_dir / "hf"),
            revision=src.revision,
        )

        if shuffle:
            dataset = dataset.shuffle(seed=seed)
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        for idx, row in enumerate(dataset):
            yield self._build_sample(manifest, row, split, idx)

    def _load_local(
        self,
        manifest: BenchmarkManifest,
        split: str,
        max_samples: int | None,
        shuffle: bool = False,
        seed: int = 42,
    ) -> Iterator[BenchmarkSample]:
        data_path = Path(manifest.source.path)
        split_file = data_path / f"{split}.jsonl"
        if not split_file.exists():
            split_file = data_path / f"{split}.json"
        if not split_file.exists():
            raise FileNotFoundError(
                f"No data file for split '{split}' at {data_path}/{split}.jsonl"
            )

        with open(split_file) as f:
            if split_file.suffix == ".jsonl":
                rows = [json.loads(line) for line in f if line.strip()]
            else:
                rows = json.load(f)

        if shuffle:
            random.Random(seed).shuffle(rows)
        if max_samples is not None:
            rows = rows[:max_samples]

        for idx, row in enumerate(rows):
            yield self._build_sample(manifest, row, split, idx, base_dir=data_path)

    def _build_sample(
        self,
        manifest: BenchmarkManifest,
        row: dict,
        split: str,
        idx: int,
        base_dir: Path | None = None,
    ) -> BenchmarkSample:
        fields = manifest.fields
        images = self._extract_images(row, fields.images, base_dir=base_dir)

        max_images = manifest.image_config.max_images
        if len(images) > max_images:
            images = images[:max_images]
        if not images and manifest.image_config.missing_strategy == "error":
            raise ValueError(
                f"sample {idx} of '{manifest.name}' has no usable image in columns "
                f"{fields.images} (image_config.missing_strategy is 'error')"
            )

        text_fields: dict[str, Any] = {}
        if fields.question:
            text_fields["question"] = row.get(fields.question, "")
        if fields.choices:
            text_fields["choices"] = row.get(fields.choices)
        if fields.context:
            text_fields["context"] = row.get(fields.context)
        for alias, column in fields.text_fields.items():
            text_fields[alias] = row.get(column)

        references: list[str] = []
        if fields.answer:
            references.extend(_coerce_references(row.get(fields.answer)))
        if fields.answers:
            references.extend(_coerce_references(row.get(fields.answers)))

        metadata: dict[str, Any] = {}
        if fields.subject and fields.subject in row:
            metadata["subject"] = row[fields.subject]
        if fields.difficulty and fields.difficulty in row:
            metadata["difficulty"] = row[fields.difficulty]
        for name in fields.metadata_fields:
            if name in row:
                metadata[name] = row[name]

        return BenchmarkSample(
            sample_id=str(row.get("id", f"{manifest.name}_{split}_{idx}")),
            images=images,
            text_fields=text_fields,
            references=references,
            metadata=metadata,
        )

    def _extract_images(
        self, row: dict, image_fields: list[str], base_dir: Path | None = None
    ) -> list:
        import io

        from PIL import Image

        images = []
        for field_name in image_fields:
            val = row.get(field_name)
            if val is None:
                continue
            if isinstance(val, Image.Image):
                images.append(val)
            elif isinstance(val, (str, Path)):
                p = Path(val)
                if not p.is_absolute() and base_dir is not None:
                    p = base_dir / p
                if p.exists():
                    images.append(Image.open(p))
            elif isinstance(val, bytes):
                images.append(Image.open(io.BytesIO(val)))
            elif isinstance(val, dict) and "bytes" in val and val["bytes"]:
                images.append(Image.open(io.BytesIO(val["bytes"])))
        return images
