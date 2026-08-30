"""Dataset loading and caching for benchmarks."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from vlm_harness.benchmarks.schema import BenchmarkManifest

_CACHE_DIR = Path.home() / ".vlm-harness" / "cache"


class BenchmarkSample:
    """A single sample from a benchmark dataset."""

    __slots__ = ("sample_id", "images", "text_fields", "answer", "metadata")

    def __init__(
        self,
        sample_id: str,
        images: list,
        text_fields: dict[str, Any],
        answer: str | None,
        metadata: dict[str, Any],
    ):
        self.sample_id = sample_id
        self.images = images
        self.text_fields = text_fields
        self.answer = answer
        self.metadata = metadata


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
            yield from self._load_local(manifest, split, max_samples)
        else:
            raise ValueError(f"Unsupported source type: {src.type}")

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
        dataset = load_dataset(
            src.path,
            src.subset,
            split=split,
            trust_remote_code=True,
            cache_dir=str(self._cache_dir / "hf"),
            revision=src.revision,
        )

        if shuffle:
            dataset = dataset.shuffle(seed=seed)
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        fields = manifest.fields
        image_field_names = (
            fields.image_fields if hasattr(fields, "image_fields") else fields.images
        )
        for idx, row in enumerate(dataset):
            images = self._extract_images(row, image_field_names)
            text_fields = {
                "question": row.get(fields.question, ""),
                "choices": row.get(fields.choices, None) if fields.choices else None,
                "context": row.get(fields.context, None) if fields.context else None,
            }
            answer = str(row.get(fields.answer, "")) if fields.answer else None
            metadata = {}
            if fields.subject and fields.subject in row:
                metadata["subject"] = row[fields.subject]
            if fields.difficulty and fields.difficulty in row:
                metadata["difficulty"] = row[fields.difficulty]
            for name in fields.metadata_fields:
                if name in row:
                    metadata[name] = row[name]

            yield BenchmarkSample(
                sample_id=f"{manifest.name}_{split}_{idx}",
                images=images,
                text_fields=text_fields,
                answer=answer,
                metadata=metadata,
            )

    def _load_local(
        self,
        manifest: BenchmarkManifest,
        split: str,
        max_samples: int | None,
    ) -> Iterator[BenchmarkSample]:
        data_path = Path(manifest.source.path)
        split_file = data_path / f"{split}.jsonl"
        if not split_file.exists():
            split_file = data_path / f"{split}.json"
        if not split_file.exists():
            raise FileNotFoundError(f"No data file found at {split_file}")

        fields = manifest.fields
        with open(split_file) as f:
            if split_file.suffix == ".jsonl":
                rows = [json.loads(line) for line in f if line.strip()]
            else:
                rows = json.load(f)

        for idx, row in enumerate(rows[:max_samples]):
            images = self._extract_images(row, fields.images, base_dir=data_path)
            text_fields = {
                "question": row.get(fields.question, ""),
                "choices": row.get(fields.choices) if fields.choices else None,
                "context": row.get(fields.context) if fields.context else None,
            }
            answer = str(row.get(fields.answer, "")) if fields.answer else None
            metadata = {}
            if fields.subject and fields.subject in row:
                metadata["subject"] = row[fields.subject]
            if fields.difficulty and fields.difficulty in row:
                metadata["difficulty"] = row[fields.difficulty]
            for name in fields.metadata_fields:
                if name in row:
                    metadata[name] = row[name]

            yield BenchmarkSample(
                sample_id=row.get("id", f"{manifest.name}_{split}_{idx}"),
                images=images,
                text_fields=text_fields,
                answer=answer,
                metadata=metadata,
            )

    def _extract_images(
        self, row: dict, image_fields: list[str], base_dir: Path | None = None
    ) -> list:
        images = []
        for field_name in image_fields:
            val = row.get(field_name)
            if val is None:
                continue
            # HuggingFace datasets return PIL images directly
            from PIL import Image

            if isinstance(val, Image.Image):
                images.append(val)
            elif isinstance(val, (str, Path)):
                p = Path(val)
                if not p.is_absolute() and base_dir is not None:
                    p = base_dir / p
                if p.exists():
                    images.append(Image.open(p))
            elif isinstance(val, bytes):
                import io

                images.append(Image.open(io.BytesIO(val)))
        return images
