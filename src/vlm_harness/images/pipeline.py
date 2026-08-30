"""Image normalization pipeline."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

from PIL import Image


@dataclass
class ImagePipelineConfig:
    max_resolution: tuple[int, int] = (2048, 2048)
    min_resolution: tuple[int, int] = (32, 32)
    output_format: str = "PNG"
    color_space: str = "RGB"
    hash_algorithm: str = "sha256"


@dataclass
class ProcessedImage:
    image: Image.Image
    original_size: tuple[int, int]
    final_size: tuple[int, int]
    hash: str
    format: str


class ImagePipeline:
    """Normalizes images for consistent model input."""

    def __init__(self, config: ImagePipelineConfig | None = None):
        self.config = config or ImagePipelineConfig()

    def process(self, image: Image.Image | str) -> ProcessedImage:
        if isinstance(image, str):
            from pathlib import Path

            image = Image.open(Path(image))

        original_size = image.size

        # Convert color space
        if self.config.color_space == "RGB" and image.mode != "RGB":
            image = image.convert("RGB")

        # Enforce minimum resolution
        w, h = image.size
        min_w, min_h = self.config.min_resolution
        if w < min_w or h < min_h:
            scale = max(min_w / w, min_h / h)
            image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

        # Enforce maximum resolution (maintain aspect ratio)
        w, h = image.size
        max_w, max_h = self.config.max_resolution
        if w > max_w or h > max_h:
            scale = min(max_w / w, max_h / h)
            image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

        final_size = image.size
        img_hash = self._hash(image)

        return ProcessedImage(
            image=image,
            original_size=original_size,
            final_size=final_size,
            hash=img_hash,
            format=self.config.output_format,
        )

    def process_batch(self, images: list[Image.Image | str]) -> list[ProcessedImage]:
        return [self.process(img) for img in images]

    def _hash(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format=self.config.output_format)
        h = hashlib.new(self.config.hash_algorithm)
        h.update(buf.getvalue())
        return f"{self.config.hash_algorithm}:{h.hexdigest()}"
