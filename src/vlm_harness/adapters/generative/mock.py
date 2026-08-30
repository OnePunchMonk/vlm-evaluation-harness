"""Deterministic, offline text-to-image adapter for tests, demos, and CI.

Unlike the discriminative MockAdapter (which can't see ground truth and just
hashes its way to a pseudo-random answer), this adapter can legitimately
*render* what the prompt asks for: it parses "{count} {color} {shape}s" out
of the prompt and draws exactly that. This makes compositional metrics
(GenEval-style attribute checks) meaningfully gradable without a real model.

To make regression tracking demonstrable offline, the error rate is derived
from the model_id: ids containing "degraded"/"regressed"/"broken" corrupt a
fraction of renders (wrong count/color/shape), and "v2" corrupts a smaller
fraction. This lets a demo script show a genuine, reproducible accuracy drop
between two "model versions" without needing a real T2I backend.
"""

from __future__ import annotations

import hashlib
import re
import time

from PIL import Image, ImageDraw

from vlm_harness.adapters.generative.base import T2IResponse

_COLOR_MAP: dict[str, tuple[int, int, int]] = {
    "red": (220, 40, 40),
    "blue": (40, 90, 220),
    "green": (40, 180, 70),
    "yellow": (230, 200, 30),
    "purple": (140, 60, 190),
    "orange": (240, 130, 30),
    "black": (20, 20, 20),
    "white": (245, 245, 245),
}
_COLOR_NAMES = list(_COLOR_MAP)
_SHAPES = ("circle", "square", "triangle")
_NUM_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
}


class MockT2IAdapter:
    """Offline stand-in for a text-to-image backend. Draws what it's told to."""

    def __init__(self, model_id: str = "demo", error_rate: float | None = None):
        self._model_id = model_id
        self._error_rate = (
            error_rate if error_rate is not None else self._default_error_rate(model_id)
        )

    @staticmethod
    def _default_error_rate(model_id: str) -> float:
        low = model_id.lower()
        if any(k in low for k in ("degraded", "broken", "regressed")):
            return 0.6
        if "v2" in low:
            return 0.25
        return 0.0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def cost_per_image_usd(self) -> float | None:
        return 0.0

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        seed: int | None = None,
        width: int = 256,
        height: int = 256,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 30,
    ) -> T2IResponse:
        t0 = time.perf_counter()

        count, color_name, shape = self._parse(prompt)
        roll = self._roll(prompt)
        corrupted = roll < self._error_rate
        if corrupted:
            count, color_name, shape = self._corrupt(count, color_name, shape, roll)

        rgb = _COLOR_MAP.get(color_name, (128, 128, 128))
        image = Image.new("RGB", (width, height), color=(250, 250, 250))
        draw = ImageDraw.Draw(image)
        self._draw_shapes(draw, shape, rgb, count, width, height)

        latency_ms = (time.perf_counter() - t0) * 1000
        return T2IResponse(
            image=image,
            latency_ms=latency_ms,
            model_id=self._model_id,
            cost_usd=0.0,
            seed=seed,
            metadata={
                "rendered_count": count,
                "rendered_color": color_name,
                "rendered_shape": shape,
                "corrupted": corrupted,
            },
        )

    # ── prompt parsing ──────────────────────────────────────────────────

    def _parse(self, prompt: str) -> tuple[int, str, str]:
        low = prompt.lower()

        count = 1
        m = re.search(r"\b(\d+)\b", low)
        if m:
            count = int(m.group(1))
        else:
            for word, n in _NUM_WORDS.items():
                if re.search(rf"\b{word}\b", low):
                    count = n
                    break

        color_name = next((c for c in _COLOR_NAMES if c in low), "gray")
        shape = next((s for s in _SHAPES if s in low), "circle")
        return count, color_name, shape

    def _roll(self, prompt: str) -> float:
        h = hashlib.md5(f"{self._model_id}::{prompt}".encode()).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    def _corrupt(
        self, count: int, color_name: str, shape: str, roll: float
    ) -> tuple[int, str, str]:
        """Deterministically corrupt one attribute based on the roll value."""
        bucket = roll % (1.0 / 3.0) * 3.0  # re-spread within [0, 1) into thirds
        if bucket < 1 / 3:
            count = max(1, count + (1 if roll < self._error_rate / 2 else -1))
        elif bucket < 2 / 3:
            others = [c for c in _COLOR_NAMES if c != color_name]
            color_name = others[int(roll * 1000) % len(others)]
        else:
            others = [s for s in _SHAPES if s != shape]
            shape = others[int(roll * 1000) % len(others)]
        return count, color_name, shape

    # ── rendering ────────────────────────────────────────────────────────

    def _draw_shapes(
        self,
        draw: ImageDraw.ImageDraw,
        shape: str,
        rgb: tuple[int, int, int],
        count: int,
        width: int,
        height: int,
    ) -> None:
        count = max(1, min(count, 8))
        cols = min(count, 4)
        rows = (count + cols - 1) // cols
        cell_w, cell_h = width / cols, height / rows
        size = min(cell_w, cell_h) * 0.6

        for i in range(count):
            cx = (i % cols + 0.5) * cell_w
            cy = (i // cols + 0.5) * cell_h
            bbox = (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)
            if shape == "circle":
                draw.ellipse(bbox, fill=rgb)
            elif shape == "square":
                draw.rectangle(bbox, fill=rgb)
            else:  # triangle
                x0, y0, x1, y1 = bbox
                draw.polygon([((x0 + x1) / 2, y0), (x1, y1), (x0, y1)], fill=rgb)
