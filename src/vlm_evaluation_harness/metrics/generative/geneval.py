"""GenEval-style compositional accuracy, approximated with CLIP zero-shot checks.

The real GenEval benchmark (Ghosh et al., 2023) uses a trained object
detector to verify count/color/position claims. That's a heavy, separate
dependency. This metric swaps in CLIP zero-shot classification as a
lightweight stand-in: it's real signal (not fabricated), but weaker than a
detector, especially for counting. Documented as an approximation; a real
detector can be plugged in later behind the same `compute()` interface.
"""

from __future__ import annotations

from PIL import Image

from vlm_evaluation_harness.metrics.base import MetricResult
from vlm_evaluation_harness.metrics.generative.clip_score import CLIPScorer

_COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white", "gray"]
_SHAPES = ["circle", "square", "triangle"]
_COUNT_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


class GenEvalClipMetric:
    """Checks generated images against structured {count, color, shape} checks."""

    def __init__(self, model_id: str | None = None):
        self._clip = CLIPScorer(model_id)

    def _best_match(self, image: Image.Image, candidates: list[str]) -> str:
        torch = self._clip._torch
        inputs = self._clip._processor(
            text=candidates, images=[image], return_tensors="pt", padding=True
        )
        with torch.no_grad():
            out = self._clip._model(**inputs)
        logits = out.logits_per_image[0]
        return candidates[int(logits.argmax())]

    def compute(
        self,
        images: list[Image.Image],
        checks_list: list[dict | None],
        metadata: list[dict] | None = None,
        sample_ids: list[str] | None = None,
    ) -> MetricResult:
        attr_hits = {"color": 0, "shape": 0, "count": 0}
        attr_total = {"color": 0, "shape": 0, "count": 0}
        ids = sample_ids or [str(i) for i in range(len(images))]
        per_sample: dict[str, float] = {}

        for sample_id, image, checks in zip(ids, images, checks_list):
            checks = checks or {}
            all_ok = True

            if "color" in checks:
                candidates = [f"a photo with the color {c}" for c in _COLORS]
                pred = self._best_match(image, candidates)
                hit = pred.split()[-1] == checks["color"]
                attr_hits["color"] += int(hit)
                attr_total["color"] += 1
                all_ok &= hit

            if "shape" in checks:
                candidates = [f"a photo containing a {s}" for s in _SHAPES]
                pred = self._best_match(image, candidates)
                hit = pred.split()[-1] == checks["shape"]
                attr_hits["shape"] += int(hit)
                attr_total["shape"] += 1
                all_ok &= hit

            if "count" in checks:
                words = list(_COUNT_WORDS.values())
                candidates = [f"a photo of {w} objects" for w in words]
                pred = self._best_match(image, candidates)
                pred_word = pred.split()[2]
                pred_count = {v: k for k, v in _COUNT_WORDS.items()}[pred_word]
                hit = pred_count == checks["count"]
                attr_hits["count"] += int(hit)
                attr_total["count"] += 1
                all_ok &= hit

            per_sample[sample_id] = float(all_ok)

        from vlm_evaluation_harness.metrics.base import NAN

        n = len(images)
        breakdown = {
            k: (attr_hits[k] / attr_total[k]) for k in attr_hits if attr_total[k]
        }
        return MetricResult(
            metric_name="geneval_clip",
            value=sum(per_sample.values()) / n if n else NAN,
            breakdown=breakdown,
            n_samples=n,
            n_scored=len(per_sample),
            per_sample=per_sample,
        )
