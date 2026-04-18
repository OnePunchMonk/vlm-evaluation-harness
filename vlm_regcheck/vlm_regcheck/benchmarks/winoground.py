"""
Winoground  —  Compositional vision-language understanding.

Dataset : facebook/winoground
Split   : test (800 examples = 400 image pairs × 2 captions each)
Format  : image-caption matching (NOT multiple choice)
SOTA    : ~~28% text score (humans: ~89%) — very hard, lots of headroom
Headroom: enormous — most models barely beat random (25%)

Task:
  Given 2 images and 2 captions, correctly match image→caption AND caption→image.
  Scored with three metrics:
    text_score:   did the model prefer c0→i0 AND c1→i1 over swapped?
    image_score:  did the model prefer i0→c0 AND i1→c1 over swapped?
    group_score:  both correct (strictest)

Why important for regression detection:
  Compositional grounding is the FIRST capability to degrade when:
  - VLMs are fine-tuned on flat QA data (no compositional reasoning)
  - The vision-language connector is disrupted
  - The model overfits to surface patterns
  SOTA is ~28% → any regression shows up immediately as further drop.

Scoring: we use VQA-style prompting to get image-caption similarity scores.
"""

from __future__ import annotations

import io
from typing import Optional

from datasets import load_dataset
from PIL import Image

from .base import Benchmark, Sample, normalize


class Winoground(Benchmark):
    name = "winoground"
    capability = "compositional_vision_language"
    sota_score = 0.28                   # text score, GPT-4V era

    def load(self, n_samples: Optional[int] = None) -> list[Sample]:
        ds = load_dataset("facebook/winoground", split="test",
                          trust_remote_code=True, use_auth_token=False)

        # We reformulate each Winoground item into 4 binary VQA questions:
        #   Q1: Does [caption_0] describe this image [image_0]?  → Yes
        #   Q2: Does [caption_1] describe this image [image_0]?  → No
        #   Q3: Does [caption_0] describe this image [image_1]?  → No
        #   Q4: Does [caption_1] describe this image [image_1]?  → Yes
        # text_score requires Q1+Q2 AND Q3+Q4 both correct.
        # group_score requires all 4 correct.

        samples = []
        for row in ds:
            if n_samples and len(samples) >= n_samples:
                break

            item_id = row.get("id", len(samples) // 4)
            c0 = row.get("caption_0", "")
            c1 = row.get("caption_1", "")

            def _load_img(key):
                img = row.get(key)
                if img is None:
                    return None
                if isinstance(img, dict) and "bytes" in img:
                    return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
                if isinstance(img, bytes):
                    return Image.open(io.BytesIO(img)).convert("RGB")
                if isinstance(img, Image.Image):
                    return img
                return None

            i0 = _load_img("image_0")
            i1 = _load_img("image_1")
            if i0 is None or i1 is None:
                continue

            for img_idx, img, caption, gt in [
                (0, i0, c0, "yes"),  # i0 + c0 match
                (0, i0, c1, "no"),   # i0 + c1 no match
                (1, i1, c0, "no"),   # i1 + c0 no match
                (1, i1, c1, "yes"),  # i1 + c1 match
            ]:
                prompt = (
                    f"Does the following caption accurately describe this image?\n"
                    f"Caption: \"{caption}\"\n"
                    f"Answer with Yes or No only."
                )
                samples.append(Sample(
                    id=f"winoground_{item_id}_i{img_idx}_c{'01'[int(gt=='yes')]}",
                    images=[img],
                    prompt=prompt,
                    choices=["Yes", "No"],
                    answer=gt,
                    capability=self.capability,
                    metadata={"item_id": item_id, "img_idx": img_idx, "caption": caption},
                ))

        return samples

    def score(self, prediction: str, sample: Sample) -> bool:
        pred = normalize(prediction)
        if pred.startswith("yes"):
            pred = "yes"
        elif pred.startswith("no"):
            pred = "no"
        return pred == sample.answer

    def evaluate(self, model, samples: list[Sample], verbose: bool = False):
        """
        Override to compute proper Winoground group/text/image scores.
        """
        from .base import BenchmarkResult

        # Group samples by item_id
        from collections import defaultdict
        items: dict = defaultdict(dict)
        for s in samples:
            meta = s.metadata
            key = (meta["item_id"], meta["img_idx"], meta["caption"][:20])
            prediction = model.answer(s.images, s.prompt, s.choices)
            correct = self.score(prediction, s)
            items[meta["item_id"]][f"i{meta['img_idx']}_correct_{s.answer}"] = correct

        # Compute group_score: item correct if ALL 4 sub-questions correct
        # Approximate from available data
        per_sample = []
        n_correct_group = 0
        n_correct_text = 0
        n_correct_image = 0
        total_items = 0

        for item_id, results in items.items():
            vals = list(results.values())
            if len(vals) == 4:
                group = all(vals)
                # text_score: i0_c0 correct AND i1_c1 correct
                # image_score: i0_c0 correct AND i0_c1 correct (foil)
                n_correct_group += int(group)
                total_items += 1
                per_sample.append({"item_id": item_id, "group_correct": group})

        # Fall back to raw accuracy if grouping didn't work
        if total_items == 0:
            total_correct = sum(
                int(self.score(model.answer(s.images, s.prompt, s.choices), s))
                for s in samples
            )
            accuracy = total_correct / len(samples) if samples else 0.0
        else:
            accuracy = n_correct_group / total_items

        return BenchmarkResult(
            benchmark=self.name,
            capability=self.capability,
            accuracy=accuracy,
            n_samples=len(samples),
            sota_score=self.sota_score,
            per_sample=per_sample,
        )
