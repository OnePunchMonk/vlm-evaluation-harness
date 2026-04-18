"""
MMVP  —  Multimodal Visual Patterns benchmark.

Dataset : MMVP/MMVP
Split   : test
Format  : multiple-choice (A/B), paired confusable images
SOTA    : ~38% (GPT-4V, 2024) — humans: ~95%
Headroom: enormous — tests fine-grained visual details most models miss

What it tests:
  Visual patterns that CLIP-based models systematically fail at:
  - Orientation (left/right, up/down)
  - Spatial location
  - Color differentiation
  - Quantity counting (small numbers)
  - Text rendering in images
  - Object attributes (size, shape)

Why critical for regression detection:
  This benchmark specifically targets the VISION ENCODER side of VLMs.
  If fine-tuning (especially text-heavy SFT) degrades the vision encoder's
  contribution, MMVP will catch it while VQAv2/TextVQA won't.
  A fine-tuned model that language-side compensates will still fail here.
"""

from __future__ import annotations

import io
from typing import Optional

from datasets import load_dataset
from PIL import Image

from .base import Benchmark, Sample, build_mc_prompt, extract_letter


class MMVP(Benchmark):
    name = "mmvp"
    capability = "fine_grained_visual_perception"
    sota_score = 0.38                   # GPT-4V

    def load(self, n_samples: Optional[int] = None) -> list[Sample]:
        ds = load_dataset("MMVP/MMVP", split="test", trust_remote_code=True)

        samples = []
        for row in ds:
            if n_samples and len(samples) >= n_samples:
                break

            img = row.get("image")
            if img is None:
                # Try index-based image field
                img = row.get("img")
            if img is None:
                continue
            if isinstance(img, dict) and "bytes" in img:
                img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
            elif isinstance(img, bytes):
                img = Image.open(io.BytesIO(img)).convert("RGB")
            elif not isinstance(img, Image.Image):
                continue

            question = row.get("question", "")
            options = row.get("options", [])
            if isinstance(options, str):
                import ast
                try:
                    options = ast.literal_eval(options)
                except Exception:
                    options = [options]

            if not options:
                options = [row.get("option1", ""), row.get("option2", "")]
                options = [o for o in options if o]

            if not options:
                continue

            prompt = build_mc_prompt(question, options)
            answer = str(row.get("correct_answer", row.get("answer", "A"))).strip().upper()
            if answer not in "ABCDE":
                # Try to map option text to letter
                for i, opt in enumerate(options):
                    if answer.lower() in opt.lower() or opt.lower() in answer.lower():
                        answer = "ABCDE"[i]
                        break
                else:
                    answer = "A"

            samples.append(Sample(
                id=f"mmvp_{row.get('index', len(samples))}",
                images=[img],
                prompt=prompt,
                choices=options,
                answer=answer,
                capability=self.capability,
                metadata={"visual_pattern": row.get("visual_pattern", "")},
            ))

        return samples

    def score(self, prediction: str, sample: Sample) -> bool:
        return extract_letter(prediction) == sample.answer.upper()
