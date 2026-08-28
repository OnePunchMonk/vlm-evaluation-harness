"""Proof that self-consistency (sample-N + majority-vote) recovers accuracy
lost to a noisy decoding process, without needing a live API.

`PixelColorAdapter` genuinely reads the fixture image (unlike MockAdapter,
which ignores pixels) and answers correctly with fixed probability per call,
independently across calls — a stand-in for a real model's sampling noise
at temperature > 0. Self-consistency should recover accuracy on top of that
per-call noise, exactly as it does for real LLMs (Wang et al., 2022) and as
recent (2026) compositional-reasoning work argues single-shot decoding
underestimates.
"""

from __future__ import annotations

import random
import re

from vlm_harness.adapters.base import VLMResponse
from vlm_harness.engine.runner import EvalConfig, EvalRunner

_COLOR_TO_LETTER = {
    "red": "A",
    "blue": "B",
    "green": "C",
    "yellow": "D",
    "purple": "E",
    "orange": "F",
}
# Matches this fixture's actual (muted) rendered pixel values, not naive
# CSS color names — the fixture draws yellow closer to orange than to pure
# (255,255,0), and nearest-CSS-name matching would misclassify it.
_NAMED_RGB = {
    "red": (220, 40, 40),
    "blue": (40, 90, 220),
    "green": (40, 180, 70),
    "yellow": (230, 200, 30),
    "purple": (140, 60, 190),
    "orange": (240, 130, 30),
}


class PixelColorAdapter:
    """Reads the dominant pixel color and answers correctly with a fixed
    per-call probability, wrong (random other option) otherwise. Noise is
    seeded per call so results are reproducible."""

    def __init__(self, model_id: str = "pixel-color", correct_prob: float = 0.55):
        self._model_id = model_id
        self._correct_prob = correct_prob
        self._counter = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def supports_multi_image(self) -> bool:
        return True

    @property
    def supports_video(self) -> bool:
        return False

    @property
    def max_resolution(self):
        return None

    @property
    def cost_per_million_input_tokens(self):
        return 0.0

    @property
    def cost_per_million_output_tokens(self):
        return 0.0

    def generate(
        self,
        images,
        prompt,
        system=None,
        history=None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> VLMResponse:
        mean = tuple(int(x) for x in images[0].resize((1, 1)).getpixel((0, 0))[:3])
        true_color = min(
            _NAMED_RGB, key=lambda c: sum((a - b) ** 2 for a, b in zip(_NAMED_RGB[c], mean))
        )
        correct_letter = _COLOR_TO_LETTER[true_color]
        letters = sorted(set(re.findall(r"^([A-F])\.\s", prompt, re.MULTILINE)))

        rng = random.Random(f"noise-{self._counter}")
        self._counter += 1
        if rng.random() < self._correct_prob:
            letter = correct_letter
        else:
            others = [letter for letter in letters if letter != correct_letter] or letters
            letter = rng.choice(others)

        return VLMResponse(
            text=letter, input_tokens=len(prompt.split()), output_tokens=1, model_id=self._model_id
        )


def _run(n: int) -> float:
    adapter = PixelColorAdapter()
    runner = EvalRunner(adapter)
    config = EvalConfig(
        model_spec="mock:pixel-color",
        benchmark="demo_mc",
        split="validation",
        use_cache=False,
        temperature=0.9,
        self_consistency_n=n,
    )
    result = runner.run(config)
    accuracy = next(m for m in result.metrics if m.metric_name == "accuracy")
    return accuracy.value


def test_self_consistency_recovers_accuracy_from_noisy_decoding():
    single_shot = _run(1)
    majority_of_5 = _run(5)
    assert majority_of_5 > single_shot


def test_self_consistency_n1_is_unchanged_single_call_path():
    """N=1 must be exactly today's behavior — self-consistency is opt-in."""
    adapter = PixelColorAdapter(correct_prob=1.0)
    runner = EvalRunner(adapter)
    config = EvalConfig(
        model_spec="mock:pixel-color",
        benchmark="demo_mc",
        split="validation",
        use_cache=False,
        self_consistency_n=1,
    )
    result = runner.run(config)
    accuracy = next(m for m in result.metrics if m.metric_name == "accuracy")
    assert accuracy.value == 1.0
