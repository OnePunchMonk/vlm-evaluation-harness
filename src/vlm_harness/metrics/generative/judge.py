"""LLM/VLM-as-judge scoring for generated images.

Reuses the existing discriminative VLMAdapter interface as the judge model —
the judge just receives the generated image plus a rubric and returns a
score, same shape as any other image-in/text-out call. Works offline via
`judge_model: mock:judge-v1`.
"""

from __future__ import annotations

from PIL import Image

from vlm_harness.adapters.registry import get_adapter
from vlm_harness.benchmarks.schema import AnswerExtractionConfig
from vlm_harness.metrics.base import MetricResult
from vlm_harness.parsing.extractor import AnswerExtractor


class LLMJudgeMetric:
    """Scores prompt-image alignment/quality via an LLM/VLM judge and a rubric."""

    def __init__(self, judge_model: str, rubric: str, max_score: float = 10.0):
        self._adapter = get_adapter(judge_model)
        self._rubric = rubric
        self._max_score = max_score
        self._extractor = AnswerExtractor()

    def compute(
        self, prompts: list[str], images: list[Image.Image], metadata: list[dict] | None = None
    ) -> MetricResult:
        scores: list[float] = []
        for prompt, image in zip(prompts, images):
            judge_prompt = (
                f'Prompt given to the image generator: "{prompt}"\n\n'
                f"{self._rubric}\n\n"
                f"Respond with only a single number from 1 to {int(self._max_score)}."
            )
            response = self._adapter.generate(
                images=[image], prompt=judge_prompt, max_tokens=16, temperature=0.0
            )
            extraction = self._extractor.extract(
                response.text, AnswerExtractionConfig(strategy="number")
            )
            try:
                score = float(extraction.normalized)
            except ValueError:
                score = 0.0
            scores.append(min(max(score, 0.0), self._max_score))

        avg = sum(scores) / len(scores) if scores else 0.0
        return MetricResult(
            metric_name="llm_judge",
            value=avg / self._max_score if self._max_score else 0.0,
            n_samples=len(scores),
            metadata={"raw_scores": scores, "max_score": self._max_score},
        )
