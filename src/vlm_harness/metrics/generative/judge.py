"""LLM/VLM-as-judge scoring for generated images.

Reuses the discriminative VLMAdapter interface as the judge — the judge
receives the generated image plus a rubric and returns a score. Works
offline via `judge_model: mock:judge-v1`.

Two things a judge metric must not do, both of which this one previously
did: score an unparseable judge response as 0.0 (which is indistinguishable
from a genuinely terrible image), and report a mean without saying how often
the judge failed. Parse failures are now excluded from the mean and surfaced
as `judge_failure_rate`.
"""

from __future__ import annotations

from PIL import Image

from vlm_harness.adapters.registry import get_adapter
from vlm_harness.benchmarks.schema import AnswerExtractionConfig
from vlm_harness.metrics.base import NAN, MetricResult
from vlm_harness.parsing.extractor import AnswerExtractor


class LLMJudgeMetric:
    """Scores prompt-image alignment/quality via an LLM/VLM judge and a rubric."""

    def __init__(self, judge_model: str, rubric: str, max_score: float = 10.0):
        self._adapter = get_adapter(judge_model)
        self._judge_model = judge_model
        self._rubric = rubric
        self._max_score = max_score
        self._extractor = AnswerExtractor()

    def compute(
        self,
        prompts: list[str],
        images: list[Image.Image],
        metadata: list[dict] | None = None,
        sample_ids: list[str] | None = None,
    ) -> MetricResult:
        ids = sample_ids or [str(i) for i in range(len(prompts))]
        per_sample: dict[str, float] = {}
        failures = 0

        for sample_id, prompt, image in zip(ids, prompts, images):
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
                failures += 1
                continue
            clamped = min(max(score, 0.0), self._max_score)
            per_sample[sample_id] = clamped / self._max_score if self._max_score else 0.0

        value = sum(per_sample.values()) / len(per_sample) if per_sample else NAN
        n = len(prompts)
        return MetricResult(
            metric_name="llm_judge",
            value=value,
            breakdown={"judge_failure_rate": failures / n if n else NAN},
            n_samples=n,
            n_scored=len(per_sample),
            per_sample=per_sample,
            metadata={"max_score": self._max_score, "judge_model": self._judge_model},
        )


class VQAScoreMetric:
    """VQAScore: prompt adherence as P(yes) under a VQA model.

    Asks a VLM "Does this figure show '{prompt}'?" and uses the probability
    it assigns to "Yes". Introduced by Lin et al. (2024), it correlates with
    human judgement substantially better than CLIPScore on compositional
    prompts — CLIP's bag-of-words text encoder is largely blind to
    attribute binding and relations, which is exactly what compositional
    T2I benchmarks probe.

    When the judge adapter exposes log-probabilities (`score_choices`) the
    score is the calibrated probability. Otherwise it degrades to a binary
    yes/no answer, which is noisier — the degradation is recorded in the
    result metadata rather than hidden.
    """

    _QUESTION = 'Does this figure show "{prompt}"? Answer Yes or No.'

    def __init__(self, judge_model: str):
        self._adapter = get_adapter(judge_model)
        self._judge_model = judge_model

    def compute(
        self,
        prompts: list[str],
        images: list[Image.Image],
        metadata: list[dict] | None = None,
        sample_ids: list[str] | None = None,
    ) -> MetricResult:
        from vlm_harness.adapters.base import supports_choice_scoring

        ids = sample_ids or [str(i) for i in range(len(prompts))]
        probabilistic = supports_choice_scoring(self._adapter)
        per_sample: dict[str, float] = {}

        for sample_id, prompt, image in zip(ids, prompts, images):
            question = self._QUESTION.format(prompt=prompt)
            if probabilistic:
                scores = self._adapter.score_choices(
                    images=[image], prompt=question, choices=["Yes", "No"]
                )
                per_sample[sample_id] = self._softmax_first(scores.logprobs)
            else:
                response = self._adapter.generate(
                    images=[image], prompt=question, max_tokens=8, temperature=0.0
                )
                per_sample[sample_id] = (
                    1.0 if response.text.strip().lower().startswith("y") else 0.0
                )

        value = sum(per_sample.values()) / len(per_sample) if per_sample else NAN
        return MetricResult(
            metric_name="vqa_score",
            value=value,
            n_samples=len(prompts),
            n_scored=len(per_sample),
            per_sample=per_sample,
            metadata={
                "judge_model": self._judge_model,
                "mode": "logprob" if probabilistic else "binary_fallback",
            },
        )

    @staticmethod
    def _softmax_first(logprobs: list[float]) -> float:
        import math

        top = max(logprobs)
        exps = [math.exp(lp - top) for lp in logprobs]
        return exps[0] / sum(exps)
