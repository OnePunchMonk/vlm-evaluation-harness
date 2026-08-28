"""Deterministic, offline VLM adapter for tests, demos, and CI.

It has no access to ground truth — it only ever sees the rendered prompt
(and images, which it ignores) — so it exercises the real discriminative
pipeline (prompt formatting -> generate -> answer extraction -> scoring)
without needing any API key or network access. Also doubles as an offline
LLM-judge backend for generative metrics (see metrics/generative/judge.py).
"""

from __future__ import annotations

import hashlib
import re
import time

from vlm_evaluation_harness.adapters.base import ChoiceScores, ConversationTurn, VLMResponse


class MockAdapter:
    """Offline stand-in for an API-based VLM."""

    def __init__(self, model_id: str = "demo"):
        self._model_id = model_id
        # Records the last `system` instruction seen by generate(), so tests
        # can verify it was actually threaded through from CLI -> config ->
        # prompt formatter -> adapter call, not just accepted and dropped.
        self.last_system: str | None = None

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
    def max_resolution(self) -> tuple[int, int] | None:
        return None

    @property
    def cost_per_million_input_tokens(self) -> float | None:
        return 0.0

    @property
    def cost_per_million_output_tokens(self) -> float | None:
        return 0.0

    def generate(
        self,
        images: list,
        prompt: str,
        system: str | None = None,
        history: list[ConversationTurn] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> VLMResponse:
        self.last_system = system
        t0 = time.perf_counter()
        text = self._respond(prompt)
        latency_ms = (time.perf_counter() - t0) * 1000
        return VLMResponse(
            text=text,
            input_tokens=len(prompt.split()),
            output_tokens=len(text.split()),
            latency_ms=latency_ms,
            model_id=self._model_id,
        )

    @property
    def supports_choice_scoring(self) -> bool:
        return True

    def score_choices(
        self,
        images: list,
        prompt: str,
        choices: list[str],
        system: str | None = None,
    ) -> ChoiceScores:
        """Deterministic pseudo log-probabilities, so the loglikelihood
        scoring path is exercised offline exactly like the generate path."""
        t0 = time.perf_counter()
        logprobs = []
        for choice in choices:
            digest = hashlib.md5(f"{self._model_id}::{prompt}::{choice}".encode()).hexdigest()
            h = int(digest[:8], 16)
            logprobs.append(-(h % 1000) / 100.0)
        n_tokens = [max(1, len(c.split())) for c in choices]
        return ChoiceScores(
            logprobs=logprobs,
            logprobs_per_token=[lp / n for lp, n in zip(logprobs, n_tokens)],
            latency_ms=(time.perf_counter() - t0) * 1000,
            model_id=self._model_id,
        )

    def _respond(self, prompt: str) -> str:
        h = int(hashlib.md5(f"{self._model_id}::{prompt}".encode()).hexdigest()[:8], 16)

        # Numeric rubric prompts (LLM-as-judge)
        m = re.search(r"from 1 to (\d+)", prompt)
        if m:
            max_score = int(m.group(1))
            return str(1 + h % max_score)

        # Yes/No prompts
        if re.search(r"answer with [\"']?yes[\"']? or [\"']?no[\"']?", prompt, re.IGNORECASE):
            return "Yes" if h % 2 == 0 else "No"

        # Multiple-choice prompts: pick from the lettered options actually offered
        letters = sorted(set(re.findall(r"^([A-E])\.\s", prompt, re.MULTILINE)))
        if letters:
            return letters[h % len(letters)]

        return "unknown"
