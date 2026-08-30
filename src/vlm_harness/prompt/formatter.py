"""Prompt formatting: resolve templates for a benchmark sample."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from vlm_harness.benchmarks.schema import BenchmarkManifest


@dataclass
class FormattedPrompt:
    """The fully resolved prompt ready to send to a model."""

    text: str
    images: list[Image.Image | str]
    system: str | None = None
    raw_fields: dict[str, Any] = field(default_factory=dict)


_CHOICE_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class PromptFormatter:
    """Resolves benchmark prompt templates against sample data."""

    def format(
        self,
        manifest: BenchmarkManifest,
        sample_images: list[Image.Image | str],
        text_fields: dict[str, Any],
        few_shot_examples: list[dict] | None = None,
    ) -> FormattedPrompt:
        variables = dict(text_fields)

        # Format choices if present
        if variables.get("choices") is not None:
            choices = variables["choices"]
            if isinstance(choices, str):
                try:
                    choices = json.loads(choices)
                except json.JSONDecodeError:
                    choices = [c.strip() for c in choices.split(",")]
            variables["choices"] = choices
            variables["formatted_choices"] = self._format_choices(choices)

        # Few-shot block
        variables["few_shot_examples"] = (
            self._render_few_shot(few_shot_examples, manifest) if few_shot_examples else ""
        )

        # Render template
        try:
            text = manifest.prompt_template.format_map(_SafeDict(variables))
        except (KeyError, ValueError):
            text = manifest.prompt_template  # fallback: use template as-is

        return FormattedPrompt(
            text=text.strip(),
            images=sample_images,
            system=manifest.system_prompt,
            raw_fields=variables,
        )

    def _format_choices(self, choices: list[str]) -> str:
        lines = []
        for i, choice in enumerate(choices):
            letter = _CHOICE_LETTERS[i] if i < len(_CHOICE_LETTERS) else str(i)
            lines.append(f"{letter}. {choice}")
        return "\n".join(lines)

    def _render_few_shot(self, examples: list[dict], manifest: BenchmarkManifest) -> str:
        parts = []
        for ex in examples:
            text = manifest.prompt_template.format_map(_SafeDict(ex))
            answer = ex.get("answer", "")
            parts.append(f"{text.strip()}\nAnswer: {answer}")
        return "\n\n".join(parts) + "\n\n" if parts else ""


class _SafeDict(dict):
    """dict subclass that returns '{key}' for missing keys instead of raising."""

    def __missing__(self, key: str) -> str:
        return f"{{{key}}}"
