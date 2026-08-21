"""Prompt formatting: resolve templates for a benchmark sample."""

from __future__ import annotations

import json
import string
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from vlm_harness.benchmarks.schema import BenchmarkManifest


class PromptFormatError(ValueError):
    """Raised when a prompt template cannot be fully resolved.

    Deliberately fatal. The previous behaviour — falling back to the raw
    template — sent literal `{caption_0}` placeholders to the model and
    scored the result as if it were a real answer.
    """


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
        template: str | None = None,
    ) -> FormattedPrompt:
        variables = dict(text_fields)

        choices = self.parse_choices(variables.get("choices"))
        if choices is not None:
            variables["choices"] = choices
            variables["formatted_choices"] = self.format_choices(choices)

        variables["few_shot_examples"] = (
            self._render_few_shot(few_shot_examples, manifest) if few_shot_examples else ""
        )

        return FormattedPrompt(
            text=self._render(template or manifest.prompt_template, variables).strip(),
            images=sample_images,
            system=manifest.system_prompt,
            raw_fields=variables,
        )

    def _render(self, template: str, variables: dict[str, Any]) -> str:
        required = {
            name.split(".")[0].split("[")[0]
            for _, name, _, _ in string.Formatter().parse(template)
            if name
        }
        missing = sorted(n for n in required if variables.get(n) in (None, ""))
        if missing:
            raise PromptFormatError(
                f"prompt template placeholder(s) {missing} could not be filled from sample "
                f"fields {sorted(variables)}. The dataset column is missing or empty."
            )
        try:
            return template.format(**variables)
        except (KeyError, IndexError, ValueError) as exc:
            raise PromptFormatError(f"could not render prompt template: {exc}") from exc

    def parse_choices(self, choices: Any) -> list[str] | None:
        if choices is None:
            return None
        if isinstance(choices, str):
            try:
                choices = json.loads(choices)
            except json.JSONDecodeError:
                choices = [c.strip() for c in choices.split(",")]
        return [str(c) for c in choices]

    def format_choices(self, choices: list[str]) -> str:
        lines = []
        for i, choice in enumerate(choices):
            letter = _CHOICE_LETTERS[i] if i < len(_CHOICE_LETTERS) else str(i)
            lines.append(f"{letter}. {choice}")
        return "\n".join(lines)

    def _render_few_shot(self, examples: list[dict], manifest: BenchmarkManifest) -> str:
        parts = []
        for ex in examples:
            variables = dict(ex)
            choices = self.parse_choices(variables.get("choices"))
            if choices is not None:
                variables["choices"] = choices
                variables["formatted_choices"] = self.format_choices(choices)
            variables["few_shot_examples"] = ""
            text = self._render(manifest.prompt_template, variables)
            parts.append(f"{text.strip()}\nAnswer: {ex.get('answer', '')}")
        return "\n\n".join(parts) + "\n\n" if parts else ""
