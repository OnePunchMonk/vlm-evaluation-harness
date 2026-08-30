"""Prompt formatting: resolve templates for a benchmark sample."""

from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from vlm_evaluation_harness.adapters.base import ConversationTurn, PromptPart
from vlm_evaluation_harness.benchmarks.schema import BenchmarkManifest


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
    # Populated only when few_shot.mode == "multi_turn": one user/assistant
    # turn pair per few-shot example, meant to be passed as `history=` to
    # VLMAdapter.generate() instead of being flattened into `text`.
    history: list[ConversationTurn] = field(default_factory=list)
    # Populated only when image_config.placement == "interleaved": the same
    # images and text as `images`/`text`, but ordered exactly as the prompt
    # template's `{image_1}`, `{image_2}`, ... placeholders positioned them.
    # See PromptPart. Empty otherwise.
    parts: list[PromptPart] = field(default_factory=list)


_CHOICE_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
# Placeholder substituted for {image_N} during rendering, then located again
# in the rendered text to split it into ordered PromptParts. NUL-delimited
# so it can't collide with anything a real prompt template would contain.
_IMAGE_MARKER = "\x00IMG{}\x00"
_IMAGE_MARKER_RE = re.compile(r"\x00IMG(\d+)\x00")


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

        multi_turn = few_shot_examples and manifest.few_shot.mode == "multi_turn"
        variables["few_shot_examples"] = (
            self._render_few_shot(few_shot_examples, manifest)
            if few_shot_examples and not multi_turn
            else ""
        )

        history = (
            self._render_few_shot_turns(few_shot_examples, manifest) if multi_turn else []
        )

        interleaved = manifest.image_config.placement == "interleaved" and bool(sample_images)
        if interleaved:
            for i in range(1, len(sample_images) + 1):
                variables[f"image_{i}"] = _IMAGE_MARKER.format(i)

        rendered = self._render(template or manifest.prompt_template, variables)

        parts: list[PromptPart] = []
        text = rendered.strip()
        if interleaved:
            parts = self._split_into_parts(rendered, sample_images)
            if not any(p.kind == "image" for p in parts):
                raise PromptFormatError(
                    f"benchmark '{manifest.name}' sets image_config.placement='interleaved' "
                    "and has images, but its prompt_template contains no {image_N} "
                    "placeholder -- every image would be silently dropped."
                )
            text = _IMAGE_MARKER_RE.sub("", rendered)
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()

        return FormattedPrompt(
            text=text,
            images=sample_images,
            system=manifest.system_prompt,
            raw_fields=variables,
            history=history,
            parts=parts,
        )

    def _split_into_parts(self, rendered: str, images: list[Image.Image | str]) -> list[PromptPart]:
        """Split text containing `_IMAGE_MARKER`s into ordered PromptParts."""
        parts: list[PromptPart] = []
        cursor = 0
        for m in _IMAGE_MARKER_RE.finditer(rendered):
            segment = rendered[cursor : m.start()]
            if segment.strip():
                parts.append(PromptPart(kind="text", text=segment))
            index = int(m.group(1))
            if 1 <= index <= len(images):
                parts.append(PromptPart(kind="image", image=images[index - 1]))
            cursor = m.end()
        tail = rendered[cursor:]
        if tail.strip():
            parts.append(PromptPart(kind="text", text=tail))
        return parts

    def _render(self, template: str, variables: dict[str, Any]) -> str:
        required = {
            name.split(".")[0].split("[")[0]
            for _, name, _, _ in string.Formatter().parse(template)
            if name
        }
        # few_shot_examples is legitimately "" whenever few_shot.count == 0
        # or mode == "multi_turn" — an empty value there is not a missing
        # dataset field, unlike every other placeholder.
        optional_when_empty = {"few_shot_examples"}
        missing = sorted(
            n
            for n in required
            if n not in optional_when_empty and variables.get(n) in (None, "")
        )
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

    def _render_few_shot_turns(
        self, examples: list[dict], manifest: BenchmarkManifest
    ) -> list[ConversationTurn]:
        """Render few-shot examples as alternating user/assistant turns.

        Each example becomes a (user question [+ its images], assistant
        answer) turn pair, for adapters whose chat template expects real
        conversation turns rather than one flattened block of text — the
        `mode: concatenated` behavior above.
        """
        turns: list[ConversationTurn] = []
        for ex in examples:
            variables = dict(ex)
            choices = self.parse_choices(variables.get("choices"))
            if choices is not None:
                variables["choices"] = choices
                variables["formatted_choices"] = self.format_choices(choices)
            variables["few_shot_examples"] = ""
            text = self._render(manifest.prompt_template, variables).strip()
            images = ex.get("images", [])
            turns.append(ConversationTurn(role="user", text=text, images=images))
            turns.append(ConversationTurn(role="assistant", text=str(ex.get("answer", ""))))
        return turns

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
