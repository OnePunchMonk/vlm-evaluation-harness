"""Tests for image_config.placement == 'interleaved' (issue #36): the prompt
template's {image_N} placeholders should resolve into an ordered sequence of
text/image PromptParts, not just validate and fall back to concatenation.
"""

from __future__ import annotations

import pytest

from vlm_evaluation_harness.benchmarks.schema import BenchmarkManifest
from vlm_evaluation_harness.prompt.formatter import PromptFormatError, PromptFormatter


def _manifest(prompt_template: str, placement: str = "interleaved") -> BenchmarkManifest:
    return BenchmarkManifest.from_dict(
        {
            "name": "InterleaveTest",
            "source": {"type": "local", "path": "."},
            "splits": [{"name": "validation", "scorable": True}],
            "fields": {"question": "question", "answer": "answer"},
            "prompt_template": prompt_template,
            "image_config": {"max_images": 2, "placement": placement},
        }
    )


def test_interleaved_produces_ordered_parts_matching_placeholders():
    manifest = _manifest("Compare {image_1} to {image_2}. {question}")
    formatter = PromptFormatter()
    result = formatter.format(
        manifest,
        sample_images=["a.png", "b.png"],
        text_fields={"question": "Which is bigger?"},
    )

    kinds = [p.kind for p in result.parts]
    assert kinds == ["text", "image", "text", "image", "text"]
    assert result.parts[1].image == "a.png"
    assert result.parts[3].image == "b.png"
    assert result.parts[0].text.strip() == "Compare"
    assert "Which is bigger?" in result.parts[-1].text


def test_interleaved_text_has_markers_stripped_for_fallback_adapters():
    """Adapters that don't consume `.parts` still get sane flattened text,
    with images attached separately via `.images` as before."""
    manifest = _manifest("Compare {image_1} to {image_2}. {question}")
    formatter = PromptFormatter()
    result = formatter.format(
        manifest,
        sample_images=["a.png", "b.png"],
        text_fields={"question": "Which is bigger?"},
    )
    assert "\x00" not in result.text
    assert "Compare" in result.text and "Which is bigger?" in result.text
    assert result.images == ["a.png", "b.png"]


def test_non_interleaved_placement_produces_no_parts():
    manifest = _manifest("{question}", placement="before_text")
    formatter = PromptFormatter()
    result = formatter.format(
        manifest, sample_images=["a.png"], text_fields={"question": "What is this?"}
    )
    assert result.parts == []


def test_interleaved_placement_with_no_images_and_no_placeholders_produces_no_parts():
    """A sample with zero images and a template that doesn't reference any
    {image_N} is fine (nothing to interleave) -- placement="interleaved"
    only requires the placeholder to exist when there's something to point
    it at."""
    manifest = _manifest("{question}")
    formatter = PromptFormatter()
    result = formatter.format(
        manifest, sample_images=[], text_fields={"question": "Which is bigger?"}
    )
    assert result.parts == []


def test_interleaved_template_referencing_images_with_zero_images_raises():
    """A template that references {image_N} but the sample has zero images
    is a genuine mismatch -- must fail loud, like the out-of-range case."""
    manifest = _manifest("Compare {image_1} to {image_2}. {question}")
    formatter = PromptFormatter()
    with pytest.raises(PromptFormatError):
        formatter.format(manifest, sample_images=[], text_fields={"question": "Which is bigger?"})


def test_interleaved_missing_placeholder_for_available_image_raises():
    """A manifest with placement=interleaved but no {image_N} at all would
    otherwise silently drop every image -- must fail loud instead."""
    manifest = _manifest("{question}")
    formatter = PromptFormatter()
    with pytest.raises(PromptFormatError, match="no \\{image_N\\} placeholder"):
        formatter.format(
            manifest, sample_images=["a.png"], text_fields={"question": "What is this?"}
        )


def test_interleaved_referencing_out_of_range_image_raises():
    manifest = _manifest("Compare {image_1} to {image_2}. {question}")
    formatter = PromptFormatter()
    with pytest.raises(PromptFormatError):
        formatter.format(
            manifest,
            sample_images=["a.png"],  # only 1 image, template wants image_2
            text_fields={"question": "Which is bigger?"},
        )
