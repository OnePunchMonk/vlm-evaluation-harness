"""Tests for concatenated vs. multi_turn few-shot prompt rendering."""

from __future__ import annotations

from vlm_evaluation_harness.benchmarks.schema import BenchmarkManifest, FewShotConfig
from vlm_evaluation_harness.prompt.formatter import PromptFormatter


def _manifest(mode: str) -> BenchmarkManifest:
    return BenchmarkManifest.from_dict(
        {
            "name": "FewShotTest",
            "source": {"type": "local", "path": "."},
            "splits": [{"name": "validation", "scorable": True}],
            "fields": {"question": "question", "answer": "answer"},
            "prompt_template": "{few_shot_examples}{question}",
            "few_shot": {"count": 2, "mode": mode},
        }
    )


_EXAMPLES = [
    {"question": "What color is the sky?", "answer": "blue", "images": ["sky.png"]},
    {"question": "What color is grass?", "answer": "green", "images": ["grass.png"]},
]


def test_concatenated_mode_flattens_into_text():
    manifest = _manifest("concatenated")
    formatter = PromptFormatter()
    result = formatter.format(
        manifest,
        sample_images=["q.png"],
        text_fields={"question": "What color is a banana?"},
        few_shot_examples=_EXAMPLES,
    )
    assert "What color is the sky?" in result.text
    assert "blue" in result.text
    assert "What color is a banana?" in result.text
    assert result.history == []


def test_multi_turn_mode_produces_conversation_turns():
    manifest = _manifest("multi_turn")
    formatter = PromptFormatter()
    result = formatter.format(
        manifest,
        sample_images=["q.png"],
        text_fields={"question": "What color is a banana?"},
        few_shot_examples=_EXAMPLES,
    )
    # Final question text does not contain the flattened few-shot blob.
    assert "sky" not in result.text
    assert result.text == "What color is a banana?"

    assert len(result.history) == 4  # 2 examples * (user, assistant)
    assert result.history[0].role == "user"
    assert result.history[0].text == "What color is the sky?"
    assert result.history[0].images == ["sky.png"]
    assert result.history[1].role == "assistant"
    assert result.history[1].text == "blue"
    assert result.history[2].role == "user"
    assert result.history[2].text == "What color is grass?"
    assert result.history[3].role == "assistant"
    assert result.history[3].text == "green"


def test_multi_turn_mode_with_no_examples_has_empty_history():
    manifest = _manifest("multi_turn")
    formatter = PromptFormatter()
    result = formatter.format(
        manifest,
        sample_images=["q.png"],
        text_fields={"question": "Q?"},
        few_shot_examples=None,
    )
    assert result.history == []


def test_few_shot_mode_defaults_to_concatenated():
    assert FewShotConfig().mode == "concatenated"
