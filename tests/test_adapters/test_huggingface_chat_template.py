"""Tests for HuggingFaceAdapter._render_prompt: chat-template usage and the
fix for history (multi-turn few-shot) being silently dropped.

No torch/transformers needed: `_render_prompt` only touches `self._processor`,
so a fake processor is injected via `object.__new__`, same pattern as
test_huggingface_batching.py.
"""

from __future__ import annotations

from vlm_evaluation_harness.adapters.base import ConversationTurn, PromptPart
from vlm_evaluation_harness.adapters.huggingface import HuggingFaceAdapter


def _make_adapter(processor) -> HuggingFaceAdapter:
    adapter = object.__new__(HuggingFaceAdapter)
    adapter._model_id = "fake-model"
    adapter._processor = processor
    return adapter


class _FakeProcessorWithTemplate:
    """Stands in for a processor whose underlying tokenizer has a chat
    template (true for most instruction-tuned VLM checkpoints)."""

    chat_template = "{% for m in messages %}...{% endfor %}"

    def __init__(self):
        self.captured_messages = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        assert add_generation_prompt is True
        self.captured_messages = messages
        return "RENDERED:" + repr(messages)


class _FakeProcessorNoTemplate:
    chat_template = None


def test_uses_chat_template_when_available_and_includes_system():
    processor = _FakeProcessorWithTemplate()
    adapter = _make_adapter(processor)

    text, images = adapter._render_prompt(
        images=[], prompt="What color?", system="Be terse.", history=None
    )

    assert text.startswith("RENDERED:")
    messages = processor.captured_messages
    assert messages[0] == {"role": "system", "content": [{"type": "text", "text": "Be terse."}]}
    assert messages[-1]["role"] == "user"
    assert {"type": "text", "text": "What color?"} in messages[-1]["content"]
    assert images == []


def test_history_is_included_via_chat_template_not_silently_dropped():
    processor = _FakeProcessorWithTemplate()
    adapter = _make_adapter(processor)

    history = [
        ConversationTurn(role="user", text="3+3?", images=["fewshot.png"]),
        ConversationTurn(role="assistant", text="6", images=[]),
    ]
    text, images = adapter._render_prompt(
        images=["real.png"], prompt="1+1?", system=None, history=history
    )

    messages = processor.captured_messages
    # system + 2 history turns + final user turn
    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert {"type": "text", "text": "3+3?"} in messages[0]["content"]
    assert {"type": "image"} in messages[0]["content"]
    assert messages[1] == {"role": "assistant", "content": [{"type": "text", "text": "6"}]}
    assert messages[2]["role"] == "user"
    assert {"type": "text", "text": "1+1?"} in messages[2]["content"]
    # images flattened in turn order: few-shot image, then the real image
    assert images == ["fewshot.png", "real.png"]


def test_falls_back_to_concatenation_when_no_chat_template_but_still_keeps_history():
    processor = _FakeProcessorNoTemplate()
    adapter = _make_adapter(processor)

    history = [
        ConversationTurn(role="user", text="3+3?"),
        ConversationTurn(role="assistant", text="6"),
    ]
    text, images = adapter._render_prompt(
        images=["real.png"], prompt="1+1?", system="Be terse.", history=history
    )

    # Before this fix, `history` was accepted as a parameter and never used
    # at all in the fallback path -- few-shot examples vanished silently.
    assert "Be terse." in text
    assert "3+3?" in text
    assert "6" in text
    assert text.endswith("1+1?")
    assert images == ["real.png"]


def test_falls_back_cleanly_with_no_system_and_no_history():
    processor = _FakeProcessorNoTemplate()
    adapter = _make_adapter(processor)

    text, images = adapter._render_prompt(images=[], prompt="1+1?", system=None, history=None)

    assert text == "1+1?"
    assert images == []


def test_prompt_parts_override_content_order_in_chat_template():
    """When `prompt_parts` is given, the user message's content must follow
    that order instead of the default images-then-text -- and `images`
    should be ignored in favor of the images embedded in the parts."""
    processor = _FakeProcessorWithTemplate()
    adapter = _make_adapter(processor)

    prompt_parts = [
        PromptPart(kind="text", text="Compare"),
        PromptPart(kind="image", image="a.png"),
        PromptPart(kind="text", text="to"),
        PromptPart(kind="image", image="b.png"),
        PromptPart(kind="text", text="Which is bigger?"),
    ]

    text, images = adapter._render_prompt(
        images=["ignored.png"],
        prompt="ignored prompt",
        system=None,
        history=None,
        prompt_parts=prompt_parts,
    )

    user_content = processor.captured_messages[-1]["content"]
    assert [c["type"] for c in user_content] == ["text", "image", "text", "image", "text"]
    assert user_content[0]["text"] == "Compare"
    assert user_content[4]["text"] == "Which is bigger?"
    assert images == ["a.png", "b.png"]
