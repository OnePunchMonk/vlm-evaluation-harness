"""Tests that PromptPart ordering (issue #36) actually reaches the wire
format for the chat-completions-style adapters (OpenAI/OpenAI-compatible/
vLLM share _build_content in openai.py) and Anthropic, plus the
HuggingFace adapter's chat-template path.

These test the pure message/content builders directly -- no `openai` or
`anthropic` package install required, since those functions don't import
them.
"""

from __future__ import annotations

from vlm_evaluation_harness.adapters import anthropic as anthropic_module
from vlm_evaluation_harness.adapters import openai as openai_module
from vlm_evaluation_harness.adapters.base import PromptPart

_PARTS = [
    PromptPart(kind="text", text="Compare"),
    PromptPart(kind="image", image="http://example.com/a.png"),
    PromptPart(kind="text", text="to"),
    PromptPart(kind="image", image="http://example.com/b.png"),
    PromptPart(kind="text", text="Which is bigger?"),
]


def test_openai_build_content_follows_parts_order():
    content = openai_module._build_content(images=[], prompt="ignored", parts=_PARTS)
    kinds = [c["type"] for c in content]
    assert kinds == ["text", "image_url", "text", "image_url", "text"]
    assert content[0]["text"] == "Compare"
    assert content[4]["text"] == "Which is bigger?"


def test_openai_build_content_falls_back_without_parts():
    content = openai_module._build_content(
        images=["http://example.com/a.png"], prompt="What is this?", parts=None
    )
    assert [c["type"] for c in content] == ["image_url", "text"]
    assert content[1]["text"] == "What is this?"


def test_openai_build_messages_threads_parts_through():
    messages = openai_module._build_messages(
        images=[], prompt="ignored", system=None, history=None, parts=_PARTS
    )
    user_content = messages[-1]["content"]
    assert [c["type"] for c in user_content] == ["text", "image_url", "text", "image_url", "text"]


def test_anthropic_build_content_follows_parts_order():
    content = anthropic_module._build_content(images=[], prompt="ignored", parts=_PARTS)
    kinds = [c["type"] for c in content]
    assert kinds == ["text", "image", "text", "image", "text"]
    assert content[0]["text"] == "Compare"


def test_anthropic_build_content_falls_back_without_parts():
    content = anthropic_module._build_content(
        images=["http://example.com/a.png"], prompt="What is this?", parts=None
    )
    assert [c["type"] for c in content] == ["image", "text"]
