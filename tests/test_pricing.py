"""Tests for config-driven model pricing (pricing.py / pricing.yaml)."""

from __future__ import annotations

import textwrap

from vlm_evaluation_harness import pricing


def test_known_model_returns_configured_price():
    pricing.clear_pricing_cache()
    cost_in, cost_out = pricing.get_pricing("anthropic", "claude-opus-4-6")
    assert cost_in == 15.0
    assert cost_out == 75.0


def test_unknown_provider_or_model_returns_zero():
    pricing.clear_pricing_cache()
    assert pricing.get_pricing("nonexistent-provider", "some-model") == (0.0, 0.0)
    assert pricing.get_pricing("openai", "nonexistent-model") == (0.0, 0.0)


def test_override_file_adds_and_overrides_entries(tmp_path, monkeypatch):
    override = tmp_path / "custom_pricing.yaml"
    override.write_text(
        textwrap.dedent(
            """\
            anthropic:
              claude-opus-4-6: [1.0, 2.0]
            openai:
              my-self-hosted-model: [0.01, 0.02]
            """
        )
    )
    monkeypatch.setenv("VLM_HARNESS_PRICING_FILE", str(override))
    pricing.clear_pricing_cache()
    try:
        # Overridden entry wins over the shipped default.
        assert pricing.get_pricing("anthropic", "claude-opus-4-6") == (1.0, 2.0)
        # A model not in the shipped defaults at all is picked up too.
        assert pricing.get_pricing("openai", "my-self-hosted-model") == (0.01, 0.02)
        # An unrelated shipped default for the same provider is untouched.
        assert pricing.get_pricing("openai", "gpt-4o") == (5.0, 15.0)
    finally:
        monkeypatch.delenv("VLM_HARNESS_PRICING_FILE", raising=False)
        pricing.clear_pricing_cache()


def test_anthropic_adapter_reads_pricing_module():
    pricing.clear_pricing_cache()
    from vlm_evaluation_harness.adapters.anthropic import AnthropicAdapter

    # AnthropicAdapter.__init__ requires the `anthropic` package; only exercise
    # the pricing properties directly against a bare instance to avoid that
    # hard dependency in this test.
    adapter = AnthropicAdapter.__new__(AnthropicAdapter)
    adapter._model_id = "claude-sonnet-4-6-20260115"
    assert adapter.cost_per_million_input_tokens == 3.0
    assert adapter.cost_per_million_output_tokens == 15.0
