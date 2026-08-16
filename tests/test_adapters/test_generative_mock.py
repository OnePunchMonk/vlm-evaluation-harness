"""Tests for the offline text-to-image mock adapter."""

from vlm_harness.adapters.generative.mock import MockT2IAdapter


def test_renders_requested_count_color_shape():
    adapter = MockT2IAdapter(model_id="clean-model")
    response = adapter.generate("a photo of three red circles on a plain background")
    assert response.metadata["rendered_count"] == 3
    assert response.metadata["rendered_color"] == "red"
    assert response.metadata["rendered_shape"] == "circle"
    assert response.metadata["corrupted"] is False
    assert response.image.size == (256, 256)


def test_defaults_when_prompt_has_no_attributes():
    adapter = MockT2IAdapter(model_id="clean-model")
    response = adapter.generate("an abstract composition")
    assert response.metadata["rendered_count"] == 1
    assert response.metadata["rendered_shape"] == "circle"


def test_degraded_model_id_corrupts_a_meaningful_fraction():
    adapter = MockT2IAdapter(model_id="my-model-degraded")
    prompts = [f"a photo of two blue squares, variant {i}" for i in range(50)]
    corrupted = sum(
        adapter.generate(p).metadata["corrupted"] for p in prompts
    )
    assert corrupted > 0  # error_rate=0.6 should trip on some of 50 prompts


def test_clean_model_never_corrupts():
    adapter = MockT2IAdapter(model_id="clean-model")
    prompts = [f"a photo of one green triangle, variant {i}" for i in range(50)]
    assert all(not adapter.generate(p).metadata["corrupted"] for p in prompts)


def test_deterministic_for_same_model_and_prompt():
    adapter = MockT2IAdapter(model_id="v2-repro")
    r1 = adapter.generate("a photo of four yellow squares")
    r2 = adapter.generate("a photo of four yellow squares")
    assert r1.metadata == r2.metadata


def test_cost_is_zero():
    adapter = MockT2IAdapter()
    assert adapter.cost_per_image_usd == 0.0
