"""Tests for the offline discriminative mock adapter."""

from vlm_harness.adapters.mock import MockAdapter


def test_answers_multiple_choice_with_an_offered_letter():
    adapter = MockAdapter(model_id="test-v1")
    prompt = (
        "What color?\n\nOptions:\nA. Red\nB. Blue\nC. Green\n\nAnswer with the option letter only."
    )
    response = adapter.generate(images=[], prompt=prompt)
    assert response.text in {"A", "B", "C"}


def test_deterministic_for_same_model_and_prompt():
    adapter = MockAdapter(model_id="test-v1")
    prompt = "Options:\nA. Yes\nB. No\n"
    r1 = adapter.generate(images=[], prompt=prompt)
    r2 = adapter.generate(images=[], prompt=prompt)
    assert r1.text == r2.text


def test_yes_no_prompt():
    adapter = MockAdapter(model_id="test-v1")
    prompt = 'Is there a dog?\n\nAnswer with "Yes" or "No" only.'
    response = adapter.generate(images=[], prompt=prompt)
    assert response.text in {"Yes", "No"}


def test_numeric_rubric_prompt_stays_in_range():
    adapter = MockAdapter(model_id="judge-v1")
    prompt = "Rate this 1-10.\n\nRespond with only a single number from 1 to 10."
    response = adapter.generate(images=[], prompt=prompt)
    assert response.text.isdigit()
    assert 1 <= int(response.text) <= 10


def test_free_form_fallback():
    adapter = MockAdapter(model_id="test-v1")
    response = adapter.generate(images=[], prompt="Describe this image.")
    assert response.text == "unknown"


def test_cost_and_metadata_are_zero_and_default():
    adapter = MockAdapter()
    assert adapter.cost_per_million_input_tokens == 0.0
    assert adapter.cost_per_million_output_tokens == 0.0
    assert adapter.model_id == "demo"
