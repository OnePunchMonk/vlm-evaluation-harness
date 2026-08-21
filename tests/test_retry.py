"""Tests for the shared retry policy."""

import pytest

from vlm_harness.retry import is_transient, with_retries


def test_is_transient_detects_rate_limit():
    assert is_transient(RuntimeError("429 Too Many Requests"))
    assert is_transient(RuntimeError("Rate limit exceeded"))


def test_is_transient_rejects_validation_errors():
    assert not is_transient(ValueError("invalid image format"))


def test_with_retries_succeeds_after_transient_failures():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("503 Service Unavailable")
        return "ok"

    result = with_retries(flaky, max_attempts=5, base_delay=0.001, sleep=lambda s: None)
    assert result == "ok"
    assert calls["n"] == 3


def test_with_retries_reraises_non_transient_immediately():
    calls = {"n": 0}

    def broken():
        calls["n"] += 1
        raise ValueError("bad request")

    with pytest.raises(ValueError):
        with_retries(broken, max_attempts=5, base_delay=0.001, sleep=lambda s: None)
    assert calls["n"] == 1


def test_with_retries_gives_up_after_max_attempts():
    def always_fails():
        raise RuntimeError("500 Internal Server Error")

    with pytest.raises(RuntimeError):
        with_retries(always_fails, max_attempts=3, base_delay=0.001, sleep=lambda s: None)
