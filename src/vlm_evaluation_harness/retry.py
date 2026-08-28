"""Retry policy shared by every network-backed adapter.

`tenacity` was already a declared dependency but nothing imported it, so a
single 429 killed a run that might be hours deep. Retries are bounded and
only cover transient failures — a malformed request should fail immediately
rather than be attempted eight times.
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

# Substrings identifying errors that are worth retrying. Matched against the
# exception's class name and message, so this works across provider SDKs
# without importing any of them.
_TRANSIENT_MARKERS = (
    "rate limit",
    "ratelimit",
    "429",
    "500",
    "502",
    "503",
    "504",
    "overloaded",
    "timeout",
    "timed out",
    "connection",
    "temporarily unavailable",
    "service unavailable",
    "apiconnectionerror",
    "internalservererror",
)


def is_transient(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def default_max_attempts() -> int:
    try:
        return max(1, int(os.environ.get("VLM_HARNESS_MAX_RETRIES", "5")))
    except ValueError:
        return 5


def with_retries(
    fn: Callable[[], T],
    max_attempts: int | None = None,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call `fn`, retrying transient failures with exponential backoff and jitter."""
    attempts = max_attempts if max_attempts is not None else default_max_attempts()
    last_exc: BaseException | None = None

    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - re-raised below
            if not is_transient(exc) or attempt == attempts - 1:
                raise
            last_exc = exc
            delay = min(base_delay * (2**attempt), max_delay)
            sleep(delay * (0.5 + random.random() / 2))

    raise RuntimeError("unreachable") from last_exc
