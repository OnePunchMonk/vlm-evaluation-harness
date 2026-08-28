"""Composable post-extraction answer filters.

`normalize_answer` (normalizer.py) applies one named normalization mode.
Filters are a complementary, smaller mechanism: an ordered list of simple
string transforms a manifest can chain after extraction+normalization, for
cases a single `normalize` mode doesn't cover (e.g. "lowercase AND strip
punctuation" without pulling in `vqa` mode's article-stripping and
number-word conversion too).
"""

from __future__ import annotations

import re
from collections.abc import Callable

FilterFn = Callable[[str], str]


def _lowercase(text: str) -> str:
    return text.lower()


def _strip_whitespace(text: str) -> str:
    return text.strip()


def _strip_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text)


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


FILTERS: dict[str, FilterFn] = {
    "lowercase": _lowercase,
    "strip_whitespace": _strip_whitespace,
    "strip_punctuation": _strip_punctuation,
    "collapse_whitespace": _collapse_whitespace,
}


def apply_filters(text: str, filter_names: list[str]) -> str:
    """Apply named filters to `text` in order. Unknown names raise ValueError."""
    for name in filter_names:
        if name not in FILTERS:
            raise ValueError(f"unknown answer filter {name!r} (known: {sorted(FILTERS)})")
        text = FILTERS[name](text)
    return text
