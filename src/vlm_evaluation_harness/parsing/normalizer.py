"""Answer normalization utilities."""

from __future__ import annotations

import re
import unicodedata


def normalize_answer(text: str, mode: str = "strip") -> str:
    """
    Normalize an answer string.

    Modes:
        strip       - strip whitespace only
        uppercase   - strip + uppercase
        lowercase   - strip + lowercase
        vqa         - VQA-style: lowercase, remove articles/punctuation
        none        - no normalization
    """
    if mode == "none":
        return text
    if mode == "strip":
        return text.strip()
    if mode == "uppercase":
        return text.strip().upper()
    if mode == "lowercase":
        return text.strip().lower()
    if mode == "vqa":
        return _vqa_normalize(text)
    return text.strip()


def _vqa_normalize(text: str) -> str:
    """VQA evaluation normalization (matches official VQA eval script)."""
    # Unicode normalization
    text = unicodedata.normalize("NFD", text)
    text = text.lower().strip()

    # Remove articles (also at start/end of string)
    text = re.sub(r"\ba\b\s*", "", text)
    text = re.sub(r"\ban\b\s*", "", text)
    text = re.sub(r"\bthe\b\s*", "", text)

    # Normalize punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Number words to digits
    _WORD_TO_NUM = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10",
    }
    for word, digit in _WORD_TO_NUM.items():
        text = re.sub(rf"\b{word}\b", digit, text)

    return text
