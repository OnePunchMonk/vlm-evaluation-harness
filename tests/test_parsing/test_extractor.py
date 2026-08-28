"""Tests for answer extraction."""

from vlm_evaluation_harness.benchmarks.schema import AnswerExtractionConfig
from vlm_evaluation_harness.parsing.extractor import AnswerExtractor
from vlm_evaluation_harness.parsing.normalizer import normalize_answer

extractor = AnswerExtractor()


def cfg(strategy, normalize="strip", pattern=None):
    return AnswerExtractionConfig(strategy=strategy, normalize=normalize, regex_pattern=pattern)


class TestFirstLetterExtraction:
    def test_bare_letter(self):
        r = extractor.extract("B", cfg("first_letter", "uppercase"))
        assert r.normalized == "B"

    def test_letter_with_period(self):
        r = extractor.extract("A. Paris", cfg("first_letter", "uppercase"))
        assert r.normalized == "A"

    def test_answer_is_pattern(self):
        r = extractor.extract("The answer is C.", cfg("first_letter", "uppercase"))
        assert r.normalized == "C"

    def test_verbose_response(self):
        r = extractor.extract(
            "Looking at the options, I believe the correct answer is D because...",
            cfg("first_letter", "uppercase"),
        )
        assert r.normalized == "D"


class TestNumberExtraction:
    def test_integer(self):
        r = extractor.extract("There are 42 birds.", cfg("number"))
        assert r.normalized == "42"

    def test_float(self):
        r = extractor.extract("The value is 3.14.", cfg("number"))
        assert r.normalized == "3.14"

    def test_percentage(self):
        r = extractor.extract("About 75% of the chart.", cfg("number"))
        assert r.normalized == "75%"


class TestYesNoExtraction:
    def test_yes(self):
        r = extractor.extract("Yes, there is a cat.", cfg("yes_no", "lowercase"))
        assert r.normalized == "yes"

    def test_no(self):
        r = extractor.extract("No, I don't see any dogs.", cfg("yes_no", "lowercase"))
        assert r.normalized == "no"

    def test_true_maps_to_yes(self):
        r = extractor.extract("True", cfg("yes_no", "lowercase"))
        assert r.normalized == "yes"


class TestNormalization:
    def test_vqa_removes_articles(self):
        assert normalize_answer("a cat", "vqa") == "cat"
        assert normalize_answer("the dog", "vqa") == "dog"

    def test_vqa_lowercases(self):
        assert normalize_answer("PARIS", "vqa") == "paris"

    def test_number_words(self):
        assert normalize_answer("three", "vqa") == "3"

    def test_uppercase(self):
        assert normalize_answer("hello world", "uppercase") == "HELLO WORLD"

    def test_strip(self):
        assert normalize_answer("  hi  ", "strip") == "hi"
