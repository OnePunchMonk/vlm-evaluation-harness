"""NLP metrics: F1, ANLS, BLEU, ROUGE — all scored against multiple references."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from vlm_evaluation_harness.metrics.base import NAN, MetricResult, ScoredSample, aggregate


def _tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFD", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


class F1Metric:
    """Token-level F1, taking the best score over the sample's references."""

    def compute(self, samples: list[ScoredSample]) -> MetricResult:
        return aggregate("f1", samples, self.score)

    def score(self, sample: ScoredSample) -> float:
        return max(
            (self._token_f1(sample.prediction, ref) for ref in sample.references), default=0.0
        )

    def _token_f1(self, prediction: str, ground_truth: str) -> float:
        pred_tokens = _tokenize(prediction)
        gt_tokens = _tokenize(ground_truth)
        if not pred_tokens or not gt_tokens:
            return float(pred_tokens == gt_tokens)
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        return 2 * precision * recall / (precision + recall)


class ANLSMetric:
    """Average Normalized Levenshtein Similarity (DocVQA metric).

    The official metric takes the maximum similarity over all provided
    ground-truth strings, thresholded at 0.5.
    """

    def __init__(self, threshold: float = 0.5):
        self._threshold = threshold

    def compute(self, samples: list[ScoredSample]) -> MetricResult:
        return aggregate("anls", samples, self.score)

    def score(self, sample: ScoredSample) -> float:
        return max((self._anls(sample.prediction, ref) for ref in sample.references), default=0.0)

    def _anls(self, prediction: str, reference: str) -> float:
        pred, ref = prediction.lower().strip(), reference.lower().strip()
        max_len = max(len(pred), len(ref))
        if max_len == 0:
            return 1.0
        nls = 1.0 - self._edit_distance(pred, ref) / max_len
        return nls if nls >= self._threshold else 0.0

    def _edit_distance(self, s1: str, s2: str) -> int:
        m, n = len(s1), len(s2)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                if s1[i - 1] == s2[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = temp
        return dp[n]


class BLEUMetric:
    """Corpus BLEU. Uses sacrebleu's native multi-reference support when available."""

    def compute(self, samples: list[ScoredSample]) -> MetricResult:
        scorable = [s for s in samples if s.has_reference]
        if not scorable:
            return MetricResult(
                metric_name="bleu", value=NAN, n_samples=len(samples), n_scored=0
            )

        predictions = [s.prediction for s in scorable]
        # sacrebleu wants reference streams: one list per reference slot.
        width = max(len(s.references) for s in scorable)
        ref_streams = [
            [s.references[i] if i < len(s.references) else s.references[-1] for s in scorable]
            for i in range(width)
        ]
        try:
            import sacrebleu

            result = sacrebleu.corpus_bleu(predictions, ref_streams)
            return MetricResult(
                metric_name="bleu",
                value=result.score / 100.0,
                n_samples=len(samples),
                n_scored=len(scorable),
                metadata={"bleu_score": result.score, "backend": "sacrebleu"},
            )
        except ImportError:
            result = aggregate("bleu_approx", samples, self._unigram_precision)
            result.metadata["backend"] = "fallback_unigram_precision"
            return result

    def _unigram_precision(self, sample: ScoredSample) -> float:
        pred_tok = sample.prediction.lower().split()
        if not pred_tok:
            return 0.0
        best = 0.0
        for ref in sample.references:
            common = Counter(pred_tok) & Counter(ref.lower().split())
            best = max(best, sum(common.values()) / len(pred_tok))
        return best


class RougeMetric:
    """ROUGE-L F-measure, best over the sample's references."""

    def compute(self, samples: list[ScoredSample]) -> MetricResult:
        scorer = self._build_scorer()

        def score(sample: ScoredSample) -> float:
            return max((scorer(sample.prediction, r) for r in sample.references), default=0.0)

        return aggregate("rouge_l", samples, score)

    def _build_scorer(self):
        try:
            from rouge_score import rouge_scorer

            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            return lambda pred, ref: scorer.score(ref, pred)["rougeL"].fmeasure
        except ImportError:
            return self._rouge_l

    def _rouge_l(self, prediction: str, reference: str) -> float:
        pred_tok = prediction.lower().split()
        ref_tok = reference.lower().split()
        if not pred_tok or not ref_tok:
            return 0.0
        lcs_len = self._lcs(pred_tok, ref_tok)
        precision = lcs_len / len(pred_tok)
        recall = lcs_len / len(ref_tok)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _lcs(self, x: list, y: list) -> int:
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = dp[i - 1][j - 1] + 1 if x[i - 1] == y[j - 1] else max(
                    dp[i - 1][j], dp[i][j - 1]
                )
        return dp[m][n]
