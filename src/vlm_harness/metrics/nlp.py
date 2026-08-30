"""NLP metrics: F1, ANLS, BLEU, ROUGE."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from vlm_harness.metrics.base import MetricResult


class F1Metric:
    """Token-level F1 score (used in QA tasks like SQuAD)."""

    def compute(
        self, predictions: list[str], references: list[str], metadata: list[dict]
    ) -> MetricResult:
        scores = [self._token_f1(p, r) for p, r in zip(predictions, references)]
        return MetricResult(
            metric_name="f1",
            value=sum(scores) / len(scores) if scores else 0.0,
            n_samples=len(scores),
        )

    def _token_f1(self, prediction: str, ground_truth: str) -> float:
        pred_tokens = self._tokenize(prediction)
        gt_tokens = self._tokenize(ground_truth)
        if not pred_tokens or not gt_tokens:
            return int(pred_tokens == gt_tokens)
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        return 2 * precision * recall / (precision + recall)

    def _tokenize(self, text: str) -> list[str]:
        text = unicodedata.normalize("NFD", text).lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return text.split()


class ANLSMetric:
    """Average Normalized Levenshtein Similarity (DocVQA metric)."""

    def compute(
        self, predictions: list[str], references: list[str], metadata: list[dict]
    ) -> MetricResult:
        scores = [self._anls(p, r) for p, r in zip(predictions, references)]
        return MetricResult(
            metric_name="anls",
            value=sum(scores) / len(scores) if scores else 0.0,
            n_samples=len(scores),
        )

    def _anls(self, prediction: str, reference: str, threshold: float = 0.5) -> float:
        dist = self._edit_distance(prediction.lower(), reference.lower())
        max_len = max(len(prediction), len(reference))
        if max_len == 0:
            return 1.0
        nls = 1.0 - dist / max_len
        return nls if nls >= threshold else 0.0

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
    """Corpus BLEU score (requires sacrebleu)."""

    def compute(
        self, predictions: list[str], references: list[str], metadata: list[dict]
    ) -> MetricResult:
        try:
            import sacrebleu

            result = sacrebleu.corpus_bleu(predictions, [references])
            return MetricResult(
                metric_name="bleu",
                value=result.score / 100.0,
                n_samples=len(predictions),
                metadata={"bleu_score": result.score},
            )
        except ImportError:
            # Fallback: simple unigram precision
            scores = []
            for pred, ref in zip(predictions, references):
                pred_tok = pred.lower().split()
                ref_tok = ref.lower().split()
                if not pred_tok:
                    scores.append(0.0)
                    continue
                common = Counter(pred_tok) & Counter(ref_tok)
                scores.append(sum(common.values()) / len(pred_tok))
            return MetricResult(
                metric_name="bleu_approx",
                value=sum(scores) / len(scores) if scores else 0.0,
                n_samples=len(scores),
            )


class RougeMetric:
    """ROUGE-L score."""

    def compute(
        self, predictions: list[str], references: list[str], metadata: list[dict]
    ) -> MetricResult:
        try:
            from rouge_score import rouge_scorer

            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            scores = [
                scorer.score(ref, pred)["rougeL"].fmeasure
                for pred, ref in zip(predictions, references)
            ]
        except ImportError:
            scores = [self._rouge_l(p, r) for p, r in zip(predictions, references)]
        return MetricResult(
            metric_name="rouge_l",
            value=sum(scores) / len(scores) if scores else 0.0,
            n_samples=len(scores),
        )

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
                dp[i][j] = (
                    dp[i - 1][j - 1] + 1
                    if x[i - 1] == y[j - 1]
                    else max(dp[i - 1][j], dp[i][j - 1])
                )
        return dp[m][n]
