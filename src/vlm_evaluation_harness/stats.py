"""Significance testing and confidence intervals for benchmark comparisons.

A benchmark score is an estimate from a finite sample, and the difference
between two estimates is mostly noise until proven otherwise. On 50 samples
a four-point accuracy swing is entirely unremarkable; reporting it as a
"MEDIUM regression" — as this harness previously did, using fixed absolute
thresholds — manufactures alarm.

Two tools, both dependency-free (numpy only, already a core dependency):

* `mcnemar` — the correct test for two models scored on the *same* samples.
  It ignores samples both models get right or both get wrong and asks only
  whether the disagreements are lopsided. This needs per-sample scores,
  which is why runs now persist them.
* `bootstrap_ci` / `bootstrap_delta_ci` — percentile intervals for when only
  aggregate scores are available, or for reporting a metric's own precision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class McNemarResult:
    """Outcome of a paired comparison over per-sample correctness."""

    n_paired: int          # samples scored by both runs
    n_improved: int        # baseline wrong -> current right
    n_regressed: int       # baseline right -> current wrong
    n_discordant: int      # n_improved + n_regressed
    statistic: float       # chi-square statistic (continuity-corrected)
    p_value: float
    method: str            # "exact_binomial" | "chi_square"

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def summary(self) -> str:
        return (
            f"{self.n_regressed} regressed / {self.n_improved} improved "
            f"of {self.n_paired} paired (p={self.p_value:.4f})"
        )


def mcnemar(
    baseline: dict[str, float],
    current: dict[str, float],
    threshold: float = 0.5,
    exact_below: int = 25,
) -> McNemarResult:
    """Paired McNemar test over per-sample scores keyed by sample id.

    Scores are binarized at `threshold` (per-sample metrics in this harness
    are already in [0, 1], and most are 0/1). Only sample ids present in both
    runs are used. The exact binomial test is used when the discordant count
    is small, where the chi-square approximation is unreliable.
    """
    shared = sorted(set(baseline) & set(current))
    n_improved = 0
    n_regressed = 0
    for sample_id in shared:
        was_right = baseline[sample_id] >= threshold
        is_right = current[sample_id] >= threshold
        if was_right and not is_right:
            n_regressed += 1
        elif is_right and not was_right:
            n_improved += 1

    discordant = n_improved + n_regressed
    if discordant == 0:
        return McNemarResult(
            n_paired=len(shared),
            n_improved=0,
            n_regressed=0,
            n_discordant=0,
            statistic=0.0,
            p_value=1.0,
            method="exact_binomial",
        )

    if discordant < exact_below:
        smaller = min(n_improved, n_regressed)
        tail = sum(math.comb(discordant, i) for i in range(smaller + 1)) * (0.5**discordant)
        p_value = min(1.0, 2 * tail)
        statistic = float(smaller)
        method = "exact_binomial"
    else:
        # Edwards' continuity correction.
        statistic = (abs(n_regressed - n_improved) - 1) ** 2 / discordant
        p_value = math.erfc(math.sqrt(statistic / 2))
        method = "chi_square"

    return McNemarResult(
        n_paired=len(shared),
        n_improved=n_improved,
        n_regressed=n_regressed,
        n_discordant=discordant,
        statistic=statistic,
        p_value=p_value,
        method=method,
    )


def bootstrap_ci(
    values: list[float],
    confidence: float = 0.95,
    n_resamples: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of `values`."""
    if not values:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        return (values[0], values[0])
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = rng.choice(arr, size=(n_resamples, arr.size), replace=True).mean(axis=1)
    alpha = (1 - confidence) / 2
    return (
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1 - alpha)),
    )


def bootstrap_delta_ci(
    baseline: dict[str, float],
    current: dict[str, float],
    confidence: float = 0.95,
    n_resamples: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the *paired* difference current - baseline.

    Resampling sample ids (rather than the two runs independently) preserves
    the pairing, which is what makes the interval tight enough to be useful.
    """
    shared = sorted(set(baseline) & set(current))
    if not shared:
        return (float("nan"), float("nan"))
    diffs = np.asarray([current[i] - baseline[i] for i in shared], dtype=float)
    if diffs.size == 1:
        return (float(diffs[0]), float(diffs[0]))
    rng = np.random.default_rng(seed)
    means = rng.choice(diffs, size=(n_resamples, diffs.size), replace=True).mean(axis=1)
    alpha = (1 - confidence) / 2
    return (float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha)))


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion — well behaved near 0 and 1."""
    if n == 0:
        return (float("nan"), float("nan"))
    # Two-sided normal quantile via the inverse error function.
    z = math.sqrt(2) * _erfinv(confidence)
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _erfinv(x: float) -> float:
    """Inverse error function (Winitzki's approximation, refined by Newton)."""
    a = 0.147
    ln1mx2 = math.log(1 - x * x)
    term = 2 / (math.pi * a) + ln1mx2 / 2
    y = math.copysign(math.sqrt(math.sqrt(term * term - ln1mx2 / a) - term), x)
    for _ in range(3):
        err = math.erf(y) - x
        y -= err / (2 / math.sqrt(math.pi) * math.exp(-y * y))
    return y
