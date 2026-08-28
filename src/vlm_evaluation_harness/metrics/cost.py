"""Cost and latency tracking."""

from __future__ import annotations

from dataclasses import dataclass

from vlm_evaluation_harness.adapters.base import VLMResponse


@dataclass
class CostSummary:
    total_cost_usd: float
    cost_per_sample_usd: float
    total_input_tokens: int
    total_output_tokens: int
    avg_input_tokens: float
    avg_output_tokens: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    throughput_samples_per_min: float
    n_samples: int


class CostTracker:
    """Accumulates token usage and latency across eval calls."""

    def __init__(
        self,
        cost_per_million_input: float | None = None,
        cost_per_million_output: float | None = None,
    ):
        self._cost_in = cost_per_million_input
        self._cost_out = cost_per_million_output
        self._responses: list[VLMResponse] = []

    def record(self, response: VLMResponse) -> None:
        self._responses.append(response)

    def summary(self) -> CostSummary:
        if not self._responses:
            return CostSummary(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        n = len(self._responses)
        total_in = sum(r.input_tokens for r in self._responses)
        total_out = sum(r.output_tokens for r in self._responses)
        latencies = sorted(r.latency_ms for r in self._responses)

        cost = 0.0
        if self._cost_in is not None:
            cost += (total_in / 1_000_000) * self._cost_in
        if self._cost_out is not None:
            cost += (total_out / 1_000_000) * self._cost_out

        total_time_s = sum(r.latency_ms for r in self._responses) / 1000
        throughput = (n / total_time_s) * 60 if total_time_s > 0 else 0

        return CostSummary(
            total_cost_usd=cost,
            cost_per_sample_usd=cost / n,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            avg_input_tokens=total_in / n,
            avg_output_tokens=total_out / n,
            latency_p50_ms=self._percentile(latencies, 50),
            latency_p95_ms=self._percentile(latencies, 95),
            latency_p99_ms=self._percentile(latencies, 99),
            throughput_samples_per_min=throughput,
            n_samples=n,
        )

    def _percentile(self, sorted_values: list[float], pct: int) -> float:
        if not sorted_values:
            return 0.0
        idx = int(len(sorted_values) * pct / 100)
        idx = min(idx, len(sorted_values) - 1)
        return sorted_values[idx]


class GenCostTracker:
    """Cost/latency tracker for generative (T2I) adapters, which report a flat
    per-image cost instead of token counts."""

    def __init__(self) -> None:
        self._latencies: list[float] = []
        self._costs: list[float] = []

    def record(self, response) -> None:
        self._latencies.append(response.latency_ms)
        self._costs.append(response.cost_usd or 0.0)

    def summary(self) -> CostSummary:
        if not self._latencies:
            return CostSummary(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        n = len(self._latencies)
        latencies = sorted(self._latencies)
        total_cost = sum(self._costs)
        total_time_s = sum(self._latencies) / 1000
        throughput = (n / total_time_s) * 60 if total_time_s > 0 else 0

        return CostSummary(
            total_cost_usd=total_cost,
            cost_per_sample_usd=total_cost / n,
            total_input_tokens=0,
            total_output_tokens=0,
            avg_input_tokens=0,
            avg_output_tokens=0,
            latency_p50_ms=self._percentile(latencies, 50),
            latency_p95_ms=self._percentile(latencies, 95),
            latency_p99_ms=self._percentile(latencies, 99),
            throughput_samples_per_min=throughput,
            n_samples=n,
        )

    def _percentile(self, sorted_values: list[float], pct: int) -> float:
        if not sorted_values:
            return 0.0
        idx = int(len(sorted_values) * pct / 100)
        idx = min(idx, len(sorted_values) - 1)
        return sorted_values[idx]
