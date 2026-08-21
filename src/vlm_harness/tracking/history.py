"""Run history: an append-only local record of every tracked evaluation run.

Aggregate metrics go in the JSON-lines history file (diffable, greppable,
zero setup). Per-sample scores — needed for paired significance testing —
are too large for that file to stay diffable, so they go alongside it as one
JSON file per run, referenced by `run_id`.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _default_path() -> Path:
    return Path.home() / ".vlm-harness" / "history.jsonl"


def _default_samples_dir() -> Path:
    return Path.home() / ".vlm-harness" / "samples"


@dataclass
class HistoryEntry:
    run_id: str
    timestamp: str
    model: str
    benchmark: str
    split: str
    modality: str  # "discriminative" | "generative"
    metrics: dict[str, float]
    n_samples: int
    n_scored: dict[str, int] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "model": self.model,
            "benchmark": self.benchmark,
            "split": self.split,
            "modality": self.modality,
            "metrics": self.metrics,
            "n_samples": self.n_samples,
            "n_scored": self.n_scored,
            "provenance": self.provenance,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> HistoryEntry:
        return cls(
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            model=data["model"],
            benchmark=data["benchmark"],
            split=data.get("split", ""),
            modality=data.get("modality", "discriminative"),
            metrics=data.get("metrics", {}),
            n_samples=data.get("n_samples", 0),
            n_scored=data.get("n_scored", {}),
            provenance=data.get("provenance", {}),
            metadata=data.get("metadata", {}),
        )


class HistoryStore:
    """Appends and queries evaluation runs stored as JSON-lines."""

    def __init__(self, path: Path | None = None, samples_dir: Path | None = None):
        self.path = path or _default_path()
        self.samples_dir = samples_dir or _default_samples_dir()

    def record(
        self,
        model: str,
        benchmark: str,
        split: str,
        metrics: dict[str, float],
        n_samples: int,
        modality: str = "discriminative",
        n_scored: dict[str, int] | None = None,
        provenance: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        per_sample: dict[str, dict[str, float]] | None = None,
    ) -> HistoryEntry:
        entry = HistoryEntry(
            run_id=uuid.uuid4().hex[:12],
            timestamp=datetime.now(timezone.utc).isoformat(),
            model=model,
            benchmark=benchmark,
            split=split,
            modality=modality,
            metrics=metrics,
            n_samples=n_samples,
            n_scored=n_scored or {},
            provenance=provenance or {},
            metadata=metadata or {},
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry.to_dict(), default=str) + "\n")

        if per_sample:
            self.samples_dir.mkdir(parents=True, exist_ok=True)
            (self.samples_dir / f"{entry.run_id}.json").write_text(json.dumps(per_sample))

        return entry

    def record_result(self, result: Any, modality: str = "discriminative") -> HistoryEntry:
        """Convenience wrapper: record an EvalResult or GenEvalResult directly."""
        d = result.to_dict()
        per_sample = result.per_sample_scores() if hasattr(result, "per_sample_scores") else None
        return self.record(
            model=d["model"],
            benchmark=d["benchmark"],
            split=d["split"],
            metrics=d["metrics"],
            n_samples=d["n_samples"],
            modality=modality,
            n_scored=d.get("metric_n_scored"),
            provenance=d.get("provenance"),
            per_sample=per_sample,
        )

    def per_sample_scores(self, run_id: str) -> dict[str, dict[str, float]]:
        """Load the per-sample scores recorded for a run, if any."""
        path = self.samples_dir / f"{run_id}.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text())

    def all(self) -> list[HistoryEntry]:
        if not self.path.exists():
            return []
        entries = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(HistoryEntry.from_dict(json.loads(line)))
        return entries

    def query(
        self, model: str | None = None, benchmark: str | None = None
    ) -> list[HistoryEntry]:
        entries = self.all()
        if model:
            entries = [e for e in entries if e.model == model]
        if benchmark:
            entries = [e for e in entries if e.benchmark == benchmark]
        return sorted(entries, key=lambda e: e.timestamp)

    def latest(self, model: str, benchmark: str) -> HistoryEntry | None:
        matches = self.query(model=model, benchmark=benchmark)
        return matches[-1] if matches else None

    def benchmarks_for(self, model: str) -> list[str]:
        return sorted({e.benchmark for e in self.query(model=model)})
