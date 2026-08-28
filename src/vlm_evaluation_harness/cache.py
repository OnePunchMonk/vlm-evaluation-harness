"""Content-addressed response cache.

Every model call is keyed by a hash of everything that can change its
output: model id, rendered prompt, system prompt, the hashes of the images
actually sent, and the decoding parameters. A re-run therefore costs nothing
for samples already seen, and a run that dies at sample 800 of 900 resumes
instead of starting over.

SQLite rather than a directory of files: it is in the standard library,
survives concurrent writers from the thread pool, and stays one portable
file per user.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DEFAULT_PATH = Path.home() / ".vlm-evaluation-harness" / "cache" / "responses.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key         TEXT PRIMARY KEY,
    model_id    TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS responses_model_idx ON responses (model_id);
"""


def response_key(
    model_id: str,
    prompt: str,
    system: str | None,
    image_hashes: list[str],
    params: dict[str, Any],
) -> str:
    """Stable hash over every input that can change a model's output."""
    payload = json.dumps(
        {
            "model_id": model_id,
            "prompt": prompt,
            "system": system,
            "images": list(image_hashes),
            "params": {k: params[k] for k in sorted(params)},
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class ResponseCache:
    """Thread-safe key/value store for model responses."""

    def __init__(self, path: Path | None = None, enabled: bool = True):
        self.path = Path(path) if path is not None else _DEFAULT_PATH
        self.enabled = enabled
        self.stats = CacheStats()
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        if self.enabled:
            self._connect()

    def _connect(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def get(self, key: str) -> dict | None:
        if not self.enabled or self._conn is None:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM responses WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                self.stats.misses += 1
                return None
            self.stats.hits += 1
            return json.loads(row[0])

    def put(self, key: str, model_id: str, payload: dict) -> None:
        if not self.enabled or self._conn is None:
            return
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses (key, model_id, payload) VALUES (?, ?, ?)",
                (key, model_id, json.dumps(payload)),
            )
            self._conn.commit()
            self.stats.writes += 1

    def clear(self, model_id: str | None = None) -> int:
        """Delete cached responses, optionally only for one model. Returns rows removed."""
        if self._conn is None:
            return 0
        with self._lock:
            if model_id:
                cur = self._conn.execute(
                    "DELETE FROM responses WHERE model_id = ?", (model_id,)
                )
            else:
                cur = self._conn.execute("DELETE FROM responses")
            self._conn.commit()
            return cur.rowcount

    def size(self) -> int:
        if self._conn is None:
            return 0
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self) -> ResponseCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
