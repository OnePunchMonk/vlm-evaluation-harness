"""JSON result serialization."""

from __future__ import annotations

import json
from pathlib import Path


def save_json(result, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2))
