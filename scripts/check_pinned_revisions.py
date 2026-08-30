#!/usr/bin/env python3
"""CI check: every huggingface benchmark manifest must pin a commit sha.

A manifest may use a mutable ref (e.g. "main") only if it documents why via
`source.revision_note` -- e.g. a gated dataset that can't be resolved to a
sha without authenticating first. See issue #12.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_REPO_ROOT = Path(__file__).resolve().parent.parent
_MANIFEST_DIR = _REPO_ROOT / "src" / "vlm_harness" / "benchmarks" / "manifests"


def main() -> int:
    errors = []
    for path in sorted(_MANIFEST_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text())
        source = data.get("source", {})
        if source.get("type") != "huggingface":
            continue

        revision = source.get("revision", "main")
        note = source.get("revision_note")
        if _SHA_RE.match(revision):
            continue
        if not note:
            errors.append(
                f"{path.name}: revision {revision!r} is not a pinned commit sha "
                "and has no revision_note justifying it"
            )

    if errors:
        print("Unpinned/undocumented dataset revisions:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print(f"All {sum(1 for _ in _MANIFEST_DIR.glob('*.yaml'))} manifests OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
