"""Benchmark registry: discover, validate, and load YAML benchmark manifests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from vlm_evaluation_harness.benchmarks.schema import BenchmarkManifest, ManifestError

# Built-in manifests shipped with the package
_BUILTIN_DIR = Path(__file__).parent / "manifests"


def _normalize_key(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


class BenchmarkRegistry:
    """Discovers, loads, and validates benchmark manifests.

    Manifests that fail validation are *not* silently skipped: the failure is
    retained and re-raised when that benchmark is requested, and reported by
    `errors()` so CI can assert the shipped set is clean.
    """

    def __init__(self, extra_dirs: list[Path] | None = None):
        self._manifests: dict[str, BenchmarkManifest] = {}
        self._hashes: dict[str, str] = {}
        self._errors: dict[str, str] = {}
        search_dirs = [_BUILTIN_DIR]
        if extra_dirs:
            search_dirs.extend(extra_dirs)
        for d in search_dirs:
            self._load_dir(d)

    def _load_dir(self, directory: Path) -> None:
        if not directory.exists():
            return
        for yaml_file in sorted(directory.glob("*.yaml")):
            try:
                manifest, digest = self._load_file(yaml_file)
            except Exception as e:
                self._errors[yaml_file.stem.lower()] = str(e)
                continue
            for key in {_normalize_key(manifest.name), yaml_file.stem.lower()}:
                self._manifests[key] = manifest
                self._hashes[key] = digest

    def _load_file(self, path: Path) -> tuple[BenchmarkManifest, str]:
        raw = path.read_bytes()
        data = yaml.safe_load(raw)
        manifest = BenchmarkManifest.from_dict(data)
        if manifest.source.type == "local" and not Path(manifest.source.path).is_absolute():
            manifest.source.path = str((path.parent / manifest.source.path).resolve())
        manifest.validate()
        return manifest, hashlib.sha256(raw).hexdigest()[:16]

    def get(self, name: str) -> BenchmarkManifest:
        key = _normalize_key(name)
        if key in self._manifests:
            return self._manifests[key]
        if key in self._errors:
            raise ManifestError(
                f"Benchmark '{name}' failed validation and cannot be run:\n{self._errors[key]}"
            )
        available = sorted(set(self._manifests.keys()))
        raise KeyError(f"Benchmark '{name}' not found. Available: {available}")

    def manifest_hash(self, name: str) -> str:
        """Content hash of the manifest file, recorded in results for provenance."""
        return self._hashes.get(_normalize_key(name), "")

    def errors(self) -> dict[str, str]:
        """Manifests that failed to load or validate, keyed by filename stem."""
        return dict(self._errors)

    def list(self) -> list[str]:
        """Return unique benchmark names (deduplicated)."""
        return sorted({m.name for m in self._manifests.values()})

    def list_by_tags(self, tags: list[str]) -> list[str]:
        """Benchmark names carrying at least one of `tags`."""
        wanted = set(tags)
        return sorted(
            {
                m.name
                for m in {m.name: m for m in self._manifests.values()}.values()
                if wanted & set(m.tags)
            }
        )

    def list_by_category(self) -> dict[str, list[str]]:
        """Return benchmark names grouped by taxonomy category."""
        categories: dict[str, list[str]] = {}
        for manifest in {m.name: m for m in self._manifests.values()}.values():
            categories.setdefault(manifest.taxonomy_category, []).append(manifest.name)
        return {k: sorted(v) for k, v in sorted(categories.items())}

    def __len__(self) -> int:
        return len({m.name for m in self._manifests.values()})


# Module-level singleton
_registry: BenchmarkRegistry | None = None


def get_registry(extra_dirs: list[Path] | None = None) -> BenchmarkRegistry:
    global _registry
    if _registry is None or extra_dirs:
        _registry = BenchmarkRegistry(extra_dirs=extra_dirs)
    return _registry
