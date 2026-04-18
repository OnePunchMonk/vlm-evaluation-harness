"""Benchmark registry: discover and load YAML benchmark manifests."""

from __future__ import annotations

from pathlib import Path

import yaml

from vlm_harness.benchmarks.schema import BenchmarkManifest

# Built-in manifests shipped with the package
_BUILTIN_DIR = Path(__file__).parent / "manifests"


class BenchmarkRegistry:
    """Discovers, loads, and validates benchmark manifests."""

    def __init__(self, extra_dirs: list[Path] | None = None):
        self._manifests: dict[str, BenchmarkManifest] = {}
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
                manifest = self._load_file(yaml_file)
                key = manifest.name.lower().replace(" ", "_").replace("-", "_")
                self._manifests[key] = manifest
                # Also register by filename stem
                self._manifests[yaml_file.stem.lower()] = manifest
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to load benchmark manifest {yaml_file}: {e}")

    def _load_file(self, path: Path) -> BenchmarkManifest:
        with open(path) as f:
            data = yaml.safe_load(f)
        return BenchmarkManifest.from_dict(data)

    def get(self, name: str) -> BenchmarkManifest:
        key = name.lower().replace(" ", "_").replace("-", "_")
        if key not in self._manifests:
            available = sorted(set(self._manifests.keys()))
            raise KeyError(
                f"Benchmark '{name}' not found. Available: {available}"
            )
        return self._manifests[key]

    def list(self) -> list[str]:
        """Return unique benchmark names (deduplicated)."""
        seen: set[str] = set()
        result = []
        for manifest in self._manifests.values():
            if manifest.name not in seen:
                seen.add(manifest.name)
                result.append(manifest.name)
        return sorted(result)

    def list_by_category(self) -> dict[str, list[str]]:
        """Return benchmark names grouped by taxonomy category."""
        categories: dict[str, list[str]] = {}
        seen: set[str] = set()
        for manifest in self._manifests.values():
            if manifest.name in seen:
                continue
            seen.add(manifest.name)
            cat = manifest.taxonomy_category
            categories.setdefault(cat, []).append(manifest.name)
        return {k: sorted(v) for k, v in sorted(categories.items())}

    def __len__(self) -> int:
        return len(set(m.name for m in self._manifests.values()))


# Module-level singleton
_registry: BenchmarkRegistry | None = None


def get_registry(extra_dirs: list[Path] | None = None) -> BenchmarkRegistry:
    global _registry
    if _registry is None or extra_dirs:
        _registry = BenchmarkRegistry(extra_dirs=extra_dirs)
    return _registry
