"""vlm-compress: model-agnostic visual token compression for VLM serving.

Common API across compression methods (random baseline, FastV, ...) so
callers can swap methods without touching model-specific code. See
docs/vlm-compress-plan.html (or the design doc) for the method survey this
package implements against.
"""

from __future__ import annotations

from vlm_compress.base import CompressionResult, TokenCompressor
from vlm_compress.registry import create_compressor, list_methods, register_method

__all__ = [
    "CompressionResult",
    "TokenCompressor",
    "create_compressor",
    "list_methods",
    "register_method",
]
