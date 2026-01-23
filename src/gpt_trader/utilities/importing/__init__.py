"""Modular import helpers used throughout the GPT-Trader codebase."""

from .lazy import LazyImport, lazy_import, with_lazy_imports
from .optional import OptionalImport, conditional_import, optional_import
from .profiling import ImportProfiler, get_import_stats
from .registry import (
    aiohttp,
    opentelemetry,
    pandas,
    psutil,
    websocket,
)

__all__ = [
    # Lazy / optional wrappers
    "LazyImport",
    "lazy_import",
    "with_lazy_imports",
    "OptionalImport",
    "optional_import",
    "conditional_import",
    # Profiling utilities
    "ImportProfiler",
    "get_import_stats",
    # Registry exports
    "pandas",
    "aiohttp",
    "websocket",
    "psutil",
    "opentelemetry",
]
