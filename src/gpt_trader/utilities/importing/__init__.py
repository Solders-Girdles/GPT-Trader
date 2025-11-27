"""Modular import helpers used throughout the GPT-Trader codebase."""

from .lazy import LazyImport, lazy_import, with_lazy_imports
from .optional import OptionalImport, conditional_import, optional_import
from .profiling import ImportProfiler, get_import_stats
from .registry import (
    cvxpy,
    dev_utils,  # naming: allow
    is_test_environment,
    matplotlib,
    numpy,
    pandas,
    plotly,
    scipy,
    sklearn,
    tensorflow,
    test_utils,  # naming: allow
    torch,
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
    "numpy",
    "matplotlib",
    "plotly",
    "scipy",
    "sklearn",
    "tensorflow",
    "torch",
    "cvxpy",
    "is_test_environment",
    "test_utils",  # naming: allow
    "dev_utils",  # naming: allow
]
