"""Modular import helpers used throughout the GPT-Trader codebase."""

from .lazy import LazyImport, lazy_import, with_lazy_imports
from .optional import OptionalImport, conditional_import, optional_import
from .profiling import ImportProfiler, get_import_stats, optimize_imports
from .registry import (
    cvxpy,
    dev_utils,
    is_test_environment,
    matplotlib,
    numpy,
    pandas,
    plotly,
    scipy,
    sklearn,
    tensorflow,
    test_utils,
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
    "optimize_imports",
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
    "test_utils",
    "dev_utils",
]
