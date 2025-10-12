"""Backward-compatible facade for legacy import utility helpers.

The import utilities now live in :mod:`bot_v2.utilities.importing`. This module
re-exports the public surface so existing imports keep working while new code
can depend on the structured package.
"""

from __future__ import annotations

import warnings

from .importing import (
    ImportProfiler,
    LazyImport,
    OptionalImport,
    conditional_import,
    cvxpy,
    dev_utils,
    get_import_stats,
    is_test_environment,
    lazy_import,
    matplotlib,
    numpy,
    optimize_imports,
    optional_import,
    pandas,
    plotly,
    scipy,
    sklearn,
    tensorflow,
    test_utils,
    torch,
    with_lazy_imports,
)

__all__ = [
    "LazyImport",
    "lazy_import",
    "OptionalImport",
    "optional_import",
    "conditional_import",
    "with_lazy_imports",
    "ImportProfiler",
    "get_import_stats",
    "optimize_imports",
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

warnings.warn(
    "bot_v2.utilities.import_utils is deprecated; import from bot_v2.utilities.importing instead.",
    DeprecationWarning,
    stacklevel=2,
)
