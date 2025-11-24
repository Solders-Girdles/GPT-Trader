"""Curated optional and lazy imports shared across the project."""

from __future__ import annotations

import sys

from .lazy import lazy_import
from .optional import optional_import

# Common optional dependencies for analytics and data science workflows
pandas = optional_import("pandas")
numpy = optional_import("numpy")
matplotlib = optional_import("matplotlib")
plotly = optional_import("plotly")
scipy = optional_import("scipy")
sklearn = optional_import("sklearn")

# Heavy ML/optimization stacks loaded lazily
tensorflow = lazy_import("tensorflow")
torch = lazy_import("torch")
cvxpy = lazy_import("cvxpy")


def is_test_environment() -> bool:
    """Return True when pytest or unittest modules are present."""
    return "pytest" in sys.modules or "unittest" in sys.modules


# Conditional helpers for interactive usage
test_utils = optional_import("pytest") if is_test_environment() else None
dev_utils = optional_import("IPython") if not is_test_environment() else None


__all__ = [
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
