"""
Lazy import utilities to improve module loading performance.

This module provides utilities for lazy loading of heavy ML libraries,
reducing startup time and memory usage when those libraries are not needed.
"""

import importlib
import importlib.util
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


def is_available(module_name: str) -> bool:
    """
    Check if a module is available for import without actually importing it.

    Args:
        module_name: Name of module to check

    Returns:
        True if module is available, False otherwise
    """
    spec = importlib.util.find_spec(module_name)
    return spec is not None


class LazyImport:
    """
    A lazy import proxy that defers actual import until first attribute access.

    This helps reduce startup time by only importing heavy libraries when they're
    actually needed.

    Usage:
        # Instead of: import tensorflow as tf
        tf = LazyImport('tensorflow')

        # TensorFlow is only imported when you actually use it:
        model = tf.keras.Sequential()  # <- Import happens here
    """

    def __init__(
        self,
        module_name: str,
        package: str | None = None,
        alias: str | None = None,
        min_version: str | None = None,
    ):
        """
        Initialize lazy import.

        Args:
            module_name: Full module name to import (e.g., 'sklearn.ensemble')
            package: Package name for relative imports
            alias: Alternative name for the module
            min_version: Minimum version requirement (optional)
        """
        self._module_name = module_name
        self._package = package
        self._alias = alias or module_name
        self._min_version = min_version
        self._module = None
        self._import_lock = threading.Lock()
        self._import_attempted = False
        self._import_error = None

    def _import_module(self) -> Any:
        """Import the module if not already imported."""
        if self._module is not None:
            return self._module

        if self._import_attempted and self._import_error:
            raise self._import_error

        with self._import_lock:
            # Double-check pattern
            if self._module is not None:
                return self._module

            if self._import_attempted and self._import_error:
                raise self._import_error

            try:
                self._import_attempted = True
                logger.debug(f"Lazy importing {self._module_name}")

                # Import the module
                self._module = importlib.import_module(self._module_name, self._package)

                # Check version if specified
                if self._min_version and hasattr(self._module, "__version__"):
                    self._check_version(self._module.__version__, self._min_version)

                logger.debug(f"Successfully lazy imported {self._module_name}")
                return self._module

            except Exception as e:
                self._import_error = ImportError(f"Failed to import {self._module_name}: {e}")
                logger.warning(f"Failed to lazy import {self._module_name}: {e}")
                raise self._import_error

    def _check_version(self, current_version: str, min_version: str) -> None:
        """Check if current version meets minimum requirement."""
        try:
            from packaging import version

            if version.parse(current_version) < version.parse(min_version):
                raise ImportError(
                    f"{self._module_name} version {current_version} is below "
                    f"minimum required {min_version}"
                )
        except ImportError:
            # packaging not available, skip version check
            logger.warning(
                f"Cannot check version for {self._module_name} - packaging not installed"
            )

    def __getattr__(self, name: str) -> Any:
        """Get attribute from the imported module."""
        module = self._import_module()
        return getattr(module, name)

    def __call__(self, *args, **kwargs) -> Any:
        """Make the lazy import callable."""
        module = self._import_module()
        return module(*args, **kwargs)

    def __dir__(self) -> list:
        """Return available attributes."""
        try:
            module = self._import_module()
            return dir(module)
        except ImportError:
            return []

    def __repr__(self) -> str:
        status = "imported" if self._module is not None else "not imported"
        return f"LazyImport('{self._module_name}', {status})"


class LazyFromImport:
    """
    Lazy import for 'from module import item' style imports.

    Usage:
        # Instead of: from sklearn.ensemble import RandomForestClassifier
        RandomForestClassifier = LazyFromImport('sklearn.ensemble', 'RandomForestClassifier')
    """

    def __init__(self, module_name: str, item_name: str, package: str | None = None):
        """
        Initialize lazy from import.

        Args:
            module_name: Module to import from
            item_name: Specific item to import
            package: Package for relative imports
        """
        self._module_name = module_name
        self._item_name = item_name
        self._package = package
        self._item = None
        self._import_lock = threading.Lock()
        self._import_attempted = False
        self._import_error = None

    def _import_item(self) -> Any:
        """Import the specific item if not already imported."""
        if self._item is not None:
            return self._item

        if self._import_attempted and self._import_error:
            raise self._import_error

        with self._import_lock:
            # Double-check pattern
            if self._item is not None:
                return self._item

            if self._import_attempted and self._import_error:
                raise self._import_error

            try:
                self._import_attempted = True
                logger.debug(f"Lazy importing {self._item_name} from {self._module_name}")

                module = importlib.import_module(self._module_name, self._package)
                self._item = getattr(module, self._item_name)

                logger.debug(
                    f"Successfully lazy imported {self._item_name} from {self._module_name}"
                )
                return self._item

            except Exception as e:
                self._import_error = ImportError(
                    f"Failed to import {self._item_name} from {self._module_name}: {e}"
                )
                logger.warning(
                    f"Failed to lazy import {self._item_name} from {self._module_name}: {e}"
                )
                raise self._import_error

    def __call__(self, *args, **kwargs) -> Any:
        """Make the imported item callable."""
        item = self._import_item()
        return item(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Get attribute from the imported item."""
        item = self._import_item()
        return getattr(item, name)

    def __repr__(self) -> str:
        status = "imported" if self._item is not None else "not imported"
        return f"LazyFromImport('{self._module_name}.{self._item_name}', {status})"


# Pre-configured lazy imports for common heavy libraries
np = LazyImport("numpy")
pd = LazyImport("pandas")
plt = LazyImport("matplotlib.pyplot")
sns = LazyImport("seaborn")

# Scikit-learn lazy imports
sklearn = LazyImport("sklearn")
RandomForestClassifier = LazyFromImport("sklearn.ensemble", "RandomForestClassifier")
RandomForestRegressor = LazyFromImport("sklearn.ensemble", "RandomForestRegressor")
StandardScaler = LazyFromImport("sklearn.preprocessing", "StandardScaler")
MinMaxScaler = LazyFromImport("sklearn.preprocessing", "MinMaxScaler")
PCA = LazyFromImport("sklearn.decomposition", "PCA")
IsolationForest = LazyFromImport("sklearn.ensemble", "IsolationForest")
LogisticRegression = LazyFromImport("sklearn.linear_model", "LogisticRegression")
LinearRegression = LazyFromImport("sklearn.linear_model", "LinearRegression")
GradientBoostingClassifier = LazyFromImport("sklearn.ensemble", "GradientBoostingClassifier")

# XGBoost lazy imports
xgb = LazyImport("xgboost")
XGBClassifier = LazyFromImport("xgboost", "XGBClassifier")
XGBRegressor = LazyFromImport("xgboost", "XGBRegressor")

# Optional heavy libraries (only available if installed)
tf = LazyImport("tensorflow") if is_available("tensorflow") else None
torch = LazyImport("torch") if is_available("torch") else None


def get_import_stats() -> dict[str, dict[str, bool | str]]:
    """
    Get statistics about lazy import usage.

    Returns:
        Dictionary with import statistics
    """
    stats = {}

    # Get all LazyImport objects from global scope
    for name, obj in globals().items():
        if isinstance(obj, (LazyImport, LazyFromImport)):
            stats[name] = {
                "imported": obj._module is not None or getattr(obj, "_item", None) is not None,
                "module": getattr(obj, "_module_name", "unknown"),
                "error": str(obj._import_error) if getattr(obj, "_import_error", None) else None,
            }

    return stats


def preload_essentials() -> None:
    """
    Preload essential libraries that are commonly used.
    Call this during application startup if you know certain libraries will be needed.
    """
    logger.info("Preloading essential ML libraries...")

    # Preload numpy and pandas as they're used almost everywhere
    try:
        _ = np.__version__
        logger.debug("Preloaded numpy")
    except ImportError:
        logger.warning("Could not preload numpy")

    try:
        _ = pd.__version__
        logger.debug("Preloaded pandas")
    except ImportError:
        logger.warning("Could not preload pandas")


# Backward compatibility aliases
numpy = np
pandas = pd
matplotlib_pyplot = plt
seaborn = sns
scikit_learn = sklearn
xgboost = xgb
tensorflow = tf
pytorch = torch


# Fix the get_import_stats function to handle both LazyImport and LazyFromImport properly
def get_import_stats_fixed() -> dict[str, dict[str, bool | str]]:
    """
    Get statistics about lazy import usage.

    Returns:
        Dictionary with import statistics
    """
    stats = {}

    # Get all LazyImport objects from global scope
    for name, obj in globals().items():
        if isinstance(obj, LazyImport):
            stats[name] = {
                "imported": obj._module is not None,
                "module": obj._module_name,
                "error": str(obj._import_error) if obj._import_error else None,
            }
        elif isinstance(obj, LazyFromImport):
            stats[name] = {
                "imported": obj._item is not None,
                "module": f"{obj._module_name}.{obj._item_name}",
                "error": str(obj._import_error) if obj._import_error else None,
            }

    return stats
