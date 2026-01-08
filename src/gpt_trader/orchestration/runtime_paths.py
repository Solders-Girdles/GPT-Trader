"""
DEPRECATED: This module has moved to gpt_trader.app.runtime.paths

This shim exists for backwards compatibility. Update imports to use:
    from gpt_trader.app.runtime import RuntimePaths, resolve_runtime_paths
    # or
    from gpt_trader.app.runtime.paths import RuntimePaths, resolve_runtime_paths
"""

from __future__ import annotations

import warnings

from gpt_trader.app.runtime.paths import RuntimePaths, resolve_runtime_paths

__all__ = ["RuntimePaths", "resolve_runtime_paths"]

warnings.warn(
    "gpt_trader.orchestration.runtime_paths is deprecated. "
    "Import from gpt_trader.app.runtime instead.",
    DeprecationWarning,
    stacklevel=2,
)
