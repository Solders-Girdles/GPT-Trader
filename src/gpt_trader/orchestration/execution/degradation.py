"""
Backward compatibility re-export for DegradationState.

The canonical location is now:
    gpt_trader.features.live_trade.degradation

This module re-exports DegradationState and PauseRecord for backward
compatibility with existing imports from orchestration.execution.degradation.

.. deprecated::
    Import from gpt_trader.features.live_trade.degradation instead.

    Removal target: v3.0
    Tracker: docs/DEPRECATIONS.md
"""

from __future__ import annotations

import warnings

# Single-shot deprecation warning
_deprecation_warned = False


def _emit_deprecation_warning() -> None:
    """Emit deprecation warning once per process."""
    global _deprecation_warned
    if not _deprecation_warned:
        warnings.warn(
            "Importing DegradationState from "
            "gpt_trader.orchestration.execution.degradation is deprecated. "
            "Use gpt_trader.features.live_trade.degradation instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        _deprecation_warned = True


_emit_deprecation_warning()

# Re-export from canonical location
from gpt_trader.features.live_trade.degradation import DegradationState, PauseRecord  # noqa: E402

__all__ = ["DegradationState", "PauseRecord"]
