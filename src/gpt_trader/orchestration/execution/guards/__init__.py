"""
DEPRECATED: This package has moved to gpt_trader.features.live_trade.execution.guards

This shim re-exports all symbols for backwards compatibility.
Please update your imports to use the new location:

    # Old (deprecated)
    from gpt_trader.orchestration.execution.guards import RuntimeGuardState

    # New (preferred)
    from gpt_trader.features.live_trade.execution.guards import RuntimeGuardState
"""

from __future__ import annotations

import warnings

from gpt_trader.features.live_trade.execution.guards import (
    ApiHealthGuard,
    DailyLossGuard,
    Guard,
    GuardStateCache,
    LiquidationBufferGuard,
    MarkStalenessGuard,
    PnLTelemetryGuard,
    RiskMetricsGuard,
    RuntimeGuardState,
    VolatilityGuard,
)

__all__ = [
    # Protocol
    "Guard",
    "RuntimeGuardState",
    # Cache
    "GuardStateCache",
    # Guards
    "ApiHealthGuard",
    "DailyLossGuard",
    "LiquidationBufferGuard",
    "MarkStalenessGuard",
    "PnLTelemetryGuard",
    "RiskMetricsGuard",
    "VolatilityGuard",
]


def __getattr__(name: str) -> type:
    """Lazy import GuardManager for backward compatibility."""
    if name == "GuardManager":
        from gpt_trader.features.live_trade.execution.guard_manager import GuardManager

        return GuardManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Emit deprecation warning on import
warnings.warn(
    "gpt_trader.orchestration.execution.guards is deprecated. "
    "Import from gpt_trader.features.live_trade.execution.guards instead.",
    DeprecationWarning,
    stacklevel=2,
)
