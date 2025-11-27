"""
Guards subpackage - runtime safety checks for live trading.

This package contains individual guard implementations that follow the Guard protocol,
along with the cache and orchestration components.

Note: GuardManager is in guard_manager.py, not this package, to avoid circular imports.
Import it from gpt_trader.orchestration.execution.guard_manager or
gpt_trader.orchestration.execution (which re-exports it).
"""

from gpt_trader.orchestration.execution.guards.cache import GuardStateCache
from gpt_trader.orchestration.execution.guards.daily_loss import DailyLossGuard
from gpt_trader.orchestration.execution.guards.liquidation_buffer import LiquidationBufferGuard
from gpt_trader.orchestration.execution.guards.mark_staleness import MarkStalenessGuard
from gpt_trader.orchestration.execution.guards.pnl_telemetry import PnLTelemetryGuard
from gpt_trader.orchestration.execution.guards.protocol import Guard, RuntimeGuardState
from gpt_trader.orchestration.execution.guards.risk_metrics import RiskMetricsGuard
from gpt_trader.orchestration.execution.guards.volatility import VolatilityGuard

__all__ = [
    # Protocol
    "Guard",
    "RuntimeGuardState",
    # Cache
    "GuardStateCache",
    # Guards
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
        from gpt_trader.orchestration.execution.guard_manager import GuardManager

        return GuardManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
