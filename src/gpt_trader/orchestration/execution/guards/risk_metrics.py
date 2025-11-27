"""
Risk metrics guard - appends current risk metrics for monitoring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gpt_trader.features.live_trade.guard_errors import GuardError, RiskGuardTelemetryError
from gpt_trader.orchestration.execution.guards.protocol import Guard, RuntimeGuardState

if TYPE_CHECKING:
    from gpt_trader.features.live_trade.risk import LiveRiskManager


class RiskMetricsGuard:
    """
    Guard that logs/appends current risk metrics for monitoring.

    Ensures risk metrics are recorded even during normal operation
    for observability and post-session analysis.
    """

    def __init__(self, risk_manager: LiveRiskManager) -> None:
        """
        Initialize risk metrics guard.

        Args:
            risk_manager: Risk manager for metrics recording
        """
        self._risk_manager = risk_manager

    @property
    def name(self) -> str:
        return "risk_metrics"

    def check(self, state: RuntimeGuardState, incremental: bool = False) -> None:
        """Append risk metrics for monitoring."""
        try:
            self._risk_manager.append_risk_metrics(state.equity, state.positions_dict)
        except GuardError:
            raise
        except Exception as exc:
            raise RiskGuardTelemetryError(
                guard_name=self.name,
                message="Failed to append risk metrics",
                details={"equity": str(state.equity)},
            ) from exc


# Verify protocol compliance
_: Guard = RiskMetricsGuard(None)  # type: ignore[arg-type]

__all__ = ["RiskMetricsGuard"]
