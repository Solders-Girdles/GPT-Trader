"""
Daily loss guard - monitors cumulative daily P&L and enforces loss limits.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from gpt_trader.features.live_trade.guard_errors import RiskGuardActionError
from gpt_trader.orchestration.execution.guards.protocol import RuntimeGuardState

if TYPE_CHECKING:
    from gpt_trader.features.live_trade.risk import LiveRiskManager


class DailyLossGuard:
    """
    Guard that tracks daily P&L and triggers protective actions when breached.

    When the daily loss limit is exceeded, this guard cancels all open orders
    and invalidates the guard cache.
    """

    def __init__(
        self,
        risk_manager: LiveRiskManager,
        cancel_all_orders: Callable[[], None],
        invalidate_cache: Callable[[], None],
    ) -> None:
        """
        Initialize daily loss guard.

        Args:
            risk_manager: Risk manager for P&L tracking
            cancel_all_orders: Callback to cancel all open orders
            invalidate_cache: Callback to invalidate guard cache
        """
        self._risk_manager = risk_manager
        self._cancel_all_orders = cancel_all_orders
        self._invalidate_cache = invalidate_cache

    @property
    def name(self) -> str:
        return "daily_loss"

    def check(self, state: RuntimeGuardState, incremental: bool = False) -> None:
        """Check daily loss limits and cancel orders if breached."""
        triggered = self._risk_manager.track_daily_pnl(state.equity, state.positions_pnl)
        if triggered:
            try:
                self._cancel_all_orders()
            except Exception as exc:
                raise RiskGuardActionError(
                    guard_name=self.name,
                    message="Failed to cancel orders after daily loss breach",
                    details={"equity": str(state.equity)},
                ) from exc
            self._invalidate_cache()


__all__ = ["DailyLossGuard"]
