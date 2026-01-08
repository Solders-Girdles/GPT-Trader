"""
Mark staleness guard - detects stale mark prices that could affect trading logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gpt_trader.features.live_trade.execution.guards.protocol import RuntimeGuardState
from gpt_trader.features.live_trade.guard_errors import RiskGuardDataUnavailable

if TYPE_CHECKING:
    from gpt_trader.features.live_trade.risk import LiveRiskManager


class MarkStalenessGuard:
    """
    Guard that checks if mark prices are stale.

    Queries the broker's mark cache (if available) and validates
    mark price freshness against configured thresholds.
    """

    def __init__(
        self,
        broker: Any,  # Uses internal _mark_cache
        risk_manager: LiveRiskManager,
    ) -> None:
        """
        Initialize mark staleness guard.

        Args:
            broker: Broker with optional _mark_cache
            risk_manager: Risk manager for staleness tracking
        """
        self._broker = broker
        self._risk_manager = risk_manager

    @property
    def name(self) -> str:
        return "mark_staleness"

    def check(self, state: RuntimeGuardState, incremental: bool = False) -> None:
        """Check if mark prices are stale."""
        if not hasattr(self._broker, "_mark_cache"):
            return

        failures: list[dict[str, Any]] = []
        for symbol in list(self._risk_manager.last_mark_update.keys()):
            try:
                mark = self._broker._mark_cache.get_mark(symbol)
            except Exception as exc:
                failures.append({"symbol": symbol, "error": repr(exc)})
                continue
            if mark is None:
                self._risk_manager.check_mark_staleness(symbol)

        if failures:
            raise RiskGuardDataUnavailable(
                guard_name=self.name,
                message="Failed to refresh mark data for one or more symbols",
                details={"failures": failures},
            )


__all__ = ["MarkStalenessGuard"]
