"""Runtime monitoring delegation for the live risk manager."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from bot_v2.features.live_trade.risk_runtime import CircuitBreakerOutcome


class LiveRiskManagerRuntimeMixin:
    """Delegate runtime monitoring responsibilities to the monitor service."""

    def track_daily_pnl(
        self, current_equity: Decimal, positions_pnl: dict[str, dict[str, Decimal]]
    ) -> bool:
        """Track daily PnL and trigger reduce-only if breaching limit."""
        # Initialize start of day if needed
        if self.state_manager.start_of_day_equity == 0:
            self.state_manager.start_of_day_equity = current_equity
            self.start_of_day_equity = current_equity
            return False

        triggered, updated_pnl = self.runtime_monitor.track_daily_pnl(
            current_equity=current_equity,
            positions_pnl=positions_pnl,
            daily_pnl=self.state_manager.daily_pnl,
            start_of_day_equity=self.state_manager.start_of_day_equity,
        )

        # Update state manager's daily_pnl
        self.state_manager.daily_pnl = updated_pnl
        self.daily_pnl = updated_pnl

        return triggered

    def check_liquidation_buffer(
        self, symbol: str, position_data: dict[str, Any], equity: Decimal
    ) -> bool:
        """Monitor liquidation buffer for position."""
        result = self.runtime_monitor.check_liquidation_buffer(symbol, position_data, equity)
        # Update local reference for backward compatibility
        self.positions = self.runtime_monitor.positions
        return result

    def check_mark_staleness(self, symbol: str, mark_timestamp: datetime | None = None) -> bool:
        """Check whether mark data for symbol is stale."""
        return self.runtime_monitor.check_mark_staleness(symbol, mark_timestamp)

    def append_risk_metrics(self, equity: Decimal, positions: dict[str, Any]) -> None:
        """Append periodic risk metrics snapshot to the event store."""
        self.runtime_monitor.append_risk_metrics(
            equity=equity,
            positions=positions,
            daily_pnl=self.state_manager.daily_pnl,
            start_of_day_equity=self.state_manager.start_of_day_equity,
            is_reduce_only_mode=self.state_manager.is_reduce_only_mode(),
        )

    def check_correlation_risk(self, positions: dict[str, Any]) -> bool:
        """Check portfolio correlation and concentration risk."""
        return self.runtime_monitor.check_correlation_risk(positions)

    def check_volatility_circuit_breaker(
        self, symbol: str, recent_marks: list[Decimal]
    ) -> CircuitBreakerOutcome:
        """Check rolling volatility and trigger progressive circuit breakers."""
        outcome = self.runtime_monitor.check_volatility_circuit_breaker(symbol, recent_marks)
        # Update local reference for backward compatibility
        self._cb_state_adapter.update_state(self.runtime_monitor.circuit_breaker_state)
        return outcome


__all__ = ["LiveRiskManagerRuntimeMixin"]
