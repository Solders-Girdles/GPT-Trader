"""
Protocol definitions for risk management abstractions.

These protocols define the expected interfaces for risk manager implementations,
enabling structural typing and testability.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Protocol, runtime_checkable

from gpt_trader.features.live_trade.risk.manager import VolatilityCheckOutcome


@runtime_checkable
class RiskManagerProtocol(Protocol):
    """
    Protocol for risk management implementations.

    Defines the interface expected by trading execution and guard components.
    Implemented by: LiveRiskManager.
    """

    config: Any
    last_mark_update: dict[str, float]

    def check_order(self, order: Any) -> bool:
        """Check if an order passes risk validation."""
        ...

    def update_position(self, position: Any) -> None:
        """Update position tracking state."""
        ...

    def check_liquidation_buffer(self, symbol: str, position: Any, equity: Decimal) -> bool:
        """Check if position is approaching liquidation threshold."""
        ...

    def pre_trade_validate(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        product: Any,
        equity: Decimal,
        current_positions: dict[str, Any],
    ) -> None:
        """Validate a trade before execution. Raises ValidationError on failure."""
        ...

    def track_daily_pnl(
        self, equity: Decimal, positions_pnl: dict[str, dict[str, Decimal]]
    ) -> bool:
        """Track daily PnL and return True if loss limit triggered."""
        ...

    def check_mark_staleness(self, symbol: str) -> bool:
        """Check if mark price for symbol is stale."""
        ...

    def append_risk_metrics(
        self, equity: Decimal, positions_dict: dict[str, dict[str, Decimal]]
    ) -> None:
        """Record current risk metrics for monitoring."""
        ...

    def check_volatility_circuit_breaker(
        self, symbol: str, closes: list[Decimal]
    ) -> VolatilityCheckOutcome:
        """Check if volatility circuit breaker should trigger."""
        ...

    def set_reduce_only_mode(self, value: bool, reason: str = "") -> None:
        """Enable or disable reduce-only trading mode."""
        ...

    def is_reduce_only_mode(self) -> bool:
        """Check if reduce-only mode is active."""
        ...

    def reset_daily_tracking(self) -> None:
        """Reset daily tracking state for a new trading day."""
        ...


__all__ = ["RiskManagerProtocol"]
