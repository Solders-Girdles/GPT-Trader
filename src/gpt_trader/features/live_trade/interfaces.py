"""
Strategy protocol and common types for the live trading engine.

This module defines the interface that all trading strategies must implement,
enabling the engine to swap strategies without code changes.
"""

from collections.abc import Sequence
from decimal import Decimal
from typing import Any, Protocol, runtime_checkable

from gpt_trader.core import Product
from gpt_trader.features.live_trade.strategies.perps_baseline import Decision


@runtime_checkable
class TradingStrategy(Protocol):
    """Protocol defining the interface for all trading strategies.

    Any strategy class that implements this protocol can be used by TradingEngine.
    The protocol requires:
    - A `decide()` method for generating trading decisions
    - An optional `rehydrate()` method for crash recovery
    """

    def decide(
        self,
        symbol: str,
        current_mark: Decimal,
        position_state: dict[str, Any] | None,
        recent_marks: Sequence[Decimal],
        equity: Decimal,
        product: Product | None,
        candles: Sequence[Any] | None = None,
    ) -> Decision:
        """Generate a trading decision based on market data.

        Args:
            symbol: Trading pair symbol (e.g., "BTC-USD")
            current_mark: Current mark/spot price
            position_state: Current position info (None if no position)
            recent_marks: Historical prices (oldest first, typically 20 periods)
            equity: Account equity for position sizing
            product: Product specification (optional)
            candles: Historical candles for advanced indicators (optional)

        Returns:
            Decision with action (BUY/SELL/HOLD/CLOSE), reason, confidence, indicators
        """
        ...

    def rehydrate(self, events: Sequence[dict[str, Any]]) -> int:
        """Restore strategy state from historical events (crash recovery).

        This method is called on startup to restore any internal state
        the strategy needs. Strategies that are stateless can return 0.

        Args:
            events: Historical events from EventStore

        Returns:
            Number of events processed for state recovery
        """
        ...
