"""
Base class for stateful trading strategies.

Provides built-in support for:
- Incremental statistics (O(1) mean/variance)
- State rehydration from EventStore
- Position state tracking
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from gpt_trader.core import Decision, Product
from gpt_trader.core.math.incremental import IncrementalStats
from gpt_trader.features.live_trade.interfaces import TradingStrategy
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.features.live_trade.strategies.base import MarketDataContext

logger = get_logger(__name__, component="stateful_strategy")


class StatefulStrategy(TradingStrategy, ABC):
    """
    Base class for strategies that maintain state across ticks and restarts.

    Automatically manages IncrementalStats for each symbol and handles
    rehydration from historical events.
    """

    def __init__(self) -> None:
        self.stats: dict[str, IncrementalStats] = defaultdict(IncrementalStats)
        self._initialized = False

    def update_stats(self, symbol: str, price: Decimal) -> None:
        """Update incremental statistics for a symbol."""
        self.stats[symbol].update(price)

    def rehydrate(self, events: Sequence[dict[str, Any]]) -> int:
        """
        Restore strategy state from historical events.

        Replays 'price_tick' events to rebuild incremental statistics.

        Args:
            events: Sequence of historical events from EventStore.

        Returns:
            Number of events processed.
        """
        processed = 0
        for event in events:
            if event.get("type") == "price_tick":
                data = event.get("data", {})
                symbol = data.get("symbol")
                price_str = data.get("price")

                if symbol and price_str:
                    try:
                        price = Decimal(str(price_str))
                        self.update_stats(symbol, price)
                        processed += 1
                    except (ValueError, ArithmeticError) as e:
                        logger.debug("Failed to parse price during rehydration: %s", e)
                        continue

        if processed > 0:
            logger.info(f"Rehydrated strategy state from {processed} events")

        self._initialized = True
        return processed

    @abstractmethod
    def decide(
        self,
        symbol: str,
        current_mark: Decimal,
        position_state: dict[str, Any] | None,
        recent_marks: Sequence[Decimal],
        equity: Decimal,
        product: Product | None,
        market_data: MarketDataContext | None = None,
        candles: Sequence[Any] | None = None,
    ) -> Decision:
        """
        Generate a trading decision.

        Must be implemented by subclasses. Subclasses should call
        `self.update_stats(symbol, current_mark)` at the start of this method
        if they want to maintain up-to-date statistics.
        """
        ...
