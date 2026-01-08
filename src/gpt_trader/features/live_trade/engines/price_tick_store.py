"""
Price tick persistence and state recovery for TradingEngine.

Extracted from TradingEngine to separate concerns:
- Persist price ticks to EventStore for crash recovery
- Rehydrate price history from persisted events on startup
"""

from __future__ import annotations

import time
from collections import defaultdict
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.app.protocols import EventStoreProtocol

logger = get_logger(__name__, component="price_tick_store")

# Event type for price ticks
EVENT_PRICE_TICK = "price_tick"

# Maximum number of prices to retain per symbol
MAX_PRICE_HISTORY = 20


class PriceTickStore:
    """
    Manages price tick persistence and recovery.

    Responsibilities:
    - Record price ticks to EventStore for crash recovery
    - Rehydrate price history from EventStore on startup
    - Maintain bounded price history per symbol
    """

    def __init__(
        self,
        event_store: EventStoreProtocol | None,
        symbols: list[str],
        bot_id: str,
    ) -> None:
        """
        Initialize the price tick store.

        Args:
            event_store: EventStore for persistence (can be None)
            symbols: List of symbols to track
            bot_id: Bot identifier for event tagging
        """
        self._event_store = event_store
        self._symbols = set(symbols)
        self._bot_id = bot_id
        self._price_history: dict[str, list[Decimal]] = defaultdict(list)

    @property
    def price_history(self) -> dict[str, list[Decimal]]:
        """Access the price history dictionary."""
        return self._price_history

    def rehydrate(self, strategy_rehydrate_callback: Any = None) -> int:
        """
        Restore price history from persisted events.

        Args:
            strategy_rehydrate_callback: Optional callback for strategy-specific
                rehydration (receives list of events)

        Returns:
            Number of price ticks restored
        """
        if self._event_store is None:
            logger.debug("No event store configured - skipping rehydration")
            return 0

        events = self._event_store.get_recent(count=1000)
        restored = 0

        for event in events:
            if event.get("type") != EVENT_PRICE_TICK:
                continue

            data = event.get("data", {})
            symbol = data.get("symbol")
            price_str = data.get("price")

            if not symbol or not price_str:
                continue

            # Only restore prices for symbols we're trading
            if symbol not in self._symbols:
                continue

            try:
                price = Decimal(str(price_str))
                self._price_history[symbol].append(price)
                # Keep history bounded
                if len(self._price_history[symbol]) > MAX_PRICE_HISTORY:
                    self._price_history[symbol].pop(0)
                restored += 1
            except Exception as e:
                logger.warning(f"Failed to parse price from event: {e}")

        if restored > 0:
            logger.info(f"Rehydrated {restored} price ticks from EventStore")
            for symbol, prices in self._price_history.items():
                logger.info(f"  {symbol}: {len(prices)} prices")

        # Call strategy rehydration if callback provided
        if strategy_rehydrate_callback is not None:
            strategy_rehydrate_callback(events)

        return restored

    def record_price_tick(self, symbol: str, price: Decimal) -> None:
        """
        Persist price tick to EventStore for crash recovery.

        Also updates the in-memory price history.

        Args:
            symbol: Trading symbol
            price: Current price
        """
        # Update in-memory history
        self._price_history[symbol].append(price)
        if len(self._price_history[symbol]) > MAX_PRICE_HISTORY:
            self._price_history[symbol].pop(0)

        # Persist to event store
        if self._event_store is None:
            return

        self._event_store.store(
            {
                "type": EVENT_PRICE_TICK,
                "data": {
                    "symbol": symbol,
                    "price": str(price),
                    "timestamp": time.time(),
                    "bot_id": self._bot_id,
                },
            }
        )
