"""Market data service for managing mark price windows and updates.

Extracted from PerpsBot to separate concerns and improve testability.
Phase 1 of PerpsBot refactoring (2025-10-01).
"""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from bot_v2.features.brokerages.core.interfaces import IBrokerage
    from bot_v2.features.live_trade.risk import LiveRiskManager

logger = logging.getLogger(__name__)


class MarketDataService:
    """Manages mark price retrieval and window maintenance for trading symbols.

    This service encapsulates:
    - Fetching quotes from broker
    - Updating mark price windows (thread-safe)
    - Trimming windows to configured size
    - Updating risk manager timestamps

    Design Notes:
    - Thread-safe via shared _mark_lock (RLock)
    - Errors on individual symbols don't block others
    - No event_store writes (streaming handles that separately)
    """

    def __init__(
        self,
        symbols: list[str],
        broker: IBrokerage,
        risk_manager: LiveRiskManager,
        long_ma: int,
        short_ma: int,
        mark_lock: threading.RLock,
        mark_windows: dict[str, list[Decimal]] | None = None,
    ) -> None:
        """Initialize market data service.

        Args:
            symbols: Trading symbols to track
            broker: Brokerage interface for quotes
            risk_manager: Risk manager to update timestamps
            long_ma: Long moving average period (for window sizing)
            short_ma: Short moving average period (for window sizing)
            mark_lock: Shared RLock for thread-safe mark updates
            mark_windows: Optional pre-existing mark windows dict (for migration)
        """
        self.symbols = symbols
        self.broker = broker
        self.risk_manager = risk_manager
        self.long_ma = long_ma
        self.short_ma = short_ma
        self._mark_lock = mark_lock
        self._mark_windows = mark_windows or {s: [] for s in symbols}

    @property
    def mark_windows(self) -> dict[str, list[Decimal]]:
        """Expose mark windows (for backward compatibility during migration)."""
        return self._mark_windows

    async def update_marks(self) -> None:
        """Update mark prices for all symbols.

        For each symbol:
        1. Fetch quote from broker
        2. Extract price and timestamp
        3. Update mark window (thread-safe)
        4. Update risk manager timestamp
        5. Log errors but continue to next symbol

        Side Effects (MUST preserve exactly):
        - Updates mark_windows[symbol] (thread-safe via _mark_lock)
        - Updates risk_manager.last_mark_update[symbol] with timestamp
        - Trims mark_windows to max(long_ma, short_ma) + 5
        - Logs errors but continues processing other symbols
        """
        for symbol in self.symbols:
            try:
                quote = await asyncio.to_thread(self.broker.get_quote, symbol)
                if quote is None:
                    raise RuntimeError(f"No quote for {symbol}")
                last_price = getattr(quote, "last", getattr(quote, "last_price", None))
                if last_price is None:
                    raise RuntimeError(f"Quote missing price for {symbol}")
                mark = Decimal(str(last_price))
                if mark <= 0:
                    raise RuntimeError(f"Invalid mark price: {mark} for {symbol}")
                ts = getattr(quote, "ts", datetime.now(UTC))
                self._update_mark_window(symbol, mark)
                try:
                    self.risk_manager.last_mark_update[symbol] = (
                        ts if isinstance(ts, datetime) else datetime.utcnow()
                    )
                except Exception as exc:
                    logger.debug(
                        "Failed to update mark timestamp for %s: %s", symbol, exc, exc_info=True
                    )
            except Exception as exc:
                logger.error("Error updating mark for %s: %s", symbol, exc)

    def _update_mark_window(self, symbol: str, mark: Decimal) -> None:
        """Update mark window for symbol (thread-safe).

        Args:
            symbol: Trading symbol
            mark: Mark price to append

        Side Effects:
        - Appends mark to mark_windows[symbol]
        - Trims window to max(long_ma, short_ma) + 5
        - Thread-safe via _mark_lock
        """
        with self._mark_lock:
            if symbol not in self._mark_windows:
                self._mark_windows[symbol] = []
            self._mark_windows[symbol].append(mark)
            max_size = max(self.short_ma, self.long_ma) + 5
            if len(self._mark_windows[symbol]) > max_size:
                self._mark_windows[symbol] = self._mark_windows[symbol][-max_size:]
