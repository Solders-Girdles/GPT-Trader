"""
Paper Trading Loop - Background thread execution.

Manages the main trading loop lifecycle, including thread management,
periodic updates, and error handling. Extracted from PaperTradingSession
to improve testability and separation of concerns.
"""

import logging
import threading
import time
from collections.abc import Callable

from bot_v2.features.paper_trade.data import DataFeed
from bot_v2.features.paper_trade.execution import PaperExecutor

logger = logging.getLogger(__name__)


class TradingLoop:
    """
    Manages background trading loop execution.

    Handles thread lifecycle, periodic data updates, symbol processing,
    position updates, and equity recording. Designed for testability
    with callback-based architecture.
    """

    def __init__(
        self,
        symbols: list[str],
        update_interval: int,
        data_feed: DataFeed,
        executor: PaperExecutor,
        on_process_symbol: Callable[[str], None],
        on_record_equity: Callable[[float], None],
    ) -> None:
        """
        Initialize trading loop.

        Args:
            symbols: List of symbols to trade
            update_interval: Seconds between loop iterations
            data_feed: Data feed for price updates
            executor: Execution engine for position management
            on_process_symbol: Callback to process each symbol
            on_record_equity: Callback to record equity point
        """
        self.symbols = symbols
        self.update_interval = update_interval
        self.data_feed = data_feed
        self.executor = executor
        self.on_process_symbol = on_process_symbol
        self.on_record_equity = on_record_equity

        # Thread state
        self.is_running = False
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the trading loop in a background thread."""
        if self.is_running:
            return

        self.is_running = True

        # Create and start daemon thread
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()

    def stop(self) -> None:
        """Stop the trading loop and wait for thread to finish."""
        if not self.is_running:
            return

        self.is_running = False

        # Wait for thread to finish
        if self.thread:
            self.thread.join(timeout=5)

    def _run(self) -> None:
        """
        Main trading loop (runs in background thread).

        Continuously:
        1. Updates data feed
        2. Processes each symbol via callback
        3. Updates positions with latest prices
        4. Records equity via callback
        5. Sleeps until next iteration
        6. Handles errors gracefully
        """
        while self.is_running:
            try:
                # Update data feed
                self.data_feed.update()

                # Process each symbol
                for symbol in self.symbols:
                    self.on_process_symbol(symbol)

                # Build price map for position updates
                price_map: dict[str, float] = {}
                for symbol in self.symbols:
                    price = self.data_feed.get_latest_price(symbol)
                    if price is not None:
                        price_map[symbol] = price

                # Update positions with current prices
                self.executor.update_positions(price_map)

                # Record current equity
                status = self.executor.get_account_status()
                self.on_record_equity(status.total_equity)

                # Sleep until next update
                time.sleep(float(self.update_interval))

            except Exception as e:
                logger.warning(f"Error in trading loop: {e}")
                # Continue after error, but still sleep to avoid tight loop
                time.sleep(float(self.update_interval))
