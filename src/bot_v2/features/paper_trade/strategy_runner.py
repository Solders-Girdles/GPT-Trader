"""
Strategy Runner - Per-symbol signal processing and execution.

Encapsulates the logic for processing a single symbol:
- Fetching historical data
- Generating trading signals
- Risk checking
- Signal execution

Extracted from PaperTradingSession to improve testability
and separation of concerns.
"""

import logging
from datetime import datetime

from bot_v2.features.paper_trade.data import DataFeed
from bot_v2.features.paper_trade.execution import PaperExecutor
from bot_v2.features.paper_trade.risk import RiskManager
from bot_v2.features.paper_trade.strategies import PaperTradeStrategy

logger = logging.getLogger(__name__)


class StrategyRunner:
    """
    Processes trading signals for individual symbols.

    Handles the complete signal processing pipeline:
    1. Fetch historical data
    2. Generate signal via strategy
    3. Check risk limits
    4. Execute signal via executor

    Designed for testability with all dependencies injected.
    """

    def __init__(
        self,
        strategy: PaperTradeStrategy,
        data_feed: DataFeed,
        risk_manager: RiskManager,
        executor: PaperExecutor,
        position_size: float,
    ) -> None:
        """
        Initialize strategy runner.

        Args:
            strategy: Trading strategy for signal generation
            data_feed: Data feed for historical and current prices
            risk_manager: Risk manager for trade validation
            executor: Execution engine for trade execution
            position_size: Position sizing parameter (e.g., 0.95 for 95% of available capital)
        """
        self.strategy = strategy
        self.data_feed = data_feed
        self.risk_manager = risk_manager
        self.executor = executor
        self.position_size = position_size

    def process_symbol(self, symbol: str) -> None:
        """
        Process trading logic for a single symbol.

        Fetches historical data, generates signal, checks risk limits,
        and executes the signal if all checks pass.

        Args:
            symbol: Symbol to process (e.g., "AAPL")

        Note:
            Exceptions are caught and logged, allowing the runner
            to continue processing other symbols.
        """
        try:
            # Get historical data
            data = self.data_feed.get_historical(symbol, self.strategy.get_required_periods())

            if data.empty or len(data) < self.strategy.get_required_periods():
                return

            # Generate signal
            signal = self.strategy.analyze(data)

            # Check risk limits
            if signal != 0:
                current_price = self.data_feed.get_latest_price(symbol)
                if current_price:
                    # Apply risk checks
                    account = self.executor.get_account_status()
                    if not self.risk_manager.check_trade(symbol, signal, current_price, account):
                        return

                    # Execute signal
                    self.executor.execute_signal(
                        symbol=symbol,
                        signal=signal,
                        current_price=current_price,
                        timestamp=datetime.now(),
                        position_size=self.position_size,
                    )
        except Exception as e:
            logger.warning("Error processing symbol %s: %s", symbol, e, exc_info=True)
