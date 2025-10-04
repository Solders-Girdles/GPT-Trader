"""
Main paper trading orchestration - entry point for the slice.

Complete isolation - everything needed is local.
"""

import logging
from datetime import datetime
from typing import Any

from bot_v2.features.paper_trade.data import DataFeed
from bot_v2.features.paper_trade.execution import PaperExecutor
from bot_v2.features.paper_trade.performance import (
    PerformanceCalculator,
    PerformanceTracker,
    ResultBuilder,
)
from bot_v2.features.paper_trade.risk import RiskManager
from bot_v2.features.paper_trade.session_config import SessionConfigBuilder
from bot_v2.features.paper_trade.strategies import create_paper_strategy
from bot_v2.features.paper_trade.strategy_runner import StrategyRunner
from bot_v2.features.paper_trade.trading_loop import TradingLoop
from bot_v2.features.paper_trade.types import PaperTradeResult, PerformanceMetrics
from bot_v2.types.trading import TradingSessionResult

logger = logging.getLogger(__name__)


class PaperTradingSession:
    """Manages a paper trading session."""

    def __init__(
        self,
        strategy: str,
        symbols: list[str],
        initial_capital: float = 100000,
        **kwargs: Any,
    ) -> None:
        """
        Initialize paper trading session.

        Args:
            strategy: Strategy name
            symbols: List of symbols to trade
            initial_capital: Starting capital
            **kwargs: Strategy parameters and settings
        """
        # Build configuration from kwargs
        config = SessionConfigBuilder.from_kwargs(strategy, symbols, initial_capital, **kwargs)

        # Store configuration attributes
        self.strategy_name = config.strategy_name
        self.symbols = config.symbols
        self.initial_capital = config.initial_capital
        self.commission = config.commission
        self.slippage = config.slippage
        self.max_positions = config.max_positions
        self.position_size = config.position_size
        self.update_interval = config.update_interval

        # Initialize components
        self.strategy = create_paper_strategy(config.strategy_name, **config.strategy_params)
        self.data_feed = DataFeed(config.symbols)
        self.executor = PaperExecutor(
            initial_capital=config.initial_capital,
            commission=config.commission,
            slippage=config.slippage,
            max_positions=config.max_positions,
        )
        self.risk_manager = RiskManager()

        # Performance tracking
        self.performance_tracker = PerformanceTracker(config.initial_capital)
        self.performance_calculator = PerformanceCalculator(self.performance_tracker)
        self.result_builder = ResultBuilder(
            tracker=self.performance_tracker,
            calculator=self.performance_calculator,
        )

        # Strategy runner (processes signals for each symbol)
        self.strategy_runner = StrategyRunner(
            strategy=self.strategy,
            data_feed=self.data_feed,
            risk_manager=self.risk_manager,
            executor=self.executor,
            position_size=config.position_size,
        )

        # Trading loop (manages background thread)
        self.trading_loop = TradingLoop(
            symbols=config.symbols,
            update_interval=config.update_interval,
            data_feed=self.data_feed,
            executor=self.executor,
            on_process_symbol=self.strategy_runner.process_symbol,
            on_record_equity=self._record_equity_point,
        )

        # Session state
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None

    def start(self) -> None:
        """Start paper trading session."""
        if self.trading_loop.is_running:
            return
        self.start_time = datetime.now()
        self.trading_loop.start()

    def stop(self) -> PaperTradeResult:
        """Stop paper trading session and return results."""
        if not self.trading_loop.is_running:
            return self.get_results()

        self.trading_loop.stop()
        self.end_time = datetime.now()

        # Close all positions with current prices
        price_map = {s: p for s in self.symbols if (p := self.data_feed.get_latest_price(s))}
        self.executor.close_all_positions(price_map, self.end_time)

        return self.get_results()

    @property
    def is_running(self) -> bool:
        return self.trading_loop.is_running

    @property
    def equity_history(self) -> list[dict[str, Any]]:
        return self.performance_tracker.legacy_history

    @equity_history.setter
    def equity_history(self, value: list[dict[str, Any]]) -> None:
        self.performance_tracker.replace_history(value)

    def _record_equity_point(self, equity: float) -> None:
        """Record equity point (callback from trading loop)."""
        self.performance_tracker.record(datetime.now(), equity)

    def get_results(self) -> PaperTradeResult:
        """Get current results using the legacy paper trade schema."""
        account = self.executor.get_account_status()
        return self.result_builder.build_paper_result(
            start_time=self.start_time or datetime.now(),
            end_time=self.end_time,
            account_status=account,
            positions=list(self.executor.positions.values()),
            trade_log=self.executor.trade_log,
        )

    def get_trading_session(self) -> TradingSessionResult:
        """Return results using the shared trading type schema."""
        return self.get_results().to_trading_session()

    def _calculate_metrics(self) -> PerformanceMetrics:
        """Legacy metrics calculator wrapper (for compatibility)."""
        account = self.executor.get_account_status()
        return self.performance_calculator.calculate(
            trade_log=self.executor.trade_log,
            account_status=account,
        )


# Global session management
class _SessionManager:
    """Manages the global paper trading session."""

    def __init__(self) -> None:
        self._session: PaperTradingSession | None = None

    def start(
        self, strategy: str, symbols: list[str], initial_capital: float = 100000, **kwargs: Any
    ) -> None:
        """Start a new paper trading session."""
        if self._session and self._session.is_running:
            raise RuntimeError("A paper trading session is already running")
        self._session = PaperTradingSession(strategy, symbols, initial_capital, **kwargs)
        self._session.start()

    def stop(self) -> PaperTradeResult:
        """Stop the active session and return results."""
        if not self._session:
            raise RuntimeError("No active paper trading session")
        results = self._session.stop()
        self._session = None
        return results

    def get_status(self) -> PaperTradeResult | None:
        """Get current session results or None if no session."""
        return self._session.get_results() if self._session else None

    def get_trading_session(self) -> TradingSessionResult | None:
        """Get current session as TradingSessionResult or None."""
        return self._session.get_trading_session() if self._session else None


_manager = _SessionManager()


def start_paper_trading(
    strategy: str, symbols: list[str], initial_capital: float = 100000, **kwargs: Any
) -> None:
    """Start a paper trading session."""
    _manager.start(strategy, symbols, initial_capital, **kwargs)


def stop_paper_trading() -> PaperTradeResult:
    """Stop the active paper trading session."""
    return _manager.stop()


def get_status() -> PaperTradeResult | None:
    """Get current status of paper trading session."""
    return _manager.get_status()


def get_trading_session() -> TradingSessionResult | None:
    """Return current session summary using shared trading types."""
    return _manager.get_trading_session()
