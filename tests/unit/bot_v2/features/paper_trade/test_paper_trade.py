"""
Comprehensive tests for paper trading orchestration.

Tests cover:
- PaperTradingSession initialization and configuration
- Session lifecycle (start/stop)
- Trading loop execution
- Symbol processing and signal generation
- Risk integration
- Results calculation and metrics
- Equity curve tracking
- Global session management
- Edge cases and error handling
"""

from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from bot_v2.features.paper_trade.paper_trade import (
    PaperTradingSession,
    get_status,
    get_trading_session,
    start_paper_trading,
    stop_paper_trading,
)
from bot_v2.features.paper_trade.types import PaperTradeResult


# ============================================================================
# Test: PaperTradingSession Initialization
# ============================================================================


class TestPaperTradingSessionInitialization:
    """Test PaperTradingSession initialization."""

    def test_initialization_default_params(self):
        """Test session initialization with default parameters."""
        session = PaperTradingSession(
            strategy="SimpleMAStrategy", symbols=["AAPL", "MSFT"], initial_capital=100000
        )

        assert session.strategy_name == "SimpleMAStrategy"
        assert session.symbols == ["AAPL", "MSFT"]
        assert session.initial_capital == 100000
        assert session.commission == 0.001
        assert session.slippage == 0.0005
        assert session.max_positions == 10
        assert session.position_size == 0.95
        assert session.update_interval == 60
        assert session.is_running is False
        assert session.start_time is None
        assert session.end_time is None

    def test_initialization_custom_params(self):
        """Test session initialization with custom parameters."""
        session = PaperTradingSession(
            strategy="MomentumStrategy",
            symbols=["BTC-USD"],
            initial_capital=50000,
            commission=0.002,
            slippage=0.001,
            max_positions=5,
            position_size=0.8,
            update_interval=30,
            lookback=15,
        )

        assert session.initial_capital == 50000
        assert session.commission == 0.002
        assert session.slippage == 0.001
        assert session.max_positions == 5
        assert session.position_size == 0.8
        assert session.update_interval == 30

    def test_initialization_creates_components(self):
        """Test that initialization creates all necessary components."""
        session = PaperTradingSession(
            strategy="SimpleMAStrategy", symbols=["AAPL"], initial_capital=100000
        )

        assert session.strategy is not None
        assert session.data_feed is not None
        assert session.executor is not None
        assert session.risk_manager is not None

    def test_initialization_with_strategy_params(self):
        """Test initialization passes strategy parameters correctly."""
        session = PaperTradingSession(
            strategy="SimpleMAStrategy",
            symbols=["AAPL"],
            initial_capital=100000,
            fast_period=5,
            slow_period=20,
        )

        assert session.strategy.fast_period == 5
        assert session.strategy.slow_period == 20

    def test_initialization_multiple_symbols(self):
        """Test initialization with multiple symbols."""
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        session = PaperTradingSession(
            strategy="SimpleMAStrategy", symbols=symbols, initial_capital=100000
        )

        assert session.symbols == symbols


# ============================================================================
# Test: Session Start
# ============================================================================


class TestSessionStart:
    """Test paper trading session start."""

    @patch("bot_v2.features.paper_trade.paper_trade.threading.Thread")
    def test_start_sets_running_state(self, mock_thread):
        """Test that start sets session to running state."""
        session = PaperTradingSession(
            strategy="SimpleMAStrategy", symbols=["AAPL"], initial_capital=100000
        )

        session.start()

        assert session.is_running is True
        assert session.start_time is not None

    @patch("bot_v2.features.paper_trade.paper_trade.threading.Thread")
    def test_start_creates_thread(self, mock_thread):
        """Test that start creates background thread."""
        session = PaperTradingSession(
            strategy="SimpleMAStrategy", symbols=["AAPL"], initial_capital=100000
        )

        session.start()

        mock_thread.assert_called_once()
        # Check daemon flag
        call_kwargs = mock_thread.call_args[1]
        assert "target" in call_kwargs

    @patch("bot_v2.features.paper_trade.paper_trade.threading.Thread")
    def test_start_already_running(self, mock_thread):
        """Test that start does nothing if already running."""
        session = PaperTradingSession(
            strategy="SimpleMAStrategy", symbols=["AAPL"], initial_capital=100000
        )

        session.start()
        first_start_time = session.start_time

        # Try starting again
        session.start()

        # Should not create new thread or change start time
        assert mock_thread.call_count == 1
        assert session.start_time == first_start_time

    @patch("bot_v2.features.paper_trade.paper_trade.threading.Thread")
    @patch("builtins.print")
    def test_start_prints_status(self, mock_print, mock_thread):
        """Test that start prints session information."""
        session = PaperTradingSession(
            strategy="SimpleMAStrategy", symbols=["AAPL", "MSFT"], initial_capital=100000
        )

        session.start()

        # Should print multiple status lines
        assert mock_print.call_count >= 1


# ============================================================================
# Test: Session Stop
# ============================================================================


class TestSessionStop:
    """Test paper trading session stop."""

    @patch("bot_v2.features.paper_trade.paper_trade.threading.Thread")
    def test_stop_sets_not_running(self, mock_thread):
        """Test that stop sets session to not running."""
        session = PaperTradingSession(
            strategy="SimpleMAStrategy", symbols=["AAPL"], initial_capital=100000
        )

        # Mock data feed
        session.data_feed.get_latest_price = Mock(return_value=150.0)

        session.start()
        result = session.stop()

        assert session.is_running is False
        assert session.end_time is not None
        assert isinstance(result, PaperTradeResult)

    @patch("bot_v2.features.paper_trade.paper_trade.threading.Thread")
    def test_stop_closes_all_positions(self, mock_thread):
        """Test that stop closes all open positions."""
        session = PaperTradingSession(
            strategy="SimpleMAStrategy", symbols=["AAPL", "MSFT"], initial_capital=100000
        )

        # Mock data feed
        session.data_feed.get_latest_price = Mock(return_value=150.0)

        # Mock executor
        session.executor.close_all_positions = Mock()

        session.start()
        session.stop()

        session.executor.close_all_positions.assert_called_once()

    @patch("bot_v2.features.paper_trade.paper_trade.threading.Thread")
    def test_stop_when_not_running(self, mock_thread):
        """Test stop when session is not running."""
        session = PaperTradingSession(
            strategy="SimpleMAStrategy", symbols=["AAPL"], initial_capital=100000
        )

        # Don't start, just stop
        result = session.stop()

        assert isinstance(result, PaperTradeResult)
        assert session.is_running is False

    @patch("bot_v2.features.paper_trade.paper_trade.threading.Thread")
    def test_stop_waits_for_thread(self, mock_thread):
        """Test that stop waits for thread to finish."""
        mock_thread_instance = Mock()
        mock_thread.return_value = mock_thread_instance

        session = PaperTradingSession(
            strategy="SimpleMAStrategy", symbols=["AAPL"], initial_capital=100000
        )

        session.data_feed.get_latest_price = Mock(return_value=150.0)

        session.start()
        session.stop()

        # Should call join with timeout
        mock_thread_instance.join.assert_called_once()


# ============================================================================
# Test: Trading Loop
# ============================================================================


class TestTradingLoop:
    """Test main trading loop logic."""

    def test_process_symbol_insufficient_data(self):
        """Test symbol processing with insufficient data."""
        session = PaperTradingSession(
            strategy="SimpleMAStrategy", symbols=["AAPL"], initial_capital=100000
        )

        # Mock data feed to return insufficient data
        session.data_feed.get_historical = Mock(return_value=pd.DataFrame())
        session.executor.execute_signal = Mock()

        session._process_symbol("AAPL")

        # Should not execute any signals
        session.executor.execute_signal.assert_not_called()

    def test_process_symbol_with_valid_data(self):
        """Test symbol processing with valid data."""
        session = PaperTradingSession(
            strategy="SimpleMAStrategy", symbols=["AAPL"], initial_capital=100000
        )

        # Create valid data
        data = pd.DataFrame(
            {"open": [100] * 50, "high": [101] * 50, "low": [99] * 50, "close": [100] * 50}
        )

        session.data_feed.get_historical = Mock(return_value=data)
        session.data_feed.get_latest_price = Mock(return_value=100.0)
        session.strategy.analyze = Mock(return_value=1)  # Buy signal
        session.risk_manager.check_trade = Mock(return_value=True)
        session.executor.execute_signal = Mock()

        session._process_symbol("AAPL")

        # Should execute signal
        session.executor.execute_signal.assert_called_once()

    def test_process_symbol_zero_signal(self):
        """Test symbol processing when strategy returns zero signal."""
        session = PaperTradingSession(
            strategy="SimpleMAStrategy", symbols=["AAPL"], initial_capital=100000
        )

        data = pd.DataFrame(
            {"open": [100] * 50, "high": [101] * 50, "low": [99] * 50, "close": [100] * 50}
        )

        session.data_feed.get_historical = Mock(return_value=data)
        session.strategy.analyze = Mock(return_value=0)  # Hold signal
        session.executor.execute_signal = Mock()

        session._process_symbol("AAPL")

        # Should not execute signal
        session.executor.execute_signal.assert_not_called()

    def test_process_symbol_risk_rejection(self):
        """Test symbol processing when risk manager rejects trade."""
        session = PaperTradingSession(
            strategy="SimpleMAStrategy", symbols=["AAPL"], initial_capital=100000
        )

        data = pd.DataFrame(
            {"open": [100] * 50, "high": [101] * 50, "low": [99] * 50, "close": [100] * 50}
        )

        session.data_feed.get_historical = Mock(return_value=data)
        session.data_feed.get_latest_price = Mock(return_value=100.0)
        session.strategy.analyze = Mock(return_value=1)  # Buy signal
        session.risk_manager.check_trade = Mock(return_value=False)  # Reject
        session.executor.execute_signal = Mock()

        session._process_symbol("AAPL")

        # Should not execute signal
        session.executor.execute_signal.assert_not_called()

    def test_process_symbol_no_current_price(self):
        """Test symbol processing when current price is unavailable."""
        session = PaperTradingSession(
            strategy="SimpleMAStrategy", symbols=["AAPL"], initial_capital=100000
        )

        data = pd.DataFrame(
            {"open": [100] * 50, "high": [101] * 50, "low": [99] * 50, "close": [100] * 50}
        )

        session.data_feed.get_historical = Mock(return_value=data)
        session.data_feed.get_latest_price = Mock(return_value=None)
        session.strategy.analyze = Mock(return_value=1)  # Buy signal
        session.executor.execute_signal = Mock()

        session._process_symbol("AAPL")

        # Should not execute signal
        session.executor.execute_signal.assert_not_called()


# ============================================================================
# Test: Results and Metrics
# ============================================================================


class TestResultsAndMetrics:
    """Test results calculation and metrics."""

    def test_get_results_initial_state(self):
        """Test get_results at initial state."""
        session = PaperTradingSession(
            strategy="SimpleMAStrategy", symbols=["AAPL"], initial_capital=100000
        )

        result = session.get_results()

        assert isinstance(result, PaperTradeResult)
        assert result.account_status.cash == 100000
        assert result.account_status.total_equity == 100000
        assert len(result.trade_log) == 0
        assert result.performance.total_return == 0.0

    def test_get_results_with_equity_history(self):
        """Test get_results with equity history."""
        session = PaperTradingSession(
            strategy="SimpleMAStrategy", symbols=["AAPL"], initial_capital=100000
        )

        # Add equity history
        session.equity_history = [
            {"timestamp": datetime.now(), "equity": 100000},
            {"timestamp": datetime.now(), "equity": 105000},
            {"timestamp": datetime.now(), "equity": 110000},
        ]

        result = session.get_results()

        assert len(result.equity_curve) == 3
        assert result.equity_curve.iloc[-1] == 110000

    def test_calculate_metrics_profitable(self):
        """Test metrics calculation with profitable trading."""
        session = PaperTradingSession(
            strategy="SimpleMAStrategy", symbols=["AAPL"], initial_capital=100000
        )

        # Simulate profitable equity history
        session.equity_history = [
            {"timestamp": datetime.now(), "equity": 100000},
            {"timestamp": datetime.now(), "equity": 102000},
            {"timestamp": datetime.now(), "equity": 105000},
            {"timestamp": datetime.now(), "equity": 108000},
        ]

        metrics = session._calculate_metrics()

        assert metrics.total_return > 0
        assert metrics.max_drawdown >= 0

    def test_calculate_metrics_losing(self):
        """Test metrics calculation with losing trading."""
        session = PaperTradingSession(
            strategy="SimpleMAStrategy", symbols=["AAPL"], initial_capital=100000
        )

        # Simulate losing equity history
        session.equity_history = [
            {"timestamp": datetime.now(), "equity": 100000},
            {"timestamp": datetime.now(), "equity": 98000},
            {"timestamp": datetime.now(), "equity": 95000},
            {"timestamp": datetime.now(), "equity": 92000},
        ]

        metrics = session._calculate_metrics()

        assert metrics.total_return < 0
        assert metrics.max_drawdown > 0

    def test_calculate_metrics_no_history(self):
        """Test metrics calculation with no equity history."""
        session = PaperTradingSession(
            strategy="SimpleMAStrategy", symbols=["AAPL"], initial_capital=100000
        )

        metrics = session._calculate_metrics()

        assert metrics.total_return == 0.0
        assert metrics.daily_return == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.max_drawdown == 0.0
        assert metrics.trades_count == 0

    def test_calculate_metrics_max_drawdown(self):
        """Test max drawdown calculation."""
        session = PaperTradingSession(
            strategy="SimpleMAStrategy", symbols=["AAPL"], initial_capital=100000
        )

        # Create equity curve with drawdown
        session.equity_history = [
            {"timestamp": datetime.now(), "equity": 100000},
            {"timestamp": datetime.now(), "equity": 110000},  # Peak
            {"timestamp": datetime.now(), "equity": 99000},  # 10% drawdown
            {"timestamp": datetime.now(), "equity": 105000},
        ]

        metrics = session._calculate_metrics()

        # Max drawdown should be 10%
        assert metrics.max_drawdown == pytest.approx(0.1, rel=1e-6)

    def test_get_trading_session_conversion(self):
        """Test conversion to TradingSessionResult."""
        session = PaperTradingSession(
            strategy="SimpleMAStrategy", symbols=["AAPL"], initial_capital=100000
        )

        trading_session = session.get_trading_session()

        assert trading_session is not None


# ============================================================================
# Test: Global Session Management
# ============================================================================


class TestGlobalSessionManagement:
    """Test global session management functions."""

    @patch("bot_v2.features.paper_trade.paper_trade.PaperTradingSession")
    def test_start_paper_trading(self, mock_session_class):
        """Test starting paper trading via global function."""
        mock_session = Mock()
        mock_session.is_running = False
        mock_session_class.return_value = mock_session

        # Reset global state
        import bot_v2.features.paper_trade.paper_trade as pt_module

        pt_module._active_session = None

        start_paper_trading(strategy="SimpleMAStrategy", symbols=["AAPL"], initial_capital=100000)

        mock_session_class.assert_called_once()
        mock_session.start.assert_called_once()

    @patch("bot_v2.features.paper_trade.paper_trade.PaperTradingSession")
    def test_start_paper_trading_already_running(self, mock_session_class):
        """Test error when trying to start while session is running."""
        import bot_v2.features.paper_trade.paper_trade as pt_module

        # Set up active running session
        mock_session = Mock()
        mock_session.is_running = True
        pt_module._active_session = mock_session

        with pytest.raises(RuntimeError, match="already running"):
            start_paper_trading(
                strategy="SimpleMAStrategy", symbols=["AAPL"], initial_capital=100000
            )

        # Cleanup
        pt_module._active_session = None

    @patch("bot_v2.features.paper_trade.paper_trade.PaperTradingSession")
    def test_stop_paper_trading(self, mock_session_class):
        """Test stopping paper trading via global function."""
        import bot_v2.features.paper_trade.paper_trade as pt_module

        mock_session = Mock()
        mock_result = Mock(spec=PaperTradeResult)
        mock_session.stop.return_value = mock_result
        pt_module._active_session = mock_session

        result = stop_paper_trading()

        assert result == mock_result
        mock_session.stop.assert_called_once()
        assert pt_module._active_session is None

        # Cleanup
        pt_module._active_session = None

    def test_stop_paper_trading_no_session(self):
        """Test error when trying to stop without active session."""
        import bot_v2.features.paper_trade.paper_trade as pt_module

        pt_module._active_session = None

        with pytest.raises(RuntimeError, match="No active paper trading session"):
            stop_paper_trading()

    @patch("bot_v2.features.paper_trade.paper_trade.PaperTradingSession")
    def test_get_status_with_session(self, mock_session_class):
        """Test getting status with active session."""
        import bot_v2.features.paper_trade.paper_trade as pt_module

        mock_session = Mock()
        mock_result = Mock(spec=PaperTradeResult)
        mock_session.get_results.return_value = mock_result
        pt_module._active_session = mock_session

        result = get_status()

        assert result == mock_result
        mock_session.get_results.assert_called_once()

        # Cleanup
        pt_module._active_session = None

    def test_get_status_no_session(self):
        """Test getting status without active session."""
        import bot_v2.features.paper_trade.paper_trade as pt_module

        pt_module._active_session = None

        result = get_status()

        assert result is None

    @patch("bot_v2.features.paper_trade.paper_trade.PaperTradingSession")
    def test_get_trading_session_with_session(self, mock_session_class):
        """Test getting trading session with active session."""
        import bot_v2.features.paper_trade.paper_trade as pt_module

        mock_session = Mock()
        mock_result = Mock()
        mock_session.get_trading_session.return_value = mock_result
        pt_module._active_session = mock_session

        result = get_trading_session()

        assert result == mock_result
        mock_session.get_trading_session.assert_called_once()

        # Cleanup
        pt_module._active_session = None

    def test_get_trading_session_no_session(self):
        """Test getting trading session without active session."""
        import bot_v2.features.paper_trade.paper_trade as pt_module

        pt_module._active_session = None

        result = get_trading_session()

        assert result is None


# ============================================================================
# Test: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_initialization_with_empty_symbols(self):
        """Test initialization with empty symbols list."""
        session = PaperTradingSession(
            strategy="SimpleMAStrategy", symbols=[], initial_capital=100000
        )

        assert session.symbols == []

    def test_initialization_zero_capital(self):
        """Test initialization with zero capital."""
        session = PaperTradingSession(
            strategy="SimpleMAStrategy", symbols=["AAPL"], initial_capital=0
        )

        assert session.initial_capital == 0

    def test_initialization_negative_capital(self):
        """Test initialization with negative capital."""
        session = PaperTradingSession(
            strategy="SimpleMAStrategy", symbols=["AAPL"], initial_capital=-1000
        )

        assert session.initial_capital == -1000

    def test_process_symbol_with_exception(self):
        """Test symbol processing handles exceptions gracefully."""
        session = PaperTradingSession(
            strategy="SimpleMAStrategy", symbols=["AAPL"], initial_capital=100000
        )

        # Mock to raise exception
        session.data_feed.get_historical = Mock(side_effect=Exception("Test error"))

        # Should not raise exception
        try:
            session._process_symbol("AAPL")
        except Exception:
            pytest.fail("_process_symbol should handle exceptions")

    def test_calculate_metrics_with_zero_std(self):
        """Test metrics calculation with zero standard deviation."""
        session = PaperTradingSession(
            strategy="SimpleMAStrategy", symbols=["AAPL"], initial_capital=100000
        )

        # Constant equity (zero std)
        session.equity_history = [
            {"timestamp": datetime.now(), "equity": 100000},
            {"timestamp": datetime.now(), "equity": 100000},
            {"timestamp": datetime.now(), "equity": 100000},
        ]

        metrics = session._calculate_metrics()

        # Should handle zero std gracefully
        assert metrics.sharpe_ratio == 0.0

    def test_get_results_no_start_time(self):
        """Test get_results when start_time is None."""
        session = PaperTradingSession(
            strategy="SimpleMAStrategy", symbols=["AAPL"], initial_capital=100000
        )

        result = session.get_results()

        # Should use current time as fallback
        assert result.start_time is not None

    def test_multiple_start_stop_cycles(self):
        """Test multiple start/stop cycles."""
        with patch("bot_v2.features.paper_trade.paper_trade.threading.Thread"):
            session = PaperTradingSession(
                strategy="SimpleMAStrategy", symbols=["AAPL"], initial_capital=100000
            )

            session.data_feed.get_latest_price = Mock(return_value=150.0)

            # First cycle
            session.start()
            session.stop()

            assert session.is_running is False

            # Second cycle - should be able to start again
            session.start()
            assert session.is_running is True

    def test_equity_history_accumulation(self):
        """Test that equity history accumulates over time."""
        session = PaperTradingSession(
            strategy="SimpleMAStrategy", symbols=["AAPL"], initial_capital=100000
        )

        # Manually add equity points
        session.equity_history.append({"timestamp": datetime.now(), "equity": 100000})
        session.equity_history.append({"timestamp": datetime.now(), "equity": 105000})
        session.equity_history.append({"timestamp": datetime.now(), "equity": 103000})

        assert len(session.equity_history) == 3

    def test_update_interval_zero(self):
        """Test with zero update interval."""
        session = PaperTradingSession(
            strategy="SimpleMAStrategy",
            symbols=["AAPL"],
            initial_capital=100000,
            update_interval=0,
        )

        assert session.update_interval == 0

    def test_max_positions_limit(self):
        """Test with different max_positions limits."""
        # Very restrictive
        session1 = PaperTradingSession(
            strategy="SimpleMAStrategy",
            symbols=["AAPL"],
            initial_capital=100000,
            max_positions=1,
        )
        assert session1.max_positions == 1

        # Very permissive
        session2 = PaperTradingSession(
            strategy="SimpleMAStrategy",
            symbols=["AAPL"],
            initial_capital=100000,
            max_positions=100,
        )
        assert session2.max_positions == 100


# ============================================================================
# Test: Integration Scenarios
# ============================================================================


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    @patch("bot_v2.features.paper_trade.paper_trade.threading.Thread")
    def test_complete_session_lifecycle(self, mock_thread):
        """Test complete session lifecycle from start to finish."""
        session = PaperTradingSession(
            strategy="SimpleMAStrategy", symbols=["AAPL"], initial_capital=100000
        )

        # Mock data feed
        session.data_feed.get_latest_price = Mock(return_value=150.0)

        # Start session
        session.start()
        assert session.is_running is True
        assert session.start_time is not None

        # Get status while running
        status = session.get_results()
        assert isinstance(status, PaperTradeResult)

        # Stop session
        result = session.stop()
        assert session.is_running is False
        assert session.end_time is not None
        assert isinstance(result, PaperTradeResult)

    def test_multi_symbol_processing(self):
        """Test processing multiple symbols."""
        session = PaperTradingSession(
            strategy="SimpleMAStrategy",
            symbols=["AAPL", "MSFT", "GOOGL"],
            initial_capital=100000,
        )

        data = pd.DataFrame(
            {"open": [100] * 50, "high": [101] * 50, "low": [99] * 50, "close": [100] * 50}
        )

        session.data_feed.get_historical = Mock(return_value=data)
        session.data_feed.get_latest_price = Mock(return_value=100.0)
        session.strategy.analyze = Mock(return_value=0)  # Hold

        # Process each symbol
        for symbol in session.symbols:
            session._process_symbol(symbol)

        # Should process all symbols
        assert session.data_feed.get_historical.call_count == 3

    def test_strategy_params_passed_through(self):
        """Test that strategy parameters are passed correctly."""
        session = PaperTradingSession(
            strategy="MomentumStrategy",
            symbols=["AAPL"],
            initial_capital=100000,
            lookback=15,
            threshold=0.03,
        )

        assert session.strategy.lookback == 15
        assert session.strategy.threshold == 0.03