"""
Tests for strategy runner.

Tests cover:
- Signal processing flow
- Data validation
- Risk checking
- Signal execution
- Error handling
- Edge cases
"""

from unittest.mock import Mock

import pandas as pd
import pytest

from bot_v2.features.paper_trade.strategy_runner import StrategyRunner


# ============================================================================
# Test: Signal Processing Flow
# ============================================================================


class TestStrategyRunnerSignalProcessing:
    """Test signal processing pipeline."""

    def test_process_symbol_insufficient_data(self):
        """Test that runner returns early if data is insufficient."""
        mock_strategy = Mock()
        mock_strategy.get_required_periods.return_value = 50

        mock_data_feed = Mock()
        # Return data with fewer periods than required
        mock_data_feed.get_historical.return_value = pd.DataFrame({"close": [100, 101, 102]})

        runner = StrategyRunner(
            strategy=mock_strategy,
            data_feed=mock_data_feed,
            risk_manager=Mock(),
            executor=Mock(),
            position_size=0.95,
        )

        runner.process_symbol("AAPL")

        # Should not call analyze if data insufficient
        mock_strategy.analyze.assert_not_called()

    def test_process_symbol_empty_data(self):
        """Test that runner returns early if data is empty."""
        mock_strategy = Mock()
        mock_strategy.get_required_periods.return_value = 50

        mock_data_feed = Mock()
        mock_data_feed.get_historical.return_value = pd.DataFrame()

        runner = StrategyRunner(
            strategy=mock_strategy,
            data_feed=mock_data_feed,
            risk_manager=Mock(),
            executor=Mock(),
            position_size=0.95,
        )

        runner.process_symbol("AAPL")

        # Should not call analyze if data empty
        mock_strategy.analyze.assert_not_called()

    def test_process_symbol_with_valid_data(self):
        """Test that runner processes symbol with valid data."""
        mock_strategy = Mock()
        mock_strategy.get_required_periods.return_value = 3
        mock_strategy.analyze.return_value = 1  # Buy signal

        mock_data_feed = Mock()
        mock_data_feed.get_historical.return_value = pd.DataFrame({"close": [100, 101, 102]})
        mock_data_feed.get_latest_price.return_value = 102.5

        mock_risk_manager = Mock()
        mock_risk_manager.check_trade.return_value = True

        mock_executor = Mock()
        mock_executor.get_account_status.return_value = Mock(total_equity=100000, cash=100000)

        runner = StrategyRunner(
            strategy=mock_strategy,
            data_feed=mock_data_feed,
            risk_manager=mock_risk_manager,
            executor=mock_executor,
            position_size=0.95,
        )

        runner.process_symbol("AAPL")

        # Should analyze data and execute signal
        mock_strategy.analyze.assert_called_once()
        mock_executor.execute_signal.assert_called_once()

    def test_process_symbol_zero_signal(self):
        """Test that runner does not execute on zero signal."""
        mock_strategy = Mock()
        mock_strategy.get_required_periods.return_value = 3
        mock_strategy.analyze.return_value = 0  # Hold signal

        mock_data_feed = Mock()
        mock_data_feed.get_historical.return_value = pd.DataFrame({"close": [100, 101, 102]})

        mock_executor = Mock()

        runner = StrategyRunner(
            strategy=mock_strategy,
            data_feed=mock_data_feed,
            risk_manager=Mock(),
            executor=mock_executor,
            position_size=0.95,
        )

        runner.process_symbol("AAPL")

        # Should not execute on zero signal
        mock_executor.execute_signal.assert_not_called()

    def test_process_symbol_no_current_price(self):
        """Test that runner does not execute if current price unavailable."""
        mock_strategy = Mock()
        mock_strategy.get_required_periods.return_value = 3
        mock_strategy.analyze.return_value = 1  # Buy signal

        mock_data_feed = Mock()
        mock_data_feed.get_historical.return_value = pd.DataFrame({"close": [100, 101, 102]})
        mock_data_feed.get_latest_price.return_value = None  # No price available

        mock_executor = Mock()

        runner = StrategyRunner(
            strategy=mock_strategy,
            data_feed=mock_data_feed,
            risk_manager=Mock(),
            executor=mock_executor,
            position_size=0.95,
        )

        runner.process_symbol("AAPL")

        # Should not execute if no current price
        mock_executor.execute_signal.assert_not_called()

    def test_process_symbol_risk_rejection(self):
        """Test that runner does not execute if risk check fails."""
        mock_strategy = Mock()
        mock_strategy.get_required_periods.return_value = 3
        mock_strategy.analyze.return_value = 1  # Buy signal

        mock_data_feed = Mock()
        mock_data_feed.get_historical.return_value = pd.DataFrame({"close": [100, 101, 102]})
        mock_data_feed.get_latest_price.return_value = 102.5

        mock_risk_manager = Mock()
        mock_risk_manager.check_trade.return_value = False  # Risk check fails

        mock_executor = Mock()
        mock_executor.get_account_status.return_value = Mock(total_equity=100000, cash=100000)

        runner = StrategyRunner(
            strategy=mock_strategy,
            data_feed=mock_data_feed,
            risk_manager=mock_risk_manager,
            executor=mock_executor,
            position_size=0.95,
        )

        runner.process_symbol("AAPL")

        # Should not execute if risk check fails
        mock_executor.execute_signal.assert_not_called()


# ============================================================================
# Test: Signal Execution
# ============================================================================


class TestStrategyRunnerExecution:
    """Test signal execution details."""

    def test_execute_signal_with_correct_params(self):
        """Test that signal is executed with correct parameters."""
        mock_strategy = Mock()
        mock_strategy.get_required_periods.return_value = 3
        mock_strategy.analyze.return_value = 1  # Buy signal

        mock_data_feed = Mock()
        mock_data_feed.get_historical.return_value = pd.DataFrame({"close": [100, 101, 102]})
        mock_data_feed.get_latest_price.return_value = 102.5

        mock_risk_manager = Mock()
        mock_risk_manager.check_trade.return_value = True

        mock_executor = Mock()
        mock_executor.get_account_status.return_value = Mock(total_equity=100000, cash=100000)

        runner = StrategyRunner(
            strategy=mock_strategy,
            data_feed=mock_data_feed,
            risk_manager=mock_risk_manager,
            executor=mock_executor,
            position_size=0.95,
        )

        runner.process_symbol("AAPL")

        # Verify execute_signal was called with correct parameters
        mock_executor.execute_signal.assert_called_once()
        call_kwargs = mock_executor.execute_signal.call_args[1]
        assert call_kwargs["symbol"] == "AAPL"
        assert call_kwargs["signal"] == 1
        assert call_kwargs["current_price"] == 102.5
        assert call_kwargs["position_size"] == 0.95
        assert "timestamp" in call_kwargs

    def test_execute_sell_signal(self):
        """Test that runner executes sell signals."""
        mock_strategy = Mock()
        mock_strategy.get_required_periods.return_value = 3
        mock_strategy.analyze.return_value = -1  # Sell signal

        mock_data_feed = Mock()
        mock_data_feed.get_historical.return_value = pd.DataFrame({"close": [100, 101, 102]})
        mock_data_feed.get_latest_price.return_value = 102.5

        mock_risk_manager = Mock()
        mock_risk_manager.check_trade.return_value = True

        mock_executor = Mock()
        mock_executor.get_account_status.return_value = Mock(total_equity=100000, cash=100000)

        runner = StrategyRunner(
            strategy=mock_strategy,
            data_feed=mock_data_feed,
            risk_manager=mock_risk_manager,
            executor=mock_executor,
            position_size=0.95,
        )

        runner.process_symbol("AAPL")

        # Should execute sell signal
        mock_executor.execute_signal.assert_called_once()
        call_kwargs = mock_executor.execute_signal.call_args[1]
        assert call_kwargs["signal"] == -1

    def test_risk_check_receives_correct_params(self):
        """Test that risk manager receives correct parameters."""
        mock_strategy = Mock()
        mock_strategy.get_required_periods.return_value = 3
        mock_strategy.analyze.return_value = 1  # Buy signal

        mock_data_feed = Mock()
        mock_data_feed.get_historical.return_value = pd.DataFrame({"close": [100, 101, 102]})
        mock_data_feed.get_latest_price.return_value = 102.5

        mock_risk_manager = Mock()
        mock_risk_manager.check_trade.return_value = True

        mock_account = Mock(total_equity=100000, cash=100000)
        mock_executor = Mock()
        mock_executor.get_account_status.return_value = mock_account

        runner = StrategyRunner(
            strategy=mock_strategy,
            data_feed=mock_data_feed,
            risk_manager=mock_risk_manager,
            executor=mock_executor,
            position_size=0.95,
        )

        runner.process_symbol("AAPL")

        # Verify risk check was called with correct parameters
        mock_risk_manager.check_trade.assert_called_once_with("AAPL", 1, 102.5, mock_account)


# ============================================================================
# Test: Error Handling
# ============================================================================


class TestStrategyRunnerErrorHandling:
    """Test error handling and recovery."""

    def test_exception_in_data_fetch(self):
        """Test that runner handles exception in data fetching."""
        mock_strategy = Mock()
        mock_strategy.get_required_periods.return_value = 50

        mock_data_feed = Mock()
        mock_data_feed.get_historical.side_effect = Exception("Data fetch error")

        mock_executor = Mock()

        runner = StrategyRunner(
            strategy=mock_strategy,
            data_feed=mock_data_feed,
            risk_manager=Mock(),
            executor=mock_executor,
            position_size=0.95,
        )

        # Should not raise exception
        runner.process_symbol("AAPL")

        # Should not execute
        mock_executor.execute_signal.assert_not_called()

    def test_exception_in_signal_generation(self):
        """Test that runner handles exception in signal generation."""
        mock_strategy = Mock()
        mock_strategy.get_required_periods.return_value = 3
        mock_strategy.analyze.side_effect = Exception("Signal generation error")

        mock_data_feed = Mock()
        mock_data_feed.get_historical.return_value = pd.DataFrame({"close": [100, 101, 102]})

        mock_executor = Mock()

        runner = StrategyRunner(
            strategy=mock_strategy,
            data_feed=mock_data_feed,
            risk_manager=Mock(),
            executor=mock_executor,
            position_size=0.95,
        )

        # Should not raise exception
        runner.process_symbol("AAPL")

        # Should not execute
        mock_executor.execute_signal.assert_not_called()

    def test_exception_in_execution(self):
        """Test that runner handles exception in signal execution."""
        mock_strategy = Mock()
        mock_strategy.get_required_periods.return_value = 3
        mock_strategy.analyze.return_value = 1  # Buy signal

        mock_data_feed = Mock()
        mock_data_feed.get_historical.return_value = pd.DataFrame({"close": [100, 101, 102]})
        mock_data_feed.get_latest_price.return_value = 102.5

        mock_risk_manager = Mock()
        mock_risk_manager.check_trade.return_value = True

        mock_executor = Mock()
        mock_executor.get_account_status.return_value = Mock(total_equity=100000, cash=100000)
        mock_executor.execute_signal.side_effect = Exception("Execution error")

        runner = StrategyRunner(
            strategy=mock_strategy,
            data_feed=mock_data_feed,
            risk_manager=mock_risk_manager,
            executor=mock_executor,
            position_size=0.95,
        )

        # Should not raise exception
        runner.process_symbol("AAPL")


# ============================================================================
# Test: Edge Cases
# ============================================================================


class TestStrategyRunnerEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_exact_required_periods(self):
        """Test with exactly the required number of periods."""
        mock_strategy = Mock()
        mock_strategy.get_required_periods.return_value = 3
        mock_strategy.analyze.return_value = 1  # Buy signal

        mock_data_feed = Mock()
        # Exactly 3 periods
        mock_data_feed.get_historical.return_value = pd.DataFrame({"close": [100, 101, 102]})
        mock_data_feed.get_latest_price.return_value = 102.5

        mock_risk_manager = Mock()
        mock_risk_manager.check_trade.return_value = True

        mock_executor = Mock()
        mock_executor.get_account_status.return_value = Mock(total_equity=100000, cash=100000)

        runner = StrategyRunner(
            strategy=mock_strategy,
            data_feed=mock_data_feed,
            risk_manager=mock_risk_manager,
            executor=mock_executor,
            position_size=0.95,
        )

        runner.process_symbol("AAPL")

        # Should execute with exactly required periods
        mock_executor.execute_signal.assert_called_once()

    def test_zero_position_size(self):
        """Test with zero position size."""
        mock_strategy = Mock()
        mock_strategy.get_required_periods.return_value = 3
        mock_strategy.analyze.return_value = 1  # Buy signal

        mock_data_feed = Mock()
        mock_data_feed.get_historical.return_value = pd.DataFrame({"close": [100, 101, 102]})
        mock_data_feed.get_latest_price.return_value = 102.5

        mock_risk_manager = Mock()
        mock_risk_manager.check_trade.return_value = True

        mock_executor = Mock()
        mock_executor.get_account_status.return_value = Mock(total_equity=100000, cash=100000)

        runner = StrategyRunner(
            strategy=mock_strategy,
            data_feed=mock_data_feed,
            risk_manager=mock_risk_manager,
            executor=mock_executor,
            position_size=0.0,  # Zero position size
        )

        runner.process_symbol("AAPL")

        # Should still execute (executor handles position sizing)
        mock_executor.execute_signal.assert_called_once()
        call_kwargs = mock_executor.execute_signal.call_args[1]
        assert call_kwargs["position_size"] == 0.0

    def test_multiple_symbols_processed_independently(self):
        """Test that multiple symbols are processed independently."""
        mock_strategy = Mock()
        mock_strategy.get_required_periods.return_value = 3
        mock_strategy.analyze.return_value = 1  # Buy signal

        mock_data_feed = Mock()
        mock_data_feed.get_historical.return_value = pd.DataFrame({"close": [100, 101, 102]})
        mock_data_feed.get_latest_price.return_value = 102.5

        mock_risk_manager = Mock()
        mock_risk_manager.check_trade.return_value = True

        mock_executor = Mock()
        mock_executor.get_account_status.return_value = Mock(total_equity=100000, cash=100000)

        runner = StrategyRunner(
            strategy=mock_strategy,
            data_feed=mock_data_feed,
            risk_manager=mock_risk_manager,
            executor=mock_executor,
            position_size=0.95,
        )

        # Process multiple symbols
        runner.process_symbol("AAPL")
        runner.process_symbol("MSFT")
        runner.process_symbol("GOOGL")

        # Should execute for each symbol
        assert mock_executor.execute_signal.call_count == 3
