"""
Tests for paper trading loop.

Tests cover:
- Thread lifecycle (start/stop/is_running)
- Loop iteration behavior
- Data feed updates
- Symbol processing callbacks
- Position updates
- Equity recording callbacks
- Sleep intervals
- Error handling
- Edge cases
"""

import time
from unittest.mock import Mock, call, patch

import pytest

from bot_v2.features.paper_trade.trading_loop import TradingLoop


# ============================================================================
# Test: Thread Lifecycle
# ============================================================================


class TestTradingLoopLifecycle:
    """Test thread lifecycle management."""

    @patch("bot_v2.features.paper_trade.trading_loop.threading.Thread")
    def test_start_creates_thread(self, mock_thread):
        """Test that start creates a background thread."""
        loop = TradingLoop(
            symbols=["AAPL"],
            update_interval=60,
            data_feed=Mock(),
            executor=Mock(),
            on_process_symbol=Mock(),
            on_record_equity=Mock(),
        )

        loop.start()

        mock_thread.assert_called_once()
        assert loop.is_running is True

    @patch("bot_v2.features.paper_trade.trading_loop.threading.Thread")
    def test_start_sets_daemon_flag(self, mock_thread):
        """Test that thread is created as daemon."""
        mock_thread_instance = Mock()
        mock_thread.return_value = mock_thread_instance

        loop = TradingLoop(
            symbols=["AAPL"],
            update_interval=60,
            data_feed=Mock(),
            executor=Mock(),
            on_process_symbol=Mock(),
            on_record_equity=Mock(),
        )

        loop.start()

        assert mock_thread_instance.daemon is True
        mock_thread_instance.start.assert_called_once()

    @patch("bot_v2.features.paper_trade.trading_loop.threading.Thread")
    def test_start_already_running_does_nothing(self, mock_thread):
        """Test that start does nothing if already running."""
        loop = TradingLoop(
            symbols=["AAPL"],
            update_interval=60,
            data_feed=Mock(),
            executor=Mock(),
            on_process_symbol=Mock(),
            on_record_equity=Mock(),
        )

        loop.start()
        # Try starting again
        loop.start()

        # Should only create thread once
        assert mock_thread.call_count == 1

    @patch("bot_v2.features.paper_trade.trading_loop.threading.Thread")
    def test_stop_sets_not_running(self, mock_thread):
        """Test that stop sets is_running to False."""
        mock_thread_instance = Mock()
        mock_thread.return_value = mock_thread_instance

        loop = TradingLoop(
            symbols=["AAPL"],
            update_interval=60,
            data_feed=Mock(),
            executor=Mock(),
            on_process_symbol=Mock(),
            on_record_equity=Mock(),
        )

        loop.start()
        loop.stop()

        assert loop.is_running is False

    @patch("bot_v2.features.paper_trade.trading_loop.threading.Thread")
    def test_stop_joins_thread(self, mock_thread):
        """Test that stop joins the thread."""
        mock_thread_instance = Mock()
        mock_thread.return_value = mock_thread_instance

        loop = TradingLoop(
            symbols=["AAPL"],
            update_interval=60,
            data_feed=Mock(),
            executor=Mock(),
            on_process_symbol=Mock(),
            on_record_equity=Mock(),
        )

        loop.start()
        loop.stop()

        mock_thread_instance.join.assert_called_once_with(timeout=5)

    @patch("bot_v2.features.paper_trade.trading_loop.threading.Thread")
    def test_stop_when_not_running_does_nothing(self, mock_thread):
        """Test that stop does nothing if not running."""
        loop = TradingLoop(
            symbols=["AAPL"],
            update_interval=60,
            data_feed=Mock(),
            executor=Mock(),
            on_process_symbol=Mock(),
            on_record_equity=Mock(),
        )

        # Stop without starting
        loop.stop()

        # Should not crash
        assert loop.is_running is False


# ============================================================================
# Test: Loop Iteration Behavior
# ============================================================================


class TestTradingLoopIteration:
    """Test loop iteration and component interactions."""

    @patch("bot_v2.features.paper_trade.trading_loop.time.sleep")
    def test_loop_updates_data_feed(self, mock_sleep):
        """Test that loop calls data_feed.update()."""
        mock_data_feed = Mock()
        mock_data_feed.get_latest_price.return_value = 100.0

        mock_executor = Mock()
        mock_executor.get_account_status.return_value = Mock(total_equity=100000)

        loop = TradingLoop(
            symbols=["AAPL"],
            update_interval=60,
            data_feed=mock_data_feed,
            executor=mock_executor,
            on_process_symbol=Mock(),
            on_record_equity=Mock(),
        )

        # Mock the while loop to run once
        def run_once_side_effect(*args, **kwargs):
            loop.is_running = False

        mock_sleep.side_effect = run_once_side_effect

        # Run loop (will exit after one iteration due to side_effect)
        loop.is_running = True
        loop._run()

        mock_data_feed.update.assert_called()

    @patch("bot_v2.features.paper_trade.trading_loop.time.sleep")
    def test_loop_processes_all_symbols(self, mock_sleep):
        """Test that loop calls on_process_symbol for each symbol."""
        mock_process = Mock()
        symbols = ["AAPL", "MSFT", "GOOGL"]

        mock_data_feed = Mock()
        mock_data_feed.get_latest_price.return_value = 100.0

        mock_executor = Mock()
        mock_executor.get_account_status.return_value = Mock(total_equity=100000)

        loop = TradingLoop(
            symbols=symbols,
            update_interval=60,
            data_feed=mock_data_feed,
            executor=mock_executor,
            on_process_symbol=mock_process,
            on_record_equity=Mock(),
        )

        # Mock sleep to stop loop after one iteration
        def stop_loop(*args, **kwargs):
            loop.is_running = False

        mock_sleep.side_effect = stop_loop

        # Run loop
        loop.is_running = True
        loop._run()

        # Should process each symbol once
        assert mock_process.call_count == 3
        mock_process.assert_any_call("AAPL")
        mock_process.assert_any_call("MSFT")
        mock_process.assert_any_call("GOOGL")

    @patch("bot_v2.features.paper_trade.trading_loop.time.sleep")
    def test_loop_updates_positions_with_price_map(self, mock_sleep):
        """Test that loop updates positions with correct price map."""
        mock_data_feed = Mock()
        # Return different prices for each symbol
        prices = {"AAPL": 150.0, "MSFT": 200.0}
        mock_data_feed.get_latest_price.side_effect = lambda s: prices.get(s)

        mock_executor = Mock()
        mock_executor.get_account_status.return_value = Mock(total_equity=100000)

        loop = TradingLoop(
            symbols=["AAPL", "MSFT"],
            update_interval=60,
            data_feed=mock_data_feed,
            executor=mock_executor,
            on_process_symbol=Mock(),
            on_record_equity=Mock(),
        )

        # Mock sleep to stop loop after one iteration
        def stop_loop(*args, **kwargs):
            loop.is_running = False

        mock_sleep.side_effect = stop_loop

        # Run loop
        loop.is_running = True
        loop._run()

        # Should update positions with price map
        mock_executor.update_positions.assert_called_once()
        call_args = mock_executor.update_positions.call_args[0][0]
        assert call_args == {"AAPL": 150.0, "MSFT": 200.0}

    @patch("bot_v2.features.paper_trade.trading_loop.time.sleep")
    def test_loop_records_equity(self, mock_sleep):
        """Test that loop records equity via callback."""
        mock_executor = Mock()
        mock_executor.get_account_status.return_value = Mock(total_equity=105000.0)

        mock_data_feed = Mock()
        mock_data_feed.get_latest_price.return_value = 100.0

        mock_record = Mock()

        loop = TradingLoop(
            symbols=["AAPL"],
            update_interval=60,
            data_feed=mock_data_feed,
            executor=mock_executor,
            on_process_symbol=Mock(),
            on_record_equity=mock_record,
        )

        # Mock sleep to stop loop after one iteration
        def stop_loop(*args, **kwargs):
            loop.is_running = False

        mock_sleep.side_effect = stop_loop

        # Run loop
        loop.is_running = True
        loop._run()

        mock_record.assert_called_once_with(105000.0)

    @patch("bot_v2.features.paper_trade.trading_loop.time.sleep")
    def test_loop_sleeps_correct_interval(self, mock_sleep):
        """Test that loop sleeps for correct interval."""
        mock_data_feed = Mock()
        mock_data_feed.get_latest_price.return_value = 100.0

        mock_executor = Mock()
        mock_executor.get_account_status.return_value = Mock(total_equity=100000)

        loop = TradingLoop(
            symbols=["AAPL"],
            update_interval=30,
            data_feed=mock_data_feed,
            executor=mock_executor,
            on_process_symbol=Mock(),
            on_record_equity=Mock(),
        )

        # Mock sleep to stop loop after one iteration
        def stop_loop(*args, **kwargs):
            loop.is_running = False

        mock_sleep.side_effect = stop_loop

        # Run loop
        loop.is_running = True
        loop._run()

        mock_sleep.assert_called_with(30.0)


# ============================================================================
# Test: Error Handling
# ============================================================================


class TestTradingLoopErrorHandling:
    """Test error handling and recovery."""

    @patch("bot_v2.features.paper_trade.trading_loop.time.sleep")
    @patch("bot_v2.features.paper_trade.trading_loop.logger")
    def test_loop_continues_on_exception(self, mock_logger, mock_sleep):
        """Test that loop continues after exception."""
        mock_data_feed = Mock()
        # First call raises exception, second succeeds
        call_count = [0]

        def update_side_effect():
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Test error")

        mock_data_feed.update.side_effect = update_side_effect
        mock_data_feed.get_latest_price.return_value = 100.0

        mock_executor = Mock()
        mock_executor.get_account_status.return_value = Mock(total_equity=100000)

        loop = TradingLoop(
            symbols=["AAPL"],
            update_interval=60,
            data_feed=mock_data_feed,
            executor=mock_executor,
            on_process_symbol=Mock(),
            on_record_equity=Mock(),
        )

        # Mock sleep to stop after two iterations
        iterations = [0]

        def stop_after_two(*args, **kwargs):
            iterations[0] += 1
            if iterations[0] >= 2:
                loop.is_running = False

        mock_sleep.side_effect = stop_after_two

        # Run loop
        loop.is_running = True
        loop._run()

        # Should have logged the error
        mock_logger.warning.assert_called()

    @patch("bot_v2.features.paper_trade.trading_loop.time.sleep")
    @patch("bot_v2.features.paper_trade.trading_loop.logger")
    def test_loop_logs_exceptions(self, mock_logger, mock_sleep):
        """Test that exceptions are logged."""
        mock_data_feed = Mock()
        mock_data_feed.update.side_effect = Exception("Data feed error")

        loop = TradingLoop(
            symbols=["AAPL"],
            update_interval=60,
            data_feed=mock_data_feed,
            executor=Mock(),
            on_process_symbol=Mock(),
            on_record_equity=Mock(),
        )

        # Mock sleep to stop after one iteration
        def stop_loop(*args, **kwargs):
            loop.is_running = False

        mock_sleep.side_effect = stop_loop

        # Run loop
        loop.is_running = True
        loop._run()

        # Should have logged the error
        mock_logger.warning.assert_called()

    @patch("bot_v2.features.paper_trade.trading_loop.time.sleep")
    def test_loop_sleeps_after_exception(self, mock_sleep):
        """Test that loop sleeps after catching exception."""
        mock_data_feed = Mock()
        mock_data_feed.update.side_effect = Exception("Test error")

        loop = TradingLoop(
            symbols=["AAPL"],
            update_interval=60,
            data_feed=mock_data_feed,
            executor=Mock(),
            on_process_symbol=Mock(),
            on_record_equity=Mock(),
        )

        # Mock sleep to stop after one iteration
        def stop_loop(*args, **kwargs):
            loop.is_running = False

        mock_sleep.side_effect = stop_loop

        # Run loop (will error and sleep)
        loop.is_running = True
        loop._run()

        # Should still sleep to avoid tight loop
        mock_sleep.assert_called_with(60.0)


# ============================================================================
# Test: Edge Cases
# ============================================================================


class TestTradingLoopEdgeCases:
    """Test edge cases and boundary conditions."""

    @patch("bot_v2.features.paper_trade.trading_loop.time.sleep")
    def test_loop_with_empty_symbols(self, mock_sleep):
        """Test loop with no symbols."""
        mock_data_feed = Mock()
        mock_data_feed.get_latest_price.return_value = None

        mock_executor = Mock()
        mock_executor.get_account_status.return_value = Mock(total_equity=100000)

        mock_process = Mock()

        loop = TradingLoop(
            symbols=[],
            update_interval=60,
            data_feed=mock_data_feed,
            executor=mock_executor,
            on_process_symbol=mock_process,
            on_record_equity=Mock(),
        )

        # Mock sleep to stop after one iteration
        def stop_loop(*args, **kwargs):
            loop.is_running = False

        mock_sleep.side_effect = stop_loop

        # Run loop
        loop.is_running = True
        loop._run()

        # Should not call process_symbol
        mock_process.assert_not_called()

    @patch("bot_v2.features.paper_trade.trading_loop.time.sleep")
    def test_loop_with_none_prices(self, mock_sleep):
        """Test loop handles None prices gracefully."""
        mock_data_feed = Mock()
        mock_data_feed.get_latest_price.return_value = None

        mock_executor = Mock()
        mock_executor.get_account_status.return_value = Mock(total_equity=100000)

        loop = TradingLoop(
            symbols=["AAPL", "MSFT"],
            update_interval=60,
            data_feed=mock_data_feed,
            executor=mock_executor,
            on_process_symbol=Mock(),
            on_record_equity=Mock(),
        )

        # Mock sleep to stop after one iteration
        def stop_loop(*args, **kwargs):
            loop.is_running = False

        mock_sleep.side_effect = stop_loop

        # Run loop
        loop.is_running = True
        loop._run()

        # Should pass empty price map (all prices were None)
        mock_executor.update_positions.assert_called_once_with({})

    @patch("bot_v2.features.paper_trade.trading_loop.time.sleep")
    def test_loop_with_zero_interval(self, mock_sleep):
        """Test loop with zero update interval."""
        mock_data_feed = Mock()
        mock_data_feed.get_latest_price.return_value = 100.0

        mock_executor = Mock()
        mock_executor.get_account_status.return_value = Mock(total_equity=100000)

        loop = TradingLoop(
            symbols=["AAPL"],
            update_interval=0,
            data_feed=mock_data_feed,
            executor=mock_executor,
            on_process_symbol=Mock(),
            on_record_equity=Mock(),
        )

        # Mock sleep to stop after one iteration
        def stop_loop(*args, **kwargs):
            loop.is_running = False

        mock_sleep.side_effect = stop_loop

        # Run loop
        loop.is_running = True
        loop._run()

        mock_sleep.assert_called_with(0.0)
