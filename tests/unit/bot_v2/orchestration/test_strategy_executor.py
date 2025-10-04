"""Tests for StrategyExecutor - strategy execution and decision recording.

This module tests the StrategyExecutor's ability to:
- Execute strategies with proper parameters
- Measure and log performance telemetry
- Record decisions to bot
- Handle exceptions gracefully
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from bot_v2.orchestration.strategy_executor import StrategyExecutor


@pytest.fixture
def mock_bot():
    """Create mock bot."""
    bot = Mock()
    bot.last_decisions = {}
    bot.get_product = Mock(return_value=Mock())
    return bot


@pytest.fixture
def executor(mock_bot):
    """Create StrategyExecutor instance."""
    return StrategyExecutor(mock_bot)


@pytest.fixture
def mock_strategy():
    """Create mock strategy."""
    strategy = Mock()
    strategy.decide = Mock(return_value=Mock(action=Mock(value="BUY"), reason="test_reason"))
    return strategy


@pytest.fixture
def sample_marks():
    """Create sample mark prices."""
    return [Decimal("50000"), Decimal("50100"), Decimal("50200")]


class TestEvaluateStrategy:
    """Test strategy evaluation."""

    def test_calls_strategy_decide_with_correct_parameters(
        self, executor, mock_bot, mock_strategy, sample_marks
    ):
        """Calls strategy.decide with correct parameters."""
        position_state = {"quantity": Decimal("1.0"), "side": "long"}
        equity = Decimal("10000")
        product = Mock()
        mock_bot.get_product.return_value = product

        executor.evaluate_and_record(mock_strategy, "BTC-USD", sample_marks, position_state, equity)

        mock_strategy.decide.assert_called_once_with(
            symbol="BTC-USD",
            current_mark=Decimal("50200"),  # Last mark
            position_state=position_state,
            recent_marks=[Decimal("50000"), Decimal("50100")],  # All except last
            equity=equity,
            product=product,
        )

    def test_passes_empty_recent_marks_for_single_mark(self, executor, mock_bot, mock_strategy):
        """Passes empty list for recent_marks when only one mark."""
        marks = [Decimal("50000")]

        executor.evaluate_and_record(mock_strategy, "BTC-USD", marks, None, Decimal("10000"))

        call_args = mock_strategy.decide.call_args
        assert call_args.kwargs["recent_marks"] == []
        assert call_args.kwargs["current_mark"] == Decimal("50000")

    def test_gets_product_from_bot(self, executor, mock_bot, mock_strategy, sample_marks):
        """Gets product from bot for the symbol."""
        product = Mock()
        mock_bot.get_product.return_value = product

        executor.evaluate_and_record(mock_strategy, "ETH-USD", sample_marks, None, Decimal("10000"))

        mock_bot.get_product.assert_called_once_with("ETH-USD")
        call_args = mock_strategy.decide.call_args
        assert call_args.kwargs["product"] == product

    def test_returns_decision_from_strategy(self, executor, mock_bot, mock_strategy, sample_marks):
        """Returns the decision from strategy.decide."""
        expected_decision = Mock(action=Mock(value="SELL"), reason="exit")
        mock_strategy.decide.return_value = expected_decision

        decision = executor.evaluate_and_record(
            mock_strategy, "BTC-USD", sample_marks, None, Decimal("10000")
        )

        assert decision == expected_decision

    def test_passes_none_position_state(self, executor, mock_bot, mock_strategy, sample_marks):
        """Passes None position_state when no position."""
        executor.evaluate_and_record(mock_strategy, "BTC-USD", sample_marks, None, Decimal("10000"))

        call_args = mock_strategy.decide.call_args
        assert call_args.kwargs["position_state"] is None


class TestTelemetry:
    """Test performance telemetry."""

    @patch("bot_v2.orchestration.strategy_executor._get_plog")
    def test_logs_strategy_duration(
        self, mock_get_plog, executor, mock_bot, mock_strategy, sample_marks
    ):
        """Logs strategy execution duration."""
        mock_plog = Mock()
        mock_get_plog.return_value = mock_plog

        executor.evaluate_and_record(mock_strategy, "BTC-USD", sample_marks, None, Decimal("10000"))

        mock_plog.log_strategy_duration.assert_called_once()
        call_args = mock_plog.log_strategy_duration.call_args
        assert call_args.kwargs["strategy"] == type(mock_strategy).__name__
        assert "duration_ms" in call_args.kwargs
        assert call_args.kwargs["duration_ms"] >= 0

    @patch("bot_v2.orchestration.strategy_executor._get_plog")
    def test_handles_telemetry_logging_exception(
        self, mock_get_plog, executor, mock_bot, mock_strategy, sample_marks, caplog
    ):
        """Handles telemetry logging exceptions gracefully."""
        mock_get_plog.side_effect = Exception("Telemetry error")

        with caplog.at_level("DEBUG"):
            # Should not raise, just log
            decision = executor.evaluate_and_record(
                mock_strategy, "BTC-USD", sample_marks, None, Decimal("10000")
            )

        assert decision is not None
        assert "Failed to log strategy duration" in caplog.text

    @patch("bot_v2.orchestration.strategy_executor._time.perf_counter")
    @patch("bot_v2.orchestration.strategy_executor._get_plog")
    def test_measures_execution_time_correctly(
        self, mock_get_plog, mock_perf_counter, executor, mock_bot, mock_strategy, sample_marks
    ):
        """Measures strategy execution time correctly."""
        mock_plog = Mock()
        mock_get_plog.return_value = mock_plog
        # Simulate 50ms execution time
        mock_perf_counter.side_effect = [0.0, 0.05]

        executor.evaluate_and_record(mock_strategy, "BTC-USD", sample_marks, None, Decimal("10000"))

        call_args = mock_plog.log_strategy_duration.call_args
        assert call_args.kwargs["duration_ms"] == 50.0


class TestRecordDecision:
    """Test decision recording."""

    def test_records_decision_to_bot_last_decisions(
        self, executor, mock_bot, mock_strategy, sample_marks
    ):
        """Records decision to bot.last_decisions."""
        decision = Mock(action=Mock(value="BUY"), reason="entry")
        mock_strategy.decide.return_value = decision

        executor.evaluate_and_record(mock_strategy, "BTC-USD", sample_marks, None, Decimal("10000"))

        assert mock_bot.last_decisions["BTC-USD"] == decision

    def test_logs_decision_with_symbol_action_reason(
        self, executor, mock_bot, mock_strategy, sample_marks, caplog
    ):
        """Logs decision with symbol, action, and reason."""
        decision = Mock(action=Mock(value="SELL"), reason="stop_loss")
        mock_strategy.decide.return_value = decision

        with caplog.at_level("INFO"):
            executor.evaluate_and_record(
                mock_strategy, "ETH-USD", sample_marks, None, Decimal("10000")
            )

        assert "ETH-USD Decision: SELL - stop_loss" in caplog.text

    def test_overwrites_previous_decision_for_same_symbol(
        self, executor, mock_bot, mock_strategy, sample_marks
    ):
        """Overwrites previous decision for the same symbol."""
        old_decision = Mock(action=Mock(value="HOLD"), reason="old")
        new_decision = Mock(action=Mock(value="BUY"), reason="new")
        mock_bot.last_decisions["BTC-USD"] = old_decision
        mock_strategy.decide.return_value = new_decision

        executor.evaluate_and_record(mock_strategy, "BTC-USD", sample_marks, None, Decimal("10000"))

        assert mock_bot.last_decisions["BTC-USD"] == new_decision
        assert mock_bot.last_decisions["BTC-USD"] != old_decision


class TestIntegration:
    """Test integrated behavior."""

    @patch("bot_v2.orchestration.strategy_executor._get_plog")
    def test_full_evaluation_and_recording_flow(
        self, mock_get_plog, executor, mock_bot, mock_strategy, sample_marks, caplog
    ):
        """Complete flow: evaluate, measure, log telemetry, record, log decision."""
        mock_plog = Mock()
        mock_get_plog.return_value = mock_plog
        decision = Mock(action=Mock(value="CLOSE"), reason="exit_signal")
        mock_strategy.decide.return_value = decision
        position_state = {"quantity": Decimal("0.5"), "side": "long"}

        with caplog.at_level("INFO"):
            result = executor.evaluate_and_record(
                mock_strategy, "BTC-USD", sample_marks, position_state, Decimal("20000")
            )

        # Verify strategy was called
        mock_strategy.decide.assert_called_once()

        # Verify telemetry was logged
        mock_plog.log_strategy_duration.assert_called_once()

        # Verify decision was recorded
        assert mock_bot.last_decisions["BTC-USD"] == decision

        # Verify decision was logged
        assert "BTC-USD Decision: CLOSE - exit_signal" in caplog.text

        # Verify return value
        assert result == decision

    def test_multiple_symbols_independent_decisions(
        self, executor, mock_bot, mock_strategy, sample_marks
    ):
        """Records independent decisions for different symbols."""
        btc_decision = Mock(action=Mock(value="BUY"), reason="btc_entry")
        eth_decision = Mock(action=Mock(value="SELL"), reason="eth_exit")

        mock_strategy.decide.return_value = btc_decision
        executor.evaluate_and_record(mock_strategy, "BTC-USD", sample_marks, None, Decimal("10000"))

        mock_strategy.decide.return_value = eth_decision
        executor.evaluate_and_record(mock_strategy, "ETH-USD", sample_marks, None, Decimal("10000"))

        assert mock_bot.last_decisions["BTC-USD"] == btc_decision
        assert mock_bot.last_decisions["ETH-USD"] == eth_decision
