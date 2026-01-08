"""Integration tests for validation failure escalation flow.

Tests the end-to-end safety loop:
1. Validation checks fail repeatedly
2. Failure tracker counts consecutive failures
3. At threshold, escalation callback fires
4. Risk manager activates reduce-only mode

This validates the critical safety path that prevents trading
when validation infrastructure is broken.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.app.config import BotConfig
from gpt_trader.core import OrderSide
from gpt_trader.features.live_trade.execution.engine import LiveExecutionEngine
from gpt_trader.features.live_trade.risk import LiveRiskManager


class TestValidationEscalationFlow:
    """Integration tests for validation failure escalation."""

    @pytest.fixture
    def mock_broker(self) -> MagicMock:
        """Create a mock broker."""
        broker = MagicMock()
        broker.list_balances.return_value = []
        broker.get_product.return_value = MagicMock()
        return broker

    @pytest.fixture
    def real_risk_manager(self) -> LiveRiskManager:
        """Create a real risk manager for integration testing."""
        manager = LiveRiskManager()
        manager.set_reduce_only_mode(False, reason="test_setup")
        return manager

    @pytest.fixture
    def bot_config(self) -> BotConfig:
        """Create a basic bot config."""
        return BotConfig(symbols=["BTC-USD"])

    @pytest.fixture
    def execution_engine(
        self,
        mock_broker: MagicMock,
        real_risk_manager: LiveRiskManager,
        bot_config: BotConfig,
    ) -> LiveExecutionEngine:
        """Create execution engine with real risk manager."""
        return LiveExecutionEngine(
            broker=mock_broker,
            config=bot_config,
            risk_manager=real_risk_manager,
            enable_preview=False,
        )

    def test_escalation_flow_triggers_reduce_only_mode(
        self,
        execution_engine: LiveExecutionEngine,
        real_risk_manager: LiveRiskManager,
    ) -> None:
        """Test that 5 consecutive validation failures trigger reduce-only mode.

        This is the critical safety loop:
        - Validation infrastructure fails (API errors, timeouts)
        - Failure tracker counts consecutive failures
        - At threshold (5), escalation callback fires
        - Risk manager activates reduce-only mode
        """
        # Verify initial state - not in reduce-only mode
        assert not real_risk_manager.is_reduce_only_mode()

        # Make validation check throw exceptions (simulating API failures)
        with patch.object(
            real_risk_manager,
            "check_mark_staleness",
            side_effect=RuntimeError("Simulated API failure"),
        ):
            # First 4 failures - should NOT trigger reduce-only
            for i in range(4):
                execution_engine.order_validator.ensure_mark_is_fresh("BTC-USD")
                assert (
                    not real_risk_manager.is_reduce_only_mode()
                ), f"Reduce-only triggered too early at failure {i + 1}"

            # 5th failure - SHOULD trigger reduce-only
            execution_engine.order_validator.ensure_mark_is_fresh("BTC-USD")

        # Verify reduce-only mode was activated
        assert real_risk_manager.is_reduce_only_mode()

    def test_successful_validation_resets_failure_counter(
        self,
        execution_engine: LiveExecutionEngine,
        real_risk_manager: LiveRiskManager,
    ) -> None:
        """Test that successful validations reset the failure counter.

        A success after some failures should reset the counter,
        preventing escalation from intermittent failures.
        """
        # Verify initial state
        assert not real_risk_manager.is_reduce_only_mode()

        # Simulate 3 failures
        with patch.object(
            real_risk_manager,
            "check_mark_staleness",
            side_effect=RuntimeError("API failure"),
        ):
            for _ in range(3):
                execution_engine.order_validator.ensure_mark_is_fresh("BTC-USD")

        # Then a success (mark is not stale)
        with patch.object(
            real_risk_manager,
            "check_mark_staleness",
            return_value=False,  # Mark is fresh
        ):
            execution_engine.order_validator.ensure_mark_is_fresh("BTC-USD")

        # Then 4 more failures - should NOT escalate (counter was reset)
        with patch.object(
            real_risk_manager,
            "check_mark_staleness",
            side_effect=RuntimeError("API failure"),
        ):
            for _ in range(4):
                execution_engine.order_validator.ensure_mark_is_fresh("BTC-USD")

        # Should NOT be in reduce-only (only 4 consecutive after reset)
        assert not real_risk_manager.is_reduce_only_mode()

    def test_slippage_guard_failures_also_trigger_escalation(
        self,
        execution_engine: LiveExecutionEngine,
        real_risk_manager: LiveRiskManager,
        mock_broker: MagicMock,
    ) -> None:
        """Test that slippage guard failures also trigger escalation.

        The escalation mechanism applies to all validation checks,
        not just mark staleness.
        """
        # Verify initial state
        assert not real_risk_manager.is_reduce_only_mode()

        # Make broker's get_market_snapshot throw exceptions
        mock_broker.get_market_snapshot.side_effect = RuntimeError("API failure")

        # 5 slippage guard failures should trigger escalation
        for _ in range(5):
            execution_engine.order_validator.enforce_slippage_guard(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_quantity=Decimal("1.0"),
                effective_price=Decimal("50000"),
            )

        # Verify reduce-only mode was activated
        assert real_risk_manager.is_reduce_only_mode()

    def test_different_check_types_have_independent_counters(
        self,
        execution_engine: LiveExecutionEngine,
        real_risk_manager: LiveRiskManager,
        mock_broker: MagicMock,
    ) -> None:
        """Test that different check types track failures independently.

        Mark staleness and slippage guard each have their own counter,
        so 3 of each won't trigger escalation (needs 5 of same type).
        """
        # Verify initial state
        assert not real_risk_manager.is_reduce_only_mode()

        # 3 mark staleness failures
        with patch.object(
            real_risk_manager,
            "check_mark_staleness",
            side_effect=RuntimeError("API failure"),
        ):
            for _ in range(3):
                execution_engine.order_validator.ensure_mark_is_fresh("BTC-USD")

        # 3 slippage guard failures
        mock_broker.get_market_snapshot.side_effect = RuntimeError("API failure")
        for _ in range(3):
            execution_engine.order_validator.enforce_slippage_guard(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_quantity=Decimal("1.0"),
                effective_price=Decimal("50000"),
            )

        # Total 6 failures, but neither type reached 5
        assert not real_risk_manager.is_reduce_only_mode()

    def test_escalation_reason_is_correct(
        self,
        mock_broker: MagicMock,
        bot_config: BotConfig,
    ) -> None:
        """Test that escalation sets the correct reason on risk manager."""
        # Create fresh manager to check the reason
        risk_manager = LiveRiskManager()
        risk_manager.set_reduce_only_mode(False, reason="test_setup")

        engine = LiveExecutionEngine(
            broker=mock_broker,
            config=bot_config,
            risk_manager=risk_manager,
            enable_preview=False,
        )

        # Trigger escalation
        with patch.object(
            risk_manager,
            "check_mark_staleness",
            side_effect=RuntimeError("API failure"),
        ):
            for _ in range(5):
                engine.order_validator.ensure_mark_is_fresh("BTC-USD")

        # Check the reason
        assert risk_manager.is_reduce_only_mode()
        # Access the internal reason (may vary by implementation)
        reason = getattr(risk_manager, "_reduce_only_reason", None)
        if reason is not None:
            assert "validation" in reason.lower() or "consecutive" in reason.lower()


class TestValidationEscalationWithMetrics:
    """Integration tests verifying metrics are recorded during escalation."""

    @pytest.fixture
    def mock_broker(self) -> MagicMock:
        """Create a mock broker."""
        broker = MagicMock()
        broker.list_balances.return_value = []
        broker.get_product.return_value = MagicMock()
        return broker

    @pytest.fixture
    def bot_config(self) -> BotConfig:
        """Create a basic bot config."""
        return BotConfig(symbols=["BTC-USD"])

    @patch("gpt_trader.features.live_trade.execution.validation.record_counter")
    def test_metrics_recorded_during_escalation(
        self,
        mock_record_counter: MagicMock,
        mock_broker: MagicMock,
        bot_config: BotConfig,
    ) -> None:
        """Test that metrics are recorded during the escalation flow."""
        from gpt_trader.features.live_trade.execution.validation import (
            METRIC_CONSECUTIVE_FAILURES_ESCALATION,
            METRIC_MARK_STALENESS_CHECK_FAILED,
        )

        risk_manager = LiveRiskManager()

        engine = LiveExecutionEngine(
            broker=mock_broker,
            config=bot_config,
            risk_manager=risk_manager,
            enable_preview=False,
        )

        # Trigger escalation
        with patch.object(
            risk_manager,
            "check_mark_staleness",
            side_effect=RuntimeError("API failure"),
        ):
            for _ in range(5):
                engine.order_validator.ensure_mark_is_fresh("BTC-USD")

        # Verify failure metrics were recorded
        failure_calls = [
            call
            for call in mock_record_counter.call_args_list
            if call[0][0] == METRIC_MARK_STALENESS_CHECK_FAILED
        ]
        assert len(failure_calls) == 5

        # Verify escalation metric was recorded
        escalation_calls = [
            call
            for call in mock_record_counter.call_args_list
            if call[0][0] == METRIC_CONSECUTIVE_FAILURES_ESCALATION
        ]
        assert len(escalation_calls) == 1
