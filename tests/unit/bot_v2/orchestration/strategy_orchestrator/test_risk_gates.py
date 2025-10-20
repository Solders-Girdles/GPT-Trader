"""
Tests for risk gates in StrategyOrchestrator.

Tests for volatility CB, kill-switch, mark staleness, and equity handling scenarios.
"""

from decimal import Decimal
from unittest.mock import Mock

import pytest

from bot_v2.features.live_trade.risk_runtime import CircuitBreakerAction


class TestRiskGates:
    """Test risk gate functionality."""

    @pytest.mark.asyncio
    async def test_process_symbol_kill_switch_engaged(
        self, async_orchestrator, fake_perps_bot, test_balance
    ):
        """Test process_symbol skips when kill switch is engaged."""
        fake_perps_bot.risk_manager.config.kill_switch_enabled = True
        fake_perps_bot.broker.list_balances.return_value = [test_balance]

        await async_orchestrator.process_symbol("BTC-PERP")

        # Should not execute decision
        fake_perps_bot.execute_decision.assert_not_called()
        fake_perps_bot.broker.list_balances.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_symbol_volatility_circuit_breaker_kill_switch(
        self, async_orchestrator, fake_perps_bot, test_balance, test_position
    ):
        """Test process_symbol skips when volatility CB triggers kill switch."""
        fake_perps_bot.broker.list_balances.return_value = [test_balance]
        fake_perps_bot.broker.list_positions.return_value = [test_position]
        fake_perps_bot.runtime_state.mark_windows["BTC-PERP"] = [Decimal("50000")] * 35

        # Mock CB to trigger kill switch
        cb_outcome = Mock()
        cb_outcome.triggered = True
        cb_outcome.action = CircuitBreakerAction.KILL_SWITCH
        fake_perps_bot.risk_manager.check_volatility_circuit_breaker.return_value = cb_outcome

        await async_orchestrator.process_symbol("BTC-PERP")

        fake_perps_bot.execute_decision.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_symbol_volatility_circuit_breaker_hold(
        self, async_orchestrator, fake_perps_bot, test_balance, test_position
    ):
        """Test process_symbol skips when volatility CB triggers reduce_only action."""
        fake_perps_bot.broker.list_balances.return_value = [test_balance]
        fake_perps_bot.broker.list_positions.return_value = [test_position]
        fake_perps_bot.runtime_state.mark_windows["BTC-PERP"] = [Decimal("50000")] * 35

        # Mock CB to trigger reduce_only
        cb_outcome = Mock()
        cb_outcome.triggered = True
        cb_outcome.action = CircuitBreakerAction.REDUCE_ONLY
        fake_perps_bot.risk_manager.check_volatility_circuit_breaker.return_value = cb_outcome
        fake_perps_bot.risk_manager.check_mark_staleness.return_value = False

        # Mock strategy to avoid NoneType error
        from bot_v2.features.live_trade.strategies.decisions import Action, Decision

        strategy = Mock()
        decision = Decision(action=Action.HOLD, reason="circuit_breaker")
        strategy.decide.return_value = decision
        fake_perps_bot.runtime_state.strategy = strategy

        await async_orchestrator.process_symbol("BTC-PERP")

        fake_perps_bot.execute_decision.assert_not_called()
        # Check that the decision was recorded as HOLD (not REDUCE_ONLY, which is the CB action)
        recorded_decision = fake_perps_bot.last_decisions["BTC-PERP"]
        assert recorded_decision.action == Action.HOLD
        assert "circuit_breaker" in recorded_decision.reason

    @pytest.mark.asyncio
    async def test_process_symbol_mark_staleness_failure(
        self, async_orchestrator, fake_perps_bot, test_balance, test_position
    ):
        """Test process_symbol skips when marks are stale."""
        fake_perps_bot.broker.list_balances.return_value = [test_balance]
        fake_perps_bot.broker.list_positions.return_value = [test_position]
        fake_perps_bot.runtime_state.mark_windows["BTC-PERP"] = [Decimal("50000")] * 35

        # Mock risk gates
        cb_outcome = Mock()
        cb_outcome.triggered = False
        fake_perps_bot.risk_manager.check_volatility_circuit_breaker.return_value = cb_outcome
        fake_perps_bot.risk_manager.check_mark_staleness.return_value = True  # Stale

        await async_orchestrator.process_symbol("BTC-PERP")

        fake_perps_bot.execute_decision.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_symbol_insufficient_marks(
        self, async_orchestrator, fake_perps_bot, test_balance, test_position
    ):
        """Test process_symbol skips when there are insufficient marks."""
        fake_perps_bot.broker.list_balances.return_value = [test_balance]
        fake_perps_bot.broker.list_positions.return_value = [test_position]
        # Set insufficient marks (less than required for strategy)
        fake_perps_bot.runtime_state.mark_windows["BTC-PERP"] = [Decimal("50000")] * 10

        # Mock risk gates to pass
        cb_outcome = Mock()
        cb_outcome.triggered = False
        fake_perps_bot.risk_manager.check_volatility_circuit_breaker.return_value = cb_outcome
        fake_perps_bot.risk_manager.check_mark_staleness.return_value = False

        await async_orchestrator.process_symbol("BTC-PERP")

        fake_perps_bot.execute_decision.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_symbol_missing_marks(
        self, async_orchestrator, fake_perps_bot, test_balance, test_position
    ):
        """Test process_symbol skips when marks are missing."""
        fake_perps_bot.broker.list_balances.return_value = [test_balance]
        fake_perps_bot.broker.list_positions.return_value = [test_position]
        # No marks in runtime state
        fake_perps_bot.runtime_state.mark_windows = {}

        # Mock risk gates to pass
        cb_outcome = Mock()
        cb_outcome.triggered = False
        fake_perps_bot.risk_manager.check_volatility_circuit_breaker.return_value = cb_outcome
        fake_perps_bot.risk_manager.check_mark_staleness.return_value = False

        await async_orchestrator.process_symbol("BTC-PERP")

        fake_perps_bot.execute_decision.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_symbol_equity_adjustment_zero_equity(
        self, async_orchestrator, fake_perps_bot, test_balance, test_position
    ):
        """Test process_symbol handles zero equity after adjustment."""
        # Setup balance that will result in zero equity after adjustment
        test_balance.total = Decimal("0")  # Zero equity
        fake_perps_bot.broker.list_balances.return_value = [test_balance]
        fake_perps_bot.broker.list_positions.return_value = [test_position]
        fake_perps_bot.runtime_state.mark_windows["BTC-PERP"] = [Decimal("50000")] * 35

        # Mock risk gates to pass
        cb_outcome = Mock()
        cb_outcome.triggered = False
        fake_perps_bot.risk_manager.check_volatility_circuit_breaker.return_value = cb_outcome
        fake_perps_bot.risk_manager.check_mark_staleness.return_value = False

        await async_orchestrator.process_symbol("BTC-PERP")

        fake_perps_bot.execute_decision.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_symbol_negative_equity(
        self, async_orchestrator, fake_perps_bot, test_balance, test_position
    ):
        """Test process_symbol handles negative equity."""
        # Setup balance with negative equity
        test_balance.total = Decimal("-100")  # Negative equity
        fake_perps_bot.broker.list_balances.return_value = [test_balance]
        fake_perps_bot.broker.list_positions.return_value = [test_position]
        fake_perps_bot.runtime_state.mark_windows["BTC-PERP"] = [Decimal("50000")] * 35

        # Mock risk gates to pass
        cb_outcome = Mock()
        cb_outcome.triggered = False
        fake_perps_bot.risk_manager.check_volatility_circuit_breaker.return_value = cb_outcome
        fake_perps_bot.risk_manager.check_mark_staleness.return_value = False

        await async_orchestrator.process_symbol("BTC-PERP")

        fake_perps_bot.execute_decision.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_symbol_risk_gates_all_pass(
        self, async_orchestrator, fake_perps_bot, test_balance, test_position, test_product
    ):
        """Test process_symbol when all risk gates pass."""
        fake_perps_bot.broker.list_balances.return_value = [test_balance]
        fake_perps_bot.broker.list_positions.return_value = [test_position]
        fake_perps_bot.runtime_state.mark_windows["BTC-PERP"] = [Decimal("50000")] * 35
        fake_perps_bot.get_product.return_value = test_product

        # Mock risk gates to pass
        cb_outcome = Mock()
        cb_outcome.triggered = False
        fake_perps_bot.risk_manager.check_volatility_circuit_breaker.return_value = cb_outcome
        fake_perps_bot.risk_manager.check_mark_staleness.return_value = False

        # Mock strategy to return decision
        from bot_v2.features.live_trade.strategies.perps_baseline import Action, Decision

        strategy = Mock()
        decision = Decision(action=Action.BUY, reason="test")
        strategy.decide.return_value = decision
        fake_perps_bot.runtime_state.strategy = strategy

        await async_orchestrator.process_symbol("BTC-PERP")

        # Should execute decision
        fake_perps_bot.execute_decision.assert_called_once()
