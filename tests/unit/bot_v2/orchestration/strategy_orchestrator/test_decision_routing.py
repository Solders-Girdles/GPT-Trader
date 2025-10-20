"""
Decision routing tests for StrategyOrchestrator.
"""

from decimal import Decimal
from unittest.mock import Mock

import pytest

from bot_v2.features.live_trade.strategies.perps_baseline import Action, Decision


class TestDecisionRouting:
    """Ensure decisions are routed to execution correctly."""

    @pytest.mark.asyncio
    async def test_buy_action_executes(
        self, async_orchestrator, fake_perps_bot, test_balance, test_position, test_product
    ):
        fake_perps_bot.broker.list_balances.return_value = [test_balance]
        fake_perps_bot.broker.list_positions.return_value = [test_position]
        fake_perps_bot.runtime_state.mark_windows["BTC-PERP"] = [Decimal("50000")] * 35
        fake_perps_bot.get_product.return_value = test_product

        cb_outcome = Mock()
        cb_outcome.triggered = False
        fake_perps_bot.risk_manager.check_volatility_circuit_breaker.return_value = cb_outcome
        fake_perps_bot.risk_manager.check_mark_staleness.return_value = False

        strategy = Mock()
        decision = Decision(action=Action.BUY, reason="test_buy")
        strategy.decide.return_value = decision
        fake_perps_bot.runtime_state.strategy = strategy

        await async_orchestrator.process_symbol("BTC-PERP")

        fake_perps_bot.execute_decision.assert_called_once()
        call_args = fake_perps_bot.execute_decision.call_args
        assert call_args[0][0] == "BTC-PERP"
        assert call_args[0][1] == decision
        assert call_args[0][2] == Decimal("50000")
        assert call_args[0][3] == test_product

    @pytest.mark.asyncio
    async def test_sell_action_executes(
        self, async_orchestrator, fake_perps_bot, test_balance, test_position, test_product
    ):
        fake_perps_bot.broker.list_balances.return_value = [test_balance]
        fake_perps_bot.broker.list_positions.return_value = [test_position]
        fake_perps_bot.runtime_state.mark_windows["BTC-PERP"] = [Decimal("50000")] * 35
        fake_perps_bot.get_product.return_value = test_product

        cb_outcome = Mock()
        cb_outcome.triggered = False
        fake_perps_bot.risk_manager.check_volatility_circuit_breaker.return_value = cb_outcome
        fake_perps_bot.risk_manager.check_mark_staleness.return_value = False

        strategy = Mock()
        decision = Decision(action=Action.SELL, reason="test_sell")
        strategy.decide.return_value = decision
        fake_perps_bot.runtime_state.strategy = strategy

        await async_orchestrator.process_symbol("BTC-PERP")

        fake_perps_bot.execute_decision.assert_called_once()
        call_args = fake_perps_bot.execute_decision.call_args
        assert call_args[0][0] == "BTC-PERP"
        assert call_args[0][1] == decision

    @pytest.mark.asyncio
    async def test_close_action_executes(
        self, async_orchestrator, fake_perps_bot, test_balance, test_position, test_product
    ):
        fake_perps_bot.broker.list_balances.return_value = [test_balance]
        fake_perps_bot.broker.list_positions.return_value = [test_position]
        fake_perps_bot.runtime_state.mark_windows["BTC-PERP"] = [Decimal("50000")] * 35
        fake_perps_bot.get_product.return_value = test_product

        cb_outcome = Mock()
        cb_outcome.triggered = False
        fake_perps_bot.risk_manager.check_volatility_circuit_breaker.return_value = cb_outcome
        fake_perps_bot.risk_manager.check_mark_staleness.return_value = False

        strategy = Mock()
        decision = Decision(action=Action.CLOSE, reason="test_close")
        strategy.decide.return_value = decision
        fake_perps_bot.runtime_state.strategy = strategy

        await async_orchestrator.process_symbol("BTC-PERP")

        fake_perps_bot.execute_decision.assert_called_once()
        call_args = fake_perps_bot.execute_decision.call_args
        assert call_args[0][0] == "BTC-PERP"
        assert call_args[0][1] == decision

    @pytest.mark.asyncio
    async def test_hold_action_skips_execution(
        self, async_orchestrator, fake_perps_bot, test_balance, test_position
    ):
        fake_perps_bot.broker.list_balances.return_value = [test_balance]
        fake_perps_bot.broker.list_positions.return_value = [test_position]
        fake_perps_bot.runtime_state.mark_windows["BTC-PERP"] = [Decimal("50000")] * 35

        cb_outcome = Mock()
        cb_outcome.triggered = False
        fake_perps_bot.risk_manager.check_volatility_circuit_breaker.return_value = cb_outcome
        fake_perps_bot.risk_manager.check_mark_staleness.return_value = False

        strategy = Mock()
        decision = Decision(action=Action.HOLD, reason="test_hold")
        strategy.decide.return_value = decision
        fake_perps_bot.runtime_state.strategy = strategy

        await async_orchestrator.process_symbol("BTC-PERP")

        fake_perps_bot.execute_decision.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_product_skips_execution(
        self, async_orchestrator, fake_perps_bot, test_balance, test_position
    ):
        fake_perps_bot.broker.list_balances.return_value = [test_balance]
        fake_perps_bot.broker.list_positions.return_value = [test_position]
        fake_perps_bot.runtime_state.mark_windows["BTC-PERP"] = [Decimal("50000")] * 35
        fake_perps_bot.get_product.side_effect = Exception("Product not found")

        cb_outcome = Mock()
        cb_outcome.triggered = False
        fake_perps_bot.risk_manager.check_volatility_circuit_breaker.return_value = cb_outcome
        fake_perps_bot.risk_manager.check_mark_staleness.return_value = False

        await async_orchestrator.process_symbol("BTC-PERP")

        fake_perps_bot.execute_decision.assert_not_called()
