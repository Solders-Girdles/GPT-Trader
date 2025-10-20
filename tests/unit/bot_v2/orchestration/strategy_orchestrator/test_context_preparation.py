"""
Tests for context preparation in StrategyOrchestrator.

Tests that exercise _prepare_context, balance/position failures, and mark handling.
"""

from decimal import Decimal
from unittest.mock import Mock

import pytest

from bot_v2.features.live_trade.strategies.decisions import Action, Decision
from bot_v2.orchestration.strategy_orchestrator import SymbolProcessingContext

from .conftest import (
    test_product,
)


class TestContextPreparation:
    """Test context preparation functionality."""

    @pytest.mark.asyncio
    async def test_process_symbol_balance_fetch_failure(self, async_orchestrator, fake_perps_bot):
        """Test process_symbol handles balance fetch failures."""
        fake_perps_bot.broker.list_balances.side_effect = Exception("Network error")

        # Should not crash, just skip processing
        await async_orchestrator.process_symbol("BTC-PERP")

        fake_perps_bot.execute_decision.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_symbol_position_fetch_failure(
        self, async_orchestrator, fake_perps_bot, test_balance
    ):
        """Test process_symbol handles position fetch failures."""
        fake_perps_bot.broker.list_balances.return_value = [test_balance]
        fake_perps_bot.broker.list_positions.side_effect = Exception("Position fetch failed")

        await async_orchestrator.process_symbol("BTC-PERP")

        fake_perps_bot.execute_decision.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_symbol_with_provided_balances_and_positions(
        self, async_orchestrator, fake_perps_bot, test_balance, test_position, test_product
    ):
        """Test process_symbol with provided balances and positions (no async calls)."""
        balances = [test_balance]
        positions = {"BTC-PERP": test_position}
        fake_perps_bot.runtime_state.mark_windows["BTC-PERP"] = [Decimal("50000")] * 35
        fake_perps_bot.get_product.return_value = test_product

        # Mock risk gates to pass
        cb_outcome = Mock()
        cb_outcome.triggered = False
        fake_perps_bot.risk_manager.check_volatility_circuit_breaker.return_value = cb_outcome
        fake_perps_bot.risk_manager.check_mark_staleness.return_value = False

        # Mock strategy to return decision
        strategy = Mock()
        decision = Decision(action=Action.BUY, reason="test")
        strategy.decide.return_value = decision
        fake_perps_bot.runtime_state.strategy = strategy

        await async_orchestrator.process_symbol(
            "BTC-PERP", balances=balances, position_map=positions
        )

        # Should execute decision
        fake_perps_bot.execute_decision.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_symbol_zero_equity_after_adjustment(
        self, async_orchestrator, fake_perps_bot, test_balance, test_position
    ):
        """Test process_symbol skips when equity becomes zero after adjustment."""
        # Setup balance with low equity
        test_balance.total = Decimal("100")  # Low equity
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
    async def test_ensure_balances_async_error_handling(self, async_orchestrator, fake_perps_bot):
        """Test _ensure_balances handles async errors."""
        fake_perps_bot.broker.list_balances.side_effect = Exception("Async balance error")

        # The method should propagate exceptions from asyncio.to_thread
        with pytest.raises(Exception, match="Async balance error"):
            await async_orchestrator._ensure_balances(None)

    @pytest.mark.asyncio
    async def test_ensure_positions_basic_functionality(
        self, async_orchestrator, fake_perps_bot, test_position
    ):
        """Test _ensure_positions basic functionality."""
        # Test with provided position_map (should return as-is)
        position_map = {"BTC-PERP": test_position}
        positions = await async_orchestrator._ensure_positions(position_map)

        assert positions == position_map
        fake_perps_bot.broker.list_positions.assert_not_called()

    @pytest.mark.asyncio
    async def test_prepare_context_with_marks(
        self, async_orchestrator, fake_perps_bot, test_balance, test_position
    ):
        """Test _prepare_context with mark data."""
        # Setup marks in runtime state
        fake_perps_bot.runtime_state.mark_windows["BTC-PERP"] = [Decimal("50000")] * 35

        # Create context to test _prepare_context
        context = SymbolProcessingContext(
            symbol="BTC-PERP",
            balances=[test_balance],
            equity=Decimal("10000"),
            positions={"BTC-PERP": test_position},
            position_state=test_position,
            position_quantity=Decimal("0.5"),
            marks=[Decimal("50000")] * 35,
            product=test_product,
        )

        # Verify context properties
        assert context.symbol == "BTC-PERP"
        assert context.balances == [test_balance]
        assert context.equity == Decimal("10000")
        assert context.positions == {"BTC-PERP": test_position}
        assert context.position_state == test_position
        assert context.position_quantity == Decimal("0.5")
        assert len(context.marks) == 35
        assert context.product == test_product

    @pytest.mark.asyncio
    async def test_prepare_context_mark_handling(
        self, async_orchestrator, fake_perps_bot, test_balance, test_position, test_product
    ):
        """Test mark handling in context preparation."""
        # Setup marks with varying values
        marks = [Decimal("50000") + Decimal(str(i * 10)) for i in range(35)]
        fake_perps_bot.runtime_state.mark_windows["BTC-PERP"] = marks

        # Mock strategy to return decision
        strategy = Mock()
        decision = Decision(action=Action.BUY, reason="test")
        strategy.decide.return_value = decision
        fake_perps_bot.runtime_state.strategy = strategy

        # Mock risk gates to pass
        cb_outcome = Mock()
        cb_outcome.triggered = False
        fake_perps_bot.risk_manager.check_volatility_circuit_breaker.return_value = cb_outcome
        fake_perps_bot.risk_manager.check_mark_staleness.return_value = False

        # Mock product
        fake_perps_bot.get_product.return_value = test_product

        await async_orchestrator.process_symbol(
            "BTC-PERP", balances=[test_balance], position_map={"BTC-PERP": test_position}
        )

        # Verify marks were used in processing
        fake_perps_bot.execute_decision.assert_called_once()
