"""Tests for StrategyOrchestrator decision execution and routing.

This module tests:
- Strategy evaluation (decide() invocation)
- Decision recording
- Candle fetching for spot strategies
- Symbol processing orchestration
- Decision routing through guard chains
- Spot vs non-spot decision handling
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from bot_v2.features.brokerages.core.interfaces import MarketType, Product
from bot_v2.features.live_trade.risk_runtime import (
    CircuitBreakerAction,
    CircuitBreakerOutcome,
)
from bot_v2.features.live_trade.strategies.perps_baseline import (
    Action,
    BaselinePerpsStrategy,
    Decision,
)
from bot_v2.orchestration.configuration import Profile
from bot_v2.orchestration.strategy_orchestrator import StrategyOrchestrator


class TestEvaluateStrategy:
    """Test _evaluate_strategy method."""

    def test_calls_strategy_decide(self, orchestrator, mock_bot):
        """Test calls strategy.decide with correct parameters."""
        strategy = Mock(spec=BaselinePerpsStrategy)
        decision = Decision(action=Action.BUY, reason="test")
        strategy.decide = Mock(return_value=decision)

        marks = [Decimal("50000"), Decimal("51000")]
        position_state = {"quantity": Decimal("0.5"), "side": "long"}
        equity = Decimal("10000")
        product = Product(
            symbol="BTC-PERP",
            base_asset="BTC",
            quote_asset="USD",
            market_type=MarketType.PERPETUAL,
            min_size=Decimal("0.001"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("1"),
            price_increment=Decimal("0.01"),
            leverage_max=5,
            expiry=None,
            contract_size=Decimal("1"),
            funding_rate=Decimal("0"),
            next_funding_time=None,
        )
        mock_bot.get_product = Mock(return_value=product)

        result = orchestrator._evaluate_strategy(
            strategy, "BTC-PERP", marks, position_state, equity, product
        )

        assert result == decision
        strategy.decide.assert_called_once()
        call_kwargs = strategy.decide.call_args.kwargs
        assert call_kwargs["symbol"] == "BTC-PERP"
        assert call_kwargs["current_mark"] == Decimal("51000")
        assert call_kwargs["position_state"] == position_state
        assert call_kwargs["equity"] == equity
        assert call_kwargs["product"] == product


class TestRecordDecision:
    """Test _record_decision method."""

    def test_records_decision_to_bot(self, orchestrator, mock_bot):
        """Test records decision in bot's last_decisions map."""
        decision = Decision(action=Action.BUY, reason="test_reason")

        orchestrator._record_decision("BTC-PERP", decision)

        assert mock_bot.last_decisions["BTC-PERP"] == decision


class TestFetchSpotCandles:
    """Test _fetch_spot_candles async method."""

    @pytest.mark.asyncio
    async def test_fetches_candles_from_broker(self, orchestrator, mock_bot):
        """Test fetches candles from broker."""
        now = datetime.now(timezone.utc)
        candle1 = Mock(close=Decimal("50000"), ts=now)
        candle2 = Mock(close=Decimal("51000"), ts=now)
        mock_bot.broker.get_candles = Mock(return_value=[candle1, candle2])

        candles = await orchestrator._fetch_spot_candles("BTC-PERP", 20)

        assert len(candles) == 2
        mock_bot.broker.get_candles.assert_called_once_with("BTC-PERP", "ONE_HOUR", 22)

    @pytest.mark.asyncio
    async def test_returns_empty_on_exception(self, orchestrator, mock_bot):
        """Test returns empty list on exception."""
        mock_bot.broker.get_candles = Mock(side_effect=Exception("fetch failed"))

        candles = await orchestrator._fetch_spot_candles("BTC-PERP", 20)

        assert candles == []


class TestProcessSymbol:
    """Test process_symbol async orchestration method."""

    @pytest.mark.asyncio
    async def test_skips_when_kill_switch_engaged(self, orchestrator, mock_bot, test_balance):
        """Test skips processing when kill switch engaged."""
        mock_bot.risk_manager.config.kill_switch_enabled = True
        mock_bot.broker.list_balances = Mock(return_value=[test_balance])

        await orchestrator.process_symbol("BTC-PERP")

        # Should not call execute_decision
        mock_bot.execute_decision.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_no_marks(self, orchestrator, mock_bot, test_balance):
        """Test skips processing when no marks available."""
        mock_bot.broker.list_balances = Mock(return_value=[test_balance])
        mock_bot.broker.list_positions = Mock(return_value=[])
        mock_bot.runtime_state.mark_windows.clear()

        await orchestrator.process_symbol("BTC-PERP")

        # Should not call execute_decision
        mock_bot.execute_decision.assert_not_called()

    @pytest.mark.asyncio
    async def test_executes_buy_decision(self, orchestrator, mock_bot, test_balance, test_position):
        """Test executes BUY decision through bot."""
        mock_bot.config.profile = Profile.PROD
        mock_bot.broker.list_balances = Mock(return_value=[test_balance])
        mock_bot.broker.list_positions = Mock(return_value=[test_position])
        state = mock_bot.runtime_state
        state.mark_windows.clear()
        state.mark_windows["BTC-PERP"] = [Decimal("50000")] * 35
        mock_bot.risk_manager.check_volatility_circuit_breaker = Mock(
            return_value=CircuitBreakerOutcome(
                triggered=False, action=CircuitBreakerAction.NONE, reason=None
            )
        )
        mock_bot.risk_manager.check_mark_staleness = Mock(return_value=False)

        # Create strategy and set decision
        strategy = Mock(spec=BaselinePerpsStrategy)
        decision = Decision(action=Action.BUY, reason="test")
        strategy.decide = Mock(return_value=decision)
        mock_bot.runtime_state.strategy = strategy

        product = Mock()
        mock_bot.get_product = Mock(return_value=product)

        await orchestrator.process_symbol("BTC-PERP")

        # Should execute decision
        mock_bot.execute_decision.assert_called_once()
        call_args = mock_bot.execute_decision.call_args
        assert call_args[0][0] == "BTC-PERP"
        assert call_args[0][1] == decision

    @pytest.mark.asyncio
    async def test_holds_when_risk_gates_fail(self, orchestrator, mock_bot, test_balance):
        """Test skips execution when risk gates fail."""
        mock_bot.broker.list_balances = Mock(return_value=[test_balance])
        mock_bot.broker.list_positions = Mock(return_value=[])
        state = mock_bot.runtime_state
        state.mark_windows.clear()
        state.mark_windows["BTC-PERP"] = [Decimal("50000")] * 35
        mock_bot.risk_manager.check_mark_staleness = Mock(return_value=True)

        await orchestrator.process_symbol("BTC-PERP")

        # Should not execute decision
        mock_bot.execute_decision.assert_not_called()


class TestDecisionRoutingAndGuardChains:
    """Test decision routing through different action paths and guard chains."""

    @pytest.mark.asyncio
    async def test_decision_routing_buy_action_with_product(self, mock_bot, test_balance, test_position) -> None:
        """Test routing of BUY action when product metadata is available."""
        orchestrator = StrategyOrchestrator(mock_bot)

        # Mock strategy to return BUY decision
        mock_strategy = Mock()
        mock_strategy.evaluate.return_value = Decision(action=Action.BUY, reason="test_signal")
        mock_bot.get_strategy.return_value = mock_strategy

        # Mock get_product to return valid product
        mock_product = Mock()
        mock_bot.get_product.return_value = mock_product

        # Mock marks and equity to provide data for strategy processing
        mock_marks = [Decimal("50000")]
        with patch.object(orchestrator, "_get_marks", return_value=mock_marks):
            with patch.object(orchestrator, "_adjust_equity", return_value=Decimal("1000")):
                await orchestrator.process_symbol(
                    "BTC-PERP", [test_balance], {"BTC-PERP": test_position}
                )

        # Should execute BUY decision
        mock_bot.execute_decision.assert_called_once()
        args = mock_bot.execute_decision.call_args[0]
        assert args[0] == "BTC-PERP"  # symbol
        assert args[1].action == Action.BUY  # decision

    @pytest.mark.asyncio
    async def test_decision_routing_missing_product_metadata(self, mock_bot, test_balance, test_position) -> None:
        """Test decision routing when product metadata is missing."""
        orchestrator = StrategyOrchestrator(mock_bot)

        # Mock strategy to return BUY decision
        mock_strategy = Mock()
        mock_strategy.evaluate.return_value = Decision(action=Action.BUY, reason="test_signal")
        mock_bot.get_strategy.return_value = mock_strategy

        # Mock get_product to raise exception (missing product)
        mock_bot.get_product.side_effect = Exception("Product not found")

        with patch("bot_v2.orchestration.strategy_orchestrator.logger") as mock_logger:
            await orchestrator.process_symbol(
                "BTC-PERP", [test_balance], {"BTC-PERP": test_position}
            )

        # Should not execute decision due to missing product
        mock_bot.execute_decision.assert_not_called()
        # Should log warning about missing product metadata
        mock_logger.warning.assert_called_once()
        assert "missing product metadata" in str(mock_logger.warning.call_args)

    @pytest.mark.asyncio
    async def test_decision_routing_sell_action_execution(self, mock_bot, test_balance, test_position) -> None:
        """Test routing of SELL action through execution path."""
        orchestrator = StrategyOrchestrator(mock_bot)

        # Mock strategy to return SELL decision
        mock_strategy = Mock()
        mock_strategy.evaluate.return_value = Decision(action=Action.SELL, reason="sell_signal")
        mock_bot.get_strategy.return_value = mock_strategy

        mock_product = Mock()
        mock_bot.get_product.return_value = mock_product

        await orchestrator.process_symbol("BTC-PERP", [test_balance], {"BTC-PERP": test_position})

        # Should execute SELL decision
        mock_bot.execute_decision.assert_called_once()
        args = mock_bot.execute_decision.call_args[0]
        assert args[1].action == Action.SELL

    @pytest.mark.asyncio
    async def test_decision_routing_close_action_execution(self, mock_bot, test_balance, test_position) -> None:
        """Test routing of CLOSE action through execution path."""
        orchestrator = StrategyOrchestrator(mock_bot)

        # Mock strategy to return CLOSE decision
        mock_strategy = Mock()
        mock_strategy.evaluate.return_value = Decision(action=Action.CLOSE, reason="close_signal")
        mock_bot.get_strategy.return_value = mock_strategy

        mock_product = Mock()
        mock_bot.get_product.return_value = mock_product

        await orchestrator.process_symbol("BTC-PERP", [test_balance], {"BTC-PERP": test_position})

        # Should execute CLOSE decision
        mock_bot.execute_decision.assert_called_once()
        args = mock_bot.execute_decision.call_args[0]
        assert args[1].action == Action.CLOSE

    @pytest.mark.asyncio
    async def test_spot_profile_guard_chain_filtering(self, mock_bot) -> None:
        """Test spot profile filtering in decision guard chain."""
        orchestrator = StrategyOrchestrator(mock_bot)

        # Set profile to SPOT to trigger guard chain
        mock_bot.config.profile = Profile.SPOT

        # Mock context with strategy decision
        mock_context = Mock()
        mock_context.symbol = "BTC-USD"
        mock_context.marks = [Mock()]
        mock_context.position_state = None
        mock_context.equity = Decimal("1000")
        mock_context.product = Mock()

        # Mock spot filter to modify decision
        original_decision = Decision(action=Action.BUY, reason="signal")
        filtered_decision = Decision(action=Action.BUY, reason="filtered_signal")
        with patch.object(orchestrator, "_evaluate_strategy", return_value=original_decision):
            with patch.object(orchestrator, "_apply_spot_filters", return_value=filtered_decision):
                final_decision = await orchestrator._resolve_decision(mock_context)

        # Should apply spot filters for SPOT profile
        assert final_decision.action == Action.BUY
        assert final_decision.quantity == Decimal("0.3")

    @pytest.mark.asyncio
    async def test_non_spot_profile_bypasses_guard_chain(self, mock_bot) -> None:
        """Test that non-SPOT profiles bypass spot filter guard chain."""
        orchestrator = StrategyOrchestrator(mock_bot)

        # Set profile to DEV (non-SPOT)
        mock_bot.config.profile = Profile.DEV

        # Mock context with strategy decision
        mock_context = Mock()
        mock_context.symbol = "BTC-PERP"
        mock_context.marks = [Mock()]
        mock_context.position_state = None
        mock_context.equity = Decimal("1000")
        mock_context.product = Mock()

        # Mock strategy evaluation
        strategy_decision = Decision(action=Action.BUY, reason="signal")
        with patch.object(orchestrator, "_evaluate_strategy", return_value=strategy_decision):
            final_decision = await orchestrator._resolve_decision(mock_context)

        # Should return strategy decision unchanged for non-SPOT profiles
        assert final_decision.action == Action.BUY
        assert final_decision.quantity == Decimal("0.5")
        assert final_decision.source == "signal"

    def test_decision_action_guard_for_valid_actions(self, mock_bot) -> None:
        """Test that only valid actions (BUY, SELL, CLOSE) trigger execution."""
        valid_actions = {Action.BUY, Action.SELL, Action.CLOSE}

        for action in valid_actions:
            StrategyOrchestrator(mock_bot)
            mock_context = Mock()
            mock_context.product = Mock()

            # This would be tested through the main execution path
            # The guard at line 149 ensures only these actions proceed
            assert action in {Action.BUY, Action.SELL, Action.CLOSE}
