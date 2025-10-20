"""
Spot profile override tests for StrategyOrchestrator.
"""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import Mock

import pytest

from bot_v2.features.live_trade.strategies.perps_baseline import Action, Decision
from bot_v2.orchestration.configuration import Profile
from bot_v2.orchestration.strategy_orchestrator import SymbolProcessingContext


class TestSpotProfileFilters:
    """Cover candle fetching and filter overrides."""

    @pytest.mark.asyncio
    async def test_candle_fetch_failure_records_hold(
        self, async_orchestrator, fake_perps_bot, test_balance, test_product
    ):
        fake_perps_bot.config.profile = Profile.SPOT
        fake_perps_bot.broker.list_balances.return_value = [test_balance]
        fake_perps_bot.broker.list_positions.return_value = []
        fake_perps_bot.runtime_state.mark_windows["BTC-PERP"] = [Decimal("50000")] * 35
        fake_perps_bot.get_product.return_value = test_product

        cb_outcome = Mock()
        cb_outcome.triggered = False
        fake_perps_bot.risk_manager.check_volatility_circuit_breaker.return_value = cb_outcome
        fake_perps_bot.risk_manager.check_mark_staleness.return_value = False

        strategy = Mock()
        decision = Decision(action=Action.BUY, reason="test")
        strategy.decide.return_value = decision
        fake_perps_bot.runtime_state.symbol_strategies["BTC-PERP"] = strategy

        fake_spot_profile_service = async_orchestrator._spot_profiles
        fake_spot_profile_service.get.return_value = {
            "volume_filter": {"window": 20, "multiplier": 2.0}
        }

        fake_perps_bot.broker.get_candles.return_value = []

        await async_orchestrator.process_symbol("BTC-PERP")

        fake_perps_bot.execute_decision.assert_not_called()
        recorded_decision = fake_perps_bot.last_decisions["BTC-PERP"]
        assert recorded_decision.action == Action.HOLD
        assert "indicator_data_unavailable" in recorded_decision.reason

    @pytest.mark.asyncio
    async def test_spot_filters_execute_when_passing(
        self, async_orchestrator, fake_perps_bot, test_balance, test_position, test_product
    ):
        fake_perps_bot.config.profile = Profile.SPOT
        fake_perps_bot.broker.list_balances.return_value = [test_balance]
        fake_perps_bot.broker.list_positions.return_value = [test_position]
        fake_perps_bot.runtime_state.mark_windows["BTC-PERP"] = [Decimal("50000")] * 35
        fake_perps_bot.get_product.return_value = test_product

        cb_outcome = Mock()
        cb_outcome.triggered = False
        fake_perps_bot.risk_manager.check_volatility_circuit_breaker.return_value = cb_outcome
        fake_perps_bot.risk_manager.check_mark_staleness.return_value = False

        strategy = Mock()
        decision = Decision(action=Action.BUY, reason="test")
        strategy.decide.return_value = decision
        fake_perps_bot.runtime_state.symbol_strategies["BTC-PERP"] = strategy

        fake_spot_profile_service = async_orchestrator._spot_profiles
        fake_spot_profile_service.get.return_value = {
            "volume_filter": {"window": 20, "multiplier": 2.0},
            "momentum_filter": {"window": 14, "overbought": 70, "oversold": 30},
            "trend_filter": {"window": 20, "min_slope": 0.001},
        }

        mock_candles = []
        for i in range(30):
            candle = Mock()
            candle.close = Decimal("50000") + Decimal(str(i * 10))
            candle.volume = Decimal("2000")
            candle.high = candle.close * Decimal("1.01")
            candle.low = candle.close * Decimal("0.99")
            candle.ts = datetime.now(timezone.utc)
            mock_candles.append(candle)

        fake_perps_bot.broker.get_candles.return_value = mock_candles

        await async_orchestrator.process_symbol("BTC-PERP")

        fake_perps_bot.execute_decision.assert_called_once()
        executed_decision = fake_perps_bot.execute_decision.call_args[0][1]
        assert executed_decision.action == Action.BUY

    @pytest.mark.asyncio
    async def test_apply_spot_filters_blocks_low_volume(self, async_orchestrator, fake_perps_bot):
        context = SymbolProcessingContext(
            symbol="BTC-PERP",
            balances=[],
            equity=Decimal("10000"),
            positions={},
            position_state=None,
            position_quantity=Decimal("0"),
            marks=[Decimal("50000")],
            product=None,
        )

        fake_spot_profile_service = async_orchestrator._spot_profiles
        fake_spot_profile_service.get.return_value = {
            "volume_filter": {"window": 20, "multiplier": 2.0}
        }

        mock_candles = []
        for _ in range(25):
            candle = Mock()
            candle.close = Decimal("50000")
            candle.volume = Decimal("1000")
            candle.high = Decimal("50000")
            candle.low = Decimal("50000")
            candle.ts = datetime.now(timezone.utc)
            mock_candles.append(candle)

        fake_perps_bot.broker.get_candles.return_value = mock_candles

        decision = Decision(action=Action.BUY, reason="test")
        result = await async_orchestrator._apply_spot_filters(context, decision)

        assert result.action == Action.HOLD
        assert "volume_filter_blocked" in result.reason
        fake_perps_bot.broker.get_candles.assert_called_once()

    @pytest.mark.asyncio
    async def test_spot_profile_overrides_to_hold(
        self, async_orchestrator, fake_perps_bot, test_balance, test_product
    ):
        fake_perps_bot.config.profile = Profile.SPOT
        fake_perps_bot.broker.list_balances.return_value = [test_balance]
        fake_perps_bot.broker.list_positions.return_value = []
        fake_perps_bot.runtime_state.mark_windows["BTC-PERP"] = [Decimal("50000")] * 35
        fake_perps_bot.get_product.return_value = test_product

        cb_outcome = Mock()
        cb_outcome.triggered = False
        fake_perps_bot.risk_manager.check_volatility_circuit_breaker.return_value = cb_outcome
        fake_perps_bot.risk_manager.check_mark_staleness.return_value = False

        strategy = Mock()
        original_decision = Decision(action=Action.BUY, reason="test")
        strategy.decide.return_value = original_decision
        fake_perps_bot.runtime_state.symbol_strategies["BTC-PERP"] = strategy

        fake_spot_profile_service = async_orchestrator._spot_profiles
        fake_spot_profile_service.get.return_value = {
            "volume_filter": {"window": 20, "multiplier": 2.0}
        }

        mock_candles = []
        for _ in range(25):
            candle = Mock()
            candle.close = Decimal("50000")
            candle.volume = Decimal("1000")
            candle.high = Decimal("50000")
            candle.low = Decimal("50000")
            candle.ts = datetime.now(timezone.utc)
            mock_candles.append(candle)

        fake_perps_bot.broker.get_candles.return_value = mock_candles

        await async_orchestrator.process_symbol("BTC-PERP")

        fake_perps_bot.execute_decision.assert_not_called()
        recorded_decision = fake_perps_bot.last_decisions["BTC-PERP"]
        assert recorded_decision.action == Action.HOLD
        assert "volume_filter_blocked" in recorded_decision.reason

    @pytest.mark.asyncio
    async def test_fetch_spot_candles_error_handling(self, async_orchestrator, fake_perps_bot):
        fake_perps_bot.broker.get_candles.return_value = []

        candles = await async_orchestrator._fetch_spot_candles("BTC-PERP", 20)

        assert candles == []
        fake_perps_bot.broker.get_candles.assert_called_once()
