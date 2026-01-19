"""Tests for baseline perps strategy model objects."""

from __future__ import annotations

from decimal import Decimal

from gpt_trader.features.live_trade.strategies.perps_baseline.strategy import (
    Action,
    Decision,
    IndicatorState,
    PerpsStrategyConfig,
)


class TestPerpsStrategyConfig:
    """Tests for PerpsStrategyConfig dataclass."""

    def test_default_values(self) -> None:
        config = PerpsStrategyConfig()
        assert config.long_ma_period == 20
        assert config.short_ma_period == 5
        assert config.rsi_period == 14
        assert config.rsi_overbought == 70
        assert config.rsi_oversold == 30
        assert config.stop_loss_pct == 0.02
        assert config.take_profit_pct == 0.05

    def test_custom_values(self) -> None:
        config = PerpsStrategyConfig(
            long_ma_period=50,
            short_ma_period=10,
            rsi_period=7,
        )
        assert config.long_ma_period == 50
        assert config.short_ma_period == 10
        assert config.rsi_period == 7


class TestDecision:
    """Tests for Decision dataclass."""

    def test_basic_decision(self) -> None:
        decision = Decision(Action.BUY, "test reason", 0.8)
        assert decision.action == Action.BUY
        assert decision.reason == "test reason"
        assert decision.confidence == 0.8

    def test_decision_with_indicators(self) -> None:
        indicators = {"rsi": 25.5, "trend": "bullish"}
        decision = Decision(Action.SELL, "overbought", 0.7, indicators=indicators)
        assert decision.indicators["rsi"] == 25.5
        assert decision.indicators["trend"] == "bullish"


class TestIndicatorState:
    """Tests for IndicatorState dataclass."""

    def test_default_state(self) -> None:
        state = IndicatorState()
        assert state.rsi is None
        assert state.short_ma is None
        assert state.long_ma is None
        assert state.crossover_signal == "none"
        assert state.trend == "neutral"
        assert state.rsi_signal == "neutral"

    def test_state_assignment(self) -> None:
        state = IndicatorState(
            rsi=Decimal("45"),
            short_ma=Decimal("100"),
            long_ma=Decimal("98"),
            crossover_signal="bullish",
            trend="bullish",
            rsi_signal="neutral",
        )
        assert state.rsi == Decimal("45")
        assert state.crossover_signal == "bullish"
