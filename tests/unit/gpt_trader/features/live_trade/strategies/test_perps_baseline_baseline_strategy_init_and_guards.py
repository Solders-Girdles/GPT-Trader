"""Tests for BaselinePerpsStrategy initialization and guard paths."""

from __future__ import annotations

from decimal import Decimal

from gpt_trader.features.live_trade.strategies.perps_baseline.strategy import (
    Action,
    BaselinePerpsStrategy,
    PerpsStrategyConfig,
)
from tests.unit.gpt_trader.features.live_trade.strategies.perps_baseline_test_helpers import (
    make_uptrend,
)


class TestBaselinePerpsStrategyInit:
    """Tests for strategy initialization."""

    def test_default_config(self) -> None:
        strategy = BaselinePerpsStrategy()
        assert strategy.config.long_ma_period == 20
        assert strategy.config.short_ma_period == 5

    def test_custom_config(self) -> None:
        config = PerpsStrategyConfig(long_ma_period=50, rsi_period=21)
        strategy = BaselinePerpsStrategy(config=config)
        assert strategy.config.long_ma_period == 50
        assert strategy.config.rsi_period == 21


class TestStrategyInsufficientData:
    """Tests for strategy behavior with insufficient data."""

    def test_empty_marks_returns_hold(self) -> None:
        strategy = BaselinePerpsStrategy()
        decision = strategy.decide(
            symbol="BTC-PERP",
            current_mark=Decimal("50000"),
            position_state=None,
            recent_marks=[],
            equity=Decimal("10000"),
            product=None,
        )
        assert decision.action == Action.HOLD
        assert "Insufficient data" in decision.reason

    def test_few_marks_returns_hold(self) -> None:
        strategy = BaselinePerpsStrategy()
        decision = strategy.decide(
            symbol="BTC-PERP",
            current_mark=Decimal("50000"),
            position_state=None,
            recent_marks=[Decimal("49000"), Decimal("50000")],
            equity=Decimal("10000"),
            product=None,
        )
        assert decision.action == Action.HOLD
        assert "Insufficient data" in decision.reason


class TestStrategyKillSwitch:
    """Tests for kill switch functionality."""

    def test_kill_switch_returns_hold(self) -> None:
        config = PerpsStrategyConfig(kill_switch_enabled=True)
        strategy = BaselinePerpsStrategy(config=config)
        decision = strategy.decide(
            symbol="BTC-PERP",
            current_mark=Decimal("50000"),
            position_state=None,
            recent_marks=make_uptrend(30),
            equity=Decimal("10000"),
            product=None,
        )
        assert decision.action == Action.HOLD
        assert "Kill switch" in decision.reason
        assert decision.indicators.get("kill_switch") is True
