"""Tests for BaselinePerpsStrategy entry/exit signal generation."""

from __future__ import annotations

from decimal import Decimal

from gpt_trader.features.live_trade.strategies.perps_baseline.strategy import (
    Action,
    BaselinePerpsStrategy,
    PerpsStrategyConfig,
)
from tests.unit.gpt_trader.features.live_trade.strategies.perps_baseline_test_helpers import (
    make_downtrend,
    make_sideways,
    make_uptrend,
)


class TestStrategyEntrySignals:
    """Tests for entry signal generation."""

    def test_bullish_trend_generates_buy(self) -> None:
        # Create a clear uptrend
        prices = make_uptrend(30)
        strategy = BaselinePerpsStrategy(config=PerpsStrategyConfig(min_confidence=0.3))
        decision = strategy.decide(
            symbol="BTC-PERP",
            current_mark=prices[-1],
            position_state=None,
            recent_marks=prices,
            equity=Decimal("10000"),
            product=None,
        )
        # Should generate buy or hold (depending on RSI)
        assert decision.action in (Action.BUY, Action.HOLD)
        assert decision.indicators.get("trend") == "bullish"

    def test_bearish_trend_generates_sell(self) -> None:
        # Create a clear downtrend
        prices = make_downtrend(30)
        strategy = BaselinePerpsStrategy(config=PerpsStrategyConfig(min_confidence=0.3))
        decision = strategy.decide(
            symbol="BTC-PERP",
            current_mark=prices[-1],
            position_state=None,
            recent_marks=prices,
            equity=Decimal("10000"),
            product=None,
        )
        # Should generate sell or hold (depending on RSI)
        assert decision.action in (Action.SELL, Action.HOLD)
        assert decision.indicators.get("trend") == "bearish"

    def test_sideways_market_holds(self) -> None:
        prices = make_sideways(30)
        strategy = BaselinePerpsStrategy()
        decision = strategy.decide(
            symbol="BTC-PERP",
            current_mark=prices[-1],
            position_state=None,
            recent_marks=prices,
            equity=Decimal("10000"),
            product=None,
        )
        # Sideways should typically hold
        assert decision.action == Action.HOLD


class TestStrategyExitSignals:
    """Tests for exit signal generation."""

    def test_long_position_exit_on_bearish_signal(self) -> None:
        # Create downtrend that should trigger exit
        prices = make_downtrend(30)
        strategy = BaselinePerpsStrategy(config=PerpsStrategyConfig(min_confidence=0.3))
        decision = strategy.decide(
            symbol="BTC-PERP",
            current_mark=prices[-1],
            position_state={"quantity": Decimal("1"), "side": "long"},
            recent_marks=prices,
            equity=Decimal("10000"),
            product=None,
        )
        # Should suggest closing or holding
        assert decision.action in (Action.CLOSE, Action.HOLD)

    def test_stop_loss_triggers_close(self) -> None:
        prices = make_sideways(30)
        strategy = BaselinePerpsStrategy(config=PerpsStrategyConfig(stop_loss_pct=0.02))
        # Entry at 100, current at 97 = -3% loss
        decision = strategy.decide(
            symbol="BTC-PERP",
            current_mark=Decimal("97"),
            position_state={
                "quantity": Decimal("1"),
                "side": "long",
                "entry_price": Decimal("100"),
            },
            recent_marks=prices,
            equity=Decimal("10000"),
            product=None,
        )
        assert decision.action == Action.CLOSE
        assert "Stop loss" in decision.reason

    def test_take_profit_triggers_close(self) -> None:
        prices = make_sideways(30)
        strategy = BaselinePerpsStrategy(config=PerpsStrategyConfig(take_profit_pct=0.05))
        # Entry at 100, current at 106 = +6% profit
        decision = strategy.decide(
            symbol="BTC-PERP",
            current_mark=Decimal("106"),
            position_state={
                "quantity": Decimal("1"),
                "side": "long",
                "entry_price": Decimal("100"),
            },
            recent_marks=prices,
            equity=Decimal("10000"),
            product=None,
        )
        assert decision.action == Action.CLOSE
        assert "Take profit" in decision.reason
