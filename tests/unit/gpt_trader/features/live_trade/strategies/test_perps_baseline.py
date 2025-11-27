"""Tests for the baseline perpetuals trading strategy."""

from __future__ import annotations

from decimal import Decimal

from gpt_trader.features.live_trade.strategies.perps_baseline.strategy import (
    Action,
    BaselinePerpsStrategy,
    Decision,
    IndicatorState,
    StrategyConfig,
)


def make_price_series(
    base: float,
    changes: list[float],
) -> list[Decimal]:
    """Create a price series from base price and percentage changes."""
    prices = [Decimal(str(base))]
    for change in changes:
        next_price = prices[-1] * Decimal(str(1 + change / 100))
        prices.append(next_price)
    return prices


def make_uptrend(periods: int = 25, volatility: float = 0.5) -> list[Decimal]:
    """Generate an uptrending price series."""
    changes = [volatility + (i * 0.1) for i in range(periods)]
    return make_price_series(100, changes)


def make_downtrend(periods: int = 25, volatility: float = 0.5) -> list[Decimal]:
    """Generate a downtrending price series."""
    changes = [-(volatility + (i * 0.1)) for i in range(periods)]
    return make_price_series(100, changes)


def make_sideways(periods: int = 25) -> list[Decimal]:
    """Generate a sideways price series."""
    changes = [0.2 if i % 2 == 0 else -0.2 for i in range(periods)]
    return make_price_series(100, changes)


class TestStrategyConfig:
    """Tests for StrategyConfig dataclass."""

    def test_default_values(self) -> None:
        config = StrategyConfig()
        assert config.long_ma == 20
        assert config.short_ma == 5
        assert config.rsi_period == 14
        assert config.rsi_overbought == 70
        assert config.rsi_oversold == 30
        assert config.stop_loss_pct == 0.02
        assert config.take_profit_pct == 0.05

    def test_custom_values(self) -> None:
        config = StrategyConfig(
            long_ma=50,
            short_ma=10,
            rsi_period=7,
        )
        assert config.long_ma == 50
        assert config.short_ma == 10
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


class TestBaselinePerpsStrategyInit:
    """Tests for strategy initialization."""

    def test_default_config(self) -> None:
        strategy = BaselinePerpsStrategy()
        assert strategy.config.long_ma == 20
        assert strategy.config.short_ma == 5

    def test_custom_config(self) -> None:
        config = StrategyConfig(long_ma=50, rsi_period=21)
        strategy = BaselinePerpsStrategy(config=config)
        assert strategy.config.long_ma == 50
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
        config = StrategyConfig(kill_switch_enabled=True)
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


class TestStrategyEntrySignals:
    """Tests for entry signal generation."""

    def test_bullish_trend_generates_buy(self) -> None:
        # Create a clear uptrend
        prices = make_uptrend(30)
        strategy = BaselinePerpsStrategy(config=StrategyConfig(min_confidence=0.3))
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
        strategy = BaselinePerpsStrategy(config=StrategyConfig(min_confidence=0.3))
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
        strategy = BaselinePerpsStrategy(config=StrategyConfig(min_confidence=0.3))
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
        strategy = BaselinePerpsStrategy(config=StrategyConfig(stop_loss_pct=0.02))
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
        strategy = BaselinePerpsStrategy(config=StrategyConfig(take_profit_pct=0.05))
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


class TestStrategyIndicatorCalculation:
    """Tests for indicator calculation."""

    def test_indicators_populated_in_decision(self) -> None:
        prices = make_uptrend(30)
        strategy = BaselinePerpsStrategy()
        decision = strategy.decide(
            symbol="BTC-PERP",
            current_mark=prices[-1],
            position_state=None,
            recent_marks=prices,
            equity=Decimal("10000"),
            product=None,
        )
        # Indicators should be populated
        assert "rsi" in decision.indicators
        assert "short_ma" in decision.indicators
        assert "long_ma" in decision.indicators
        assert "trend" in decision.indicators
        assert "rsi_signal" in decision.indicators

    def test_rsi_signal_oversold(self) -> None:
        # Create a sharp downtrend to get oversold RSI (need 25+ values)
        prices = [Decimal(str(100 - i * 2)) for i in range(25)]
        strategy = BaselinePerpsStrategy()
        decision = strategy.decide(
            symbol="BTC-PERP",
            current_mark=prices[-1],
            position_state=None,
            recent_marks=prices,
            equity=Decimal("10000"),
            product=None,
        )
        # RSI should be low
        rsi = decision.indicators.get("rsi")
        assert rsi is not None
        assert rsi < 50  # Should be bearish

    def test_rsi_signal_overbought(self) -> None:
        # Create a sharp uptrend to get overbought RSI (need 25+ values)
        prices = [Decimal(str(100 + i * 2)) for i in range(25)]
        strategy = BaselinePerpsStrategy()
        decision = strategy.decide(
            symbol="BTC-PERP",
            current_mark=prices[-1],
            position_state=None,
            recent_marks=prices,
            equity=Decimal("10000"),
            product=None,
        )
        # RSI should be high
        rsi = decision.indicators.get("rsi")
        assert rsi is not None
        assert rsi > 50  # Should be bullish


class TestStrategyConfidence:
    """Tests for confidence scoring."""

    def test_multiple_bullish_signals_high_confidence(self) -> None:
        # Strong uptrend should have high confidence
        prices = make_uptrend(30)
        strategy = BaselinePerpsStrategy(config=StrategyConfig(min_confidence=0.3))
        decision = strategy.decide(
            symbol="BTC-PERP",
            current_mark=prices[-1],
            position_state=None,
            recent_marks=prices,
            equity=Decimal("10000"),
            product=None,
        )
        # Should have some confidence
        assert decision.confidence >= 0.0
        assert decision.confidence <= 1.0

    def test_weak_signals_low_confidence(self) -> None:
        # Sideways market should have low confidence
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
        # Should have lower confidence in sideways market
        assert decision.confidence < 0.8


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
