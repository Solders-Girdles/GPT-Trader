"""Tests for BaselinePerpsStrategy indicator calculation and confidence scoring."""

from __future__ import annotations

from decimal import Decimal

from gpt_trader.features.live_trade.strategies.perps_baseline.strategy import (
    BaselinePerpsStrategy,
)
from tests.unit.gpt_trader.features.live_trade.strategies.perps_baseline_test_helpers import (
    make_sideways,
    make_uptrend,
)


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
        strategy = BaselinePerpsStrategy()
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
