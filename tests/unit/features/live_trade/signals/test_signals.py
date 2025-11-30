"""
Unit tests for Signal Generators.
"""

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from gpt_trader.features.live_trade.signals.mean_reversion import (
    MeanReversionSignal,
    MeanReversionSignalConfig,
)
from gpt_trader.features.live_trade.signals.momentum import (
    MomentumSignal,
    MomentumSignalConfig,
)
from gpt_trader.features.live_trade.signals.protocol import StrategyContext
from gpt_trader.features.live_trade.signals.trend import TrendSignal, TrendSignalConfig
from gpt_trader.features.live_trade.signals.types import SignalType


@pytest.fixture
def mock_context():
    return MagicMock(spec=StrategyContext)


class TestTrendSignal:
    def test_insufficient_data(self, mock_context):
        config = TrendSignalConfig(slow_period=20)
        signal_gen = TrendSignal(config)

        mock_context.recent_marks = [Decimal("100")] * 10  # Only 10 points

        output = signal_gen.generate(mock_context)
        assert output.strength == 0.0
        assert output.metadata["reason"] == "insufficient_data"

    def test_bullish_trend(self, mock_context):
        config = TrendSignalConfig(fast_period=5, slow_period=10)
        signal_gen = TrendSignal(config)

        # Create uptrend: 100, 101, ... 119
        prices = [Decimal(100 + i) for i in range(20)]
        mock_context.recent_marks = prices

        output = signal_gen.generate(mock_context)

        # Fast MA (last 5) > Slow MA (last 10)
        # Fast MA ~ 117. Slow MA ~ 114.5.
        assert output.type == SignalType.TREND
        assert output.strength > 0  # Bullish
        assert output.metadata["reason"] in ["bullish_trend", "bullish_crossover"]

    def test_bearish_trend(self, mock_context):
        config = TrendSignalConfig(fast_period=5, slow_period=10)
        signal_gen = TrendSignal(config)

        # Create downtrend: 100, 99, ... 81
        prices = [Decimal(100 - i) for i in range(20)]
        mock_context.recent_marks = prices

        output = signal_gen.generate(mock_context)

        assert output.type == SignalType.TREND
        assert output.strength < 0  # Bearish
        assert output.metadata["reason"] in ["bearish_trend", "bearish_crossover"]


class TestMeanReversionSignal:
    def test_oversold(self, mock_context):
        config = MeanReversionSignalConfig(window=10, z_entry_threshold=2.0)
        signal_gen = MeanReversionSignal(config)

        # Mean 100, StdDev 0 (initially) -> Force deviation
        # [100, 100, ..., 100, 80]
        prices = [Decimal("100")] * 9 + [Decimal("80")]
        mock_context.recent_marks = prices

        # Mean = (900 + 80) / 10 = 98.
        # Variance sum: 9*(100-98)^2 + (80-98)^2 = 9*4 + 324 = 36 + 324 = 360.
        # StdDev = sqrt(360/9) = sqrt(40) = 6.32.
        # Z = (80 - 98) / 6.32 = -18 / 6.32 = -2.85.
        # Should trigger Oversold (Buy)

        output = signal_gen.generate(mock_context)

        assert output.type == SignalType.MEAN_REVERSION
        assert output.strength > 0  # Buy signal
        assert output.metadata["reason"] == "oversold_z_score"

    def test_overbought(self, mock_context):
        config = MeanReversionSignalConfig(window=10, z_entry_threshold=2.0)
        signal_gen = MeanReversionSignal(config)

        # [100, ..., 100, 120]
        prices = [Decimal("100")] * 9 + [Decimal("120")]
        mock_context.recent_marks = prices

        output = signal_gen.generate(mock_context)

        assert output.type == SignalType.MEAN_REVERSION
        assert output.strength < 0  # Sell signal
        assert output.metadata["reason"] == "overbought_z_score"


class TestMomentumSignal:
    def test_rsi_oversold(self, mock_context):
        config = MomentumSignalConfig(period=14, oversold=30)
        signal_gen = MomentumSignal(config)

        # Create a sequence that results in low RSI
        # 14 drops in a row
        prices = [Decimal(100 - i) for i in range(20)]
        mock_context.recent_marks = prices

        # RSI should be near 0
        output = signal_gen.generate(mock_context)

        assert output.type == SignalType.MEAN_REVERSION
        assert output.strength > 0  # Buy signal (reversion from oversold)
        assert output.metadata["rsi"] < 30
        assert output.metadata["reason"] == "oversold"
