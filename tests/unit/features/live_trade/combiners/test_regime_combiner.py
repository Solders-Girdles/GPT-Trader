"""
Unit tests for RegimeAwareCombiner.
"""

from dataclasses import dataclass
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from gpt_trader.features.live_trade.combiners.regime import (
    RegimeAwareCombiner,
    RegimeCombinerConfig,
)
from gpt_trader.features.live_trade.signals.protocol import StrategyContext
from gpt_trader.features.live_trade.signals.types import SignalOutput, SignalType


@dataclass
class MockCandle:
    high: Decimal
    low: Decimal
    close: Decimal


@pytest.fixture
def mock_context():
    return MagicMock(spec=StrategyContext)


class TestRegimeAwareCombiner:
    def test_neutral_regime_no_candles(self, mock_context):
        """Test behavior when no candles are available (Neutral Regime)."""
        config = RegimeCombinerConfig()
        combiner = RegimeAwareCombiner(config)

        mock_context.candles = None

        # Create conflicting signals
        # Trend: +1.0
        # MeanRev: -1.0
        signals = [
            SignalOutput("trend", SignalType.TREND, 1.0, 1.0),
            SignalOutput("mr", SignalType.MEAN_REVERSION, -1.0, 1.0),
        ]

        output = combiner.combine(signals, mock_context)

        # In Neutral (default), weights are blended 50/50.
        # Trend Weight: (1.0 + 0.0) / 2 = 0.5
        # MeanRev Weight: (0.0 + 1.0) / 2 = 0.5
        # Net: 1.0 * 0.5 + (-1.0) * 0.5 = 0.0

        assert output.strength == 0.0
        assert output.metadata["regime"] == "neutral"

    def test_trending_regime(self, mock_context):
        """Test behavior in Trending Regime (High ADX)."""
        config = RegimeCombinerConfig(trending_threshold=25)
        combiner = RegimeAwareCombiner(config)

        # Create candles that produce high ADX (strong trend)
        # Price increasing by 2 every step, no overlap
        candles = []
        for i in range(50):
            base = Decimal(100 + i * 2)
            candles.append(MockCandle(base + 2, base, base + 1))

        mock_context.candles = candles

        # Signals
        signals = [
            SignalOutput("trend", SignalType.TREND, 1.0, 1.0),
            SignalOutput("mr", SignalType.MEAN_REVERSION, -1.0, 1.0),
        ]

        output = combiner.combine(signals, mock_context)

        # In Trending Regime:
        # Trend Weight: 1.0
        # MeanRev Weight: 0.0
        # Net: 1.0 * 1.0 + (-1.0) * 0.0 = 1.0

        assert output.metadata["regime"] == "trending"
        assert output.strength == 1.0
        assert output.metadata["adx"] > 25

    def test_ranging_regime(self, mock_context):
        """Test behavior in Ranging Regime (Low ADX)."""
        config = RegimeCombinerConfig(ranging_threshold=20)
        combiner = RegimeAwareCombiner(config)

        # Create candles that produce low ADX (chop)
        # Alternating up/down
        candles = []
        for i in range(50):
            base = Decimal(100)
            if i % 2 == 0:
                candles.append(MockCandle(base + 2, base - 2, base + 1))
            else:
                candles.append(MockCandle(base + 2, base - 2, base - 1))

        mock_context.candles = candles

        # Signals
        signals = [
            SignalOutput("trend", SignalType.TREND, 1.0, 1.0),
            SignalOutput("mr", SignalType.MEAN_REVERSION, -1.0, 1.0),
        ]

        output = combiner.combine(signals, mock_context)

        # In Ranging Regime:
        # Trend Weight: 0.0
        # MeanRev Weight: 1.0
        # Net: 1.0 * 0.0 + (-1.0) * 1.0 = -1.0

        assert output.metadata["regime"] == "ranging"
        assert output.strength == -1.0
        assert output.metadata["adx"] < 20
