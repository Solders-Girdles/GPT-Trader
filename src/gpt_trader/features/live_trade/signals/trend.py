"""
Trend following signal generator using Moving Average Crossovers.
"""

from dataclasses import dataclass

from gpt_trader.features.live_trade.indicators import (
    compute_ma_series,
    detect_crossover,
    simple_moving_average,
)
from gpt_trader.features.live_trade.signals.protocol import (
    SignalGenerator,
    StrategyContext,
)
from gpt_trader.features.live_trade.signals.types import SignalOutput, SignalType


@dataclass
class TrendSignalConfig:
    """Configuration for TrendSignal."""

    fast_period: int = 5
    slow_period: int = 20
    smoothing_type: str = "sma"  # "sma" or "ema"


class TrendSignal(SignalGenerator):
    """Generates signals based on Moving Average crossovers and trend direction."""

    def __init__(self, config: TrendSignalConfig) -> None:
        self.config = config

    def generate(self, context: StrategyContext) -> SignalOutput:
        """Generate trend signal."""
        # Minimum data requirement
        min_data = self.config.slow_period
        if len(context.recent_marks) < min_data:
            return SignalOutput(
                name="trend_ma",
                type=SignalType.TREND,
                strength=0.0,
                confidence=0.0,
                metadata={"reason": "insufficient_data"},
            )

        # Calculate MAs
        prices = list(context.recent_marks)
        fast_ma = simple_moving_average(prices, self.config.fast_period)
        slow_ma = simple_moving_average(prices, self.config.slow_period)

        if fast_ma is None or slow_ma is None:
            return SignalOutput(
                name="trend_ma",
                type=SignalType.TREND,
                strength=0.0,
                confidence=0.0,
                metadata={"reason": "calculation_error"},
            )

        # Detect Crossover (for immediate signal)
        fast_series = compute_ma_series(prices, self.config.fast_period, self.config.smoothing_type)
        slow_series = compute_ma_series(prices, self.config.slow_period, self.config.smoothing_type)
        crossover = detect_crossover(fast_series, slow_series, lookback=2)

        strength = 0.0
        confidence = 0.5
        reason = "neutral"

        # Determine Trend Direction
        if fast_ma > slow_ma:
            strength = 0.5  # Base bullish
            reason = "bullish_trend"
            # Boost if crossover just happened
            if crossover and crossover.direction == "bullish":
                strength = 1.0
                reason = "bullish_crossover"
                confidence = 0.8
        elif fast_ma < slow_ma:
            strength = -0.5  # Base bearish
            reason = "bearish_trend"
            # Boost if crossover just happened
            if crossover and crossover.direction == "bearish":
                strength = -1.0
                reason = "bearish_crossover"
                confidence = 0.8

        return SignalOutput(
            name="trend_ma",
            type=SignalType.TREND,
            strength=strength,
            confidence=confidence,
            metadata={
                "fast_ma": float(fast_ma),
                "slow_ma": float(slow_ma),
                "reason": reason,
            },
        )
