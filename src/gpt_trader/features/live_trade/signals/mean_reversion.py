"""
Mean reversion signal generator using Z-Score.
"""

import statistics
from dataclasses import dataclass

from gpt_trader.features.live_trade.signals.protocol import (
    SignalGenerator,
    StrategyContext,
)
from gpt_trader.features.live_trade.signals.types import SignalOutput, SignalType


@dataclass
class MeanReversionSignalConfig:
    """Configuration for MeanReversionSignal."""

    window: int = 20
    z_entry_threshold: float = 2.0
    z_exit_threshold: float = 0.5


class MeanReversionSignal(SignalGenerator):
    """Generates signals based on Z-Score mean reversion."""

    def __init__(self, config: MeanReversionSignalConfig) -> None:
        self.config = config

    def generate(self, context: StrategyContext) -> SignalOutput:
        """Generate mean reversion signal."""
        # Minimum data requirement
        if len(context.recent_marks) < self.config.window:
            return SignalOutput(
                name="mean_reversion_z",
                type=SignalType.MEAN_REVERSION,
                strength=0.0,
                confidence=0.0,
                metadata={"reason": "insufficient_data"},
            )

        # Calculate Z-Score
        prices = [float(p) for p in context.recent_marks]
        window_prices = prices[-self.config.window :]

        if len(window_prices) < 2:
            return SignalOutput(
                name="mean_reversion_z",
                type=SignalType.MEAN_REVERSION,
                strength=0.0,
                confidence=0.0,
                metadata={"reason": "insufficient_data"},
            )

        mean = statistics.mean(window_prices)
        std = statistics.stdev(window_prices)

        if std == 0:
            return SignalOutput(
                name="mean_reversion_z",
                type=SignalType.MEAN_REVERSION,
                strength=0.0,
                confidence=0.0,
                metadata={"reason": "zero_volatility"},
            )

        current_price = prices[-1]
        z_score = (current_price - mean) / std

        strength = 0.0
        confidence = 0.0
        reason = "neutral"

        # Logic:
        # Z < -Threshold -> Price is cheap -> Buy (Positive Strength)
        # Z > +Threshold -> Price is expensive -> Sell (Negative Strength)

        if z_score < -self.config.z_entry_threshold:
            # Strong Buy
            strength = 1.0
            confidence = min(abs(z_score) / self.config.z_entry_threshold * 0.5, 0.9)
            reason = "oversold_z_score"
        elif z_score > self.config.z_entry_threshold:
            # Strong Sell
            strength = -1.0
            confidence = min(abs(z_score) / self.config.z_entry_threshold * 0.5, 0.9)
            reason = "overbought_z_score"
        elif abs(z_score) < self.config.z_exit_threshold:
            # Near mean -> Neutral / Close
            strength = 0.0
            confidence = 0.5
            reason = "mean_reverted"
        else:
            # In between -> Weak signal (continuation of previous?)
            # For now, neutral
            strength = 0.0
            reason = "neutral_zone"

        return SignalOutput(
            name="mean_reversion_z",
            type=SignalType.MEAN_REVERSION,
            strength=strength,
            confidence=confidence,
            metadata={
                "z_score": z_score,
                "mean": mean,
                "std": std,
                "reason": reason,
            },
        )
