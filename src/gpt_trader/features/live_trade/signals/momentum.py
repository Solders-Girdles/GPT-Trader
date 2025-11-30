"""
Momentum signal generator using RSI.
"""

from dataclasses import dataclass

from gpt_trader.features.live_trade.indicators import relative_strength_index
from gpt_trader.features.live_trade.signals.protocol import (
    SignalGenerator,
    StrategyContext,
)
from gpt_trader.features.live_trade.signals.types import SignalOutput, SignalType


@dataclass
class MomentumSignalConfig:
    """Configuration for MomentumSignal."""

    period: int = 14
    overbought: int = 70
    oversold: int = 30


class MomentumSignal(SignalGenerator):
    """Generates signals based on RSI momentum."""

    def __init__(self, config: MomentumSignalConfig) -> None:
        self.config = config

    def generate(self, context: StrategyContext) -> SignalOutput:
        """Generate momentum signal."""
        # Minimum data requirement
        if len(context.recent_marks) < self.config.period + 1:
            return SignalOutput(
                name="momentum_rsi",
                type=SignalType.MEAN_REVERSION,  # RSI is often used as mean reversion
                strength=0.0,
                confidence=0.0,
                metadata={"reason": "insufficient_data"},
            )

        rsi = relative_strength_index(context.recent_marks, self.config.period)

        if rsi is None:
            return SignalOutput(
                name="momentum_rsi",
                type=SignalType.MEAN_REVERSION,
                strength=0.0,
                confidence=0.0,
                metadata={"reason": "calculation_error"},
            )

        strength = 0.0
        confidence = 0.5
        reason = "neutral"

        rsi_val = float(rsi)

        # Logic:
        # RSI < Oversold -> Buy
        # RSI > Overbought -> Sell

        if rsi_val < self.config.oversold:
            strength = 1.0
            # Confidence increases as RSI gets lower
            confidence = 0.5 + (self.config.oversold - rsi_val) / self.config.oversold * 0.5
            reason = "oversold"
        elif rsi_val > self.config.overbought:
            strength = -1.0
            # Confidence increases as RSI gets higher
            confidence = (
                0.5 + (rsi_val - self.config.overbought) / (100 - self.config.overbought) * 0.5
            )
            reason = "overbought"
        else:
            # Neutral zone
            strength = 0.0
            reason = "neutral"

        return SignalOutput(
            name="momentum_rsi",
            type=SignalType.MEAN_REVERSION,
            strength=strength,
            confidence=min(confidence, 1.0),
            metadata={
                "rsi": rsi_val,
                "reason": reason,
            },
        )
