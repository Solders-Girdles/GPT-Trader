"""
Spread signal generator for market quality assessment.

Uses bid-ask spread to assess market liquidity and modify
confidence levels for other signals.
"""

from dataclasses import dataclass

from gpt_trader.features.live_trade.signals.protocol import (
    SignalGenerator,
    StrategyContext,
)
from gpt_trader.features.live_trade.signals.types import SignalOutput, SignalType


@dataclass
class SpreadSignalConfig:
    """Configuration for SpreadSignal.

    Attributes:
        tight_spread_bps: Spread below this is considered tight/liquid.
        normal_spread_bps: Spread between tight and normal is acceptable.
        wide_spread_bps: Spread above this indicates poor liquidity.
    """

    tight_spread_bps: float = 5.0
    normal_spread_bps: float = 15.0
    wide_spread_bps: float = 30.0


class SpreadSignal(SignalGenerator):
    """Generates market quality signals based on bid-ask spread.

    This signal does not predict direction but indicates market quality:
    - Tight spread → High confidence in other signals, good execution
    - Normal spread → Standard confidence
    - Wide spread → Low confidence, poor execution expected

    The signal strength is always 0 (neutral) since spread doesn't predict
    direction. The confidence value indicates market quality.
    """

    def __init__(self, config: SpreadSignalConfig | None = None) -> None:
        self.config = config or SpreadSignalConfig()

    def generate(self, context: StrategyContext) -> SignalOutput:
        """Generate spread-based market quality signal."""
        # Check for market data availability
        if context.market_data is None:
            return self._no_data_signal("no_market_data")

        spread_bps = context.market_data.spread_bps
        if spread_bps is None:
            # Try to get from orderbook snapshot
            orderbook = context.market_data.orderbook_snapshot
            if orderbook is not None and orderbook.spread_bps is not None:
                spread_bps = orderbook.spread_bps
            else:
                return self._no_data_signal("no_spread_data")

        spread_float = float(spread_bps)

        # Calculate confidence (market quality indicator)
        confidence = self._calculate_confidence(spread_float)
        quality = self._assess_quality(spread_float)

        return SignalOutput(
            name="spread_quality",
            type=SignalType.MICROSTRUCTURE,
            strength=0.0,  # Spread doesn't predict direction
            confidence=confidence,
            metadata={
                "spread_bps": spread_float,
                "quality": quality,
                "tight_threshold": self.config.tight_spread_bps,
                "wide_threshold": self.config.wide_spread_bps,
            },
        )

    def _no_data_signal(self, reason: str) -> SignalOutput:
        """Return a signal when data is unavailable."""
        return SignalOutput(
            name="spread_quality",
            type=SignalType.MICROSTRUCTURE,
            strength=0.0,
            confidence=0.0,
            metadata={"reason": reason},
        )

    def _calculate_confidence(self, spread_bps: float) -> float:
        """Calculate confidence based on spread.

        Maps spread to a 0-1 confidence score:
        - Tight spread (< tight_threshold): High confidence (0.8-1.0)
        - Normal spread: Medium confidence (0.5-0.8)
        - Wide spread (> wide_threshold): Low confidence (0.1-0.5)
        """
        if spread_bps <= self.config.tight_spread_bps:
            # Tight spread - high confidence
            return 0.8 + (1 - spread_bps / self.config.tight_spread_bps) * 0.2

        if spread_bps <= self.config.normal_spread_bps:
            # Normal spread - scale from 0.8 down to 0.5
            progress = (spread_bps - self.config.tight_spread_bps) / (
                self.config.normal_spread_bps - self.config.tight_spread_bps
            )
            return 0.8 - progress * 0.3

        if spread_bps <= self.config.wide_spread_bps:
            # Wide spread - scale from 0.5 down to 0.2
            progress = (spread_bps - self.config.normal_spread_bps) / (
                self.config.wide_spread_bps - self.config.normal_spread_bps
            )
            return 0.5 - progress * 0.3

        # Very wide spread - low confidence
        return max(0.1, 0.2 - (spread_bps - self.config.wide_spread_bps) / 100)

    def _assess_quality(self, spread_bps: float) -> str:
        """Assess market quality category."""
        if spread_bps <= self.config.tight_spread_bps:
            return "tight"
        if spread_bps <= self.config.normal_spread_bps:
            return "normal"
        if spread_bps <= self.config.wide_spread_bps:
            return "wide"
        return "very_wide"
