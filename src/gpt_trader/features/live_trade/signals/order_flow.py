"""
Order Flow signal generator using trade aggressor analysis.

Uses trade tape data to identify buying/selling pressure based on
which side is more aggressive (taking liquidity).
"""

from dataclasses import dataclass

from gpt_trader.features.live_trade.signals.protocol import (
    SignalGenerator,
    StrategyContext,
)
from gpt_trader.features.live_trade.signals.types import SignalOutput, SignalType


@dataclass
class OrderFlowSignalConfig:
    """Configuration for OrderFlowSignal.

    Attributes:
        aggressor_threshold_bullish: Aggressor ratio above this is bullish (default 0.6 = 60%).
        aggressor_threshold_bearish: Aggressor ratio below this is bearish (default 0.4 = 40%).
        min_trades: Minimum trades required for valid signal.
        volume_weight: Whether to weight by volume (vs trade count).
    """

    aggressor_threshold_bullish: float = 0.6
    aggressor_threshold_bearish: float = 0.4
    min_trades: int = 10
    volume_weight: bool = True


class OrderFlowSignal(SignalGenerator):
    """Generates signals based on order flow analysis.

    Analyzes the aggressor ratio from recent trades:
    - High buy aggression (buyers lifting offers) → Bullish
    - High sell aggression (sellers hitting bids) → Bearish

    This signal confirms trend direction with actual order flow.
    """

    def __init__(self, config: OrderFlowSignalConfig | None = None) -> None:
        self.config = config or OrderFlowSignalConfig()

    def generate(self, context: StrategyContext) -> SignalOutput:
        """Generate order flow signal based on aggressor analysis."""
        # Check for market data availability
        if context.market_data is None:
            return self._no_data_signal("no_market_data")

        trade_stats = context.market_data.trade_volume_stats
        if trade_stats is None:
            return self._no_data_signal("no_trade_stats")

        trade_count = trade_stats.get("count", 0)
        if trade_count < self.config.min_trades:
            return self._no_data_signal(
                "insufficient_trades",
                metadata={"trade_count": trade_count, "min_required": self.config.min_trades},
            )

        aggressor_ratio = trade_stats.get("aggressor_ratio")
        if aggressor_ratio is None:
            return self._no_data_signal("no_aggressor_ratio")

        # Calculate signal strength and confidence
        strength = 0.0
        confidence = 0.5
        reason = "neutral"

        if aggressor_ratio > self.config.aggressor_threshold_bullish:
            # Strong buy aggression - bullish
            strength = self._calculate_strength(
                aggressor_ratio, self.config.aggressor_threshold_bullish, 1.0
            )
            confidence = self._calculate_confidence(trade_count, aggressor_ratio, bullish=True)
            reason = "buy_aggression"
        elif aggressor_ratio < self.config.aggressor_threshold_bearish:
            # Strong sell aggression - bearish
            strength = -self._calculate_strength(
                1.0 - aggressor_ratio, 1.0 - self.config.aggressor_threshold_bearish, 1.0
            )
            confidence = self._calculate_confidence(trade_count, aggressor_ratio, bullish=False)
            reason = "sell_aggression"
        else:
            # Neutral zone - balanced order flow
            strength = 0.0
            confidence = 0.3
            reason = "balanced_flow"

        # Include volume metrics if available
        volume = trade_stats.get("volume")
        vwap = trade_stats.get("vwap")

        return SignalOutput(
            name="order_flow",
            type=SignalType.ORDER_FLOW,
            strength=strength,
            confidence=min(confidence, 1.0),
            metadata={
                "aggressor_ratio": aggressor_ratio,
                "trade_count": trade_count,
                "volume": str(volume) if volume else None,
                "vwap": str(vwap) if vwap else None,
                "reason": reason,
            },
        )

    def _no_data_signal(self, reason: str, metadata: dict | None = None) -> SignalOutput:
        """Return a neutral signal when data is unavailable."""
        meta = {"reason": reason}
        if metadata:
            meta.update(metadata)
        return SignalOutput(
            name="order_flow",
            type=SignalType.ORDER_FLOW,
            strength=0.0,
            confidence=0.0,
            metadata=meta,
        )

    def _calculate_strength(self, value: float, threshold: float, max_value: float) -> float:
        """Calculate signal strength based on how far value exceeds threshold.

        Scales linearly from 0 (at threshold) to 1 (at max_value).
        """
        if value <= threshold:
            return 0.0
        range_size = max_value - threshold
        if range_size <= 0:
            return 1.0
        return min((value - threshold) / range_size, 1.0)

    def _calculate_confidence(
        self, trade_count: int, aggressor_ratio: float, bullish: bool
    ) -> float:
        """Calculate confidence based on trade count and signal clarity.

        More trades and more extreme ratios increase confidence.
        """
        # Base confidence from trade count (more trades = more reliable)
        count_factor = min(trade_count / 50, 1.0) * 0.3  # Max 0.3 from count

        # Confidence from signal clarity (more extreme ratio = clearer signal)
        if bullish:
            clarity = (aggressor_ratio - 0.5) * 2  # 0.5->0, 1.0->1
        else:
            clarity = (0.5 - aggressor_ratio) * 2  # 0.5->0, 0.0->1

        clarity_factor = max(0, clarity) * 0.5  # Max 0.5 from clarity

        # Base confidence
        base = 0.2

        return base + count_factor + clarity_factor
