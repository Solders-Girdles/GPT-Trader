"""
VWAP signal generator for mean reversion analysis.

Uses Volume Weighted Average Price to identify when the current
price deviates significantly from the recent trading average.
"""

from dataclasses import dataclass
from decimal import Decimal

from gpt_trader.features.live_trade.signals.protocol import (
    SignalGenerator,
    StrategyContext,
)
from gpt_trader.features.live_trade.signals.types import SignalOutput, SignalType


@dataclass
class VWAPSignalConfig:
    """Configuration for VWAPSignal.

    Attributes:
        deviation_threshold: Minimum % deviation from VWAP for signal (0.01 = 1%).
        strong_deviation_threshold: Threshold for high-confidence signals.
        min_trades: Minimum trades for VWAP to be considered reliable.
    """

    deviation_threshold: float = 0.01
    strong_deviation_threshold: float = 0.025
    min_trades: int = 20


class VWAPSignal(SignalGenerator):
    """Generates mean reversion signals based on VWAP deviation.

    VWAP represents the average price weighted by volume. Deviations from
    VWAP often revert:
    - Price below VWAP → Bullish (undervalued relative to volume)
    - Price above VWAP → Bearish (overvalued relative to volume)

    This is a classic institutional mean reversion signal.
    """

    def __init__(self, config: VWAPSignalConfig | None = None) -> None:
        self.config = config or VWAPSignalConfig()

    def generate(self, context: StrategyContext) -> SignalOutput:
        """Generate VWAP deviation signal."""
        # Check for market data availability
        if context.market_data is None:
            return self._no_data_signal("no_market_data")

        trade_stats = context.market_data.trade_volume_stats
        if trade_stats is None:
            return self._no_data_signal("no_trade_stats")

        vwap = trade_stats.get("vwap")
        if vwap is None or vwap == 0:
            return self._no_data_signal("no_vwap")

        trade_count = trade_stats.get("count", 0)
        if trade_count < self.config.min_trades:
            return self._no_data_signal(
                "insufficient_trades",
                metadata={"trade_count": trade_count, "min_required": self.config.min_trades},
            )

        # Get current price
        current_price = float(context.current_mark)
        vwap_float = float(vwap) if isinstance(vwap, Decimal) else vwap

        if vwap_float <= 0:
            return self._no_data_signal("invalid_vwap")

        # Calculate deviation from VWAP
        deviation = (current_price - vwap_float) / vwap_float
        abs_deviation = abs(deviation)

        # Calculate signal strength and confidence
        strength = 0.0
        confidence = 0.3  # Base confidence
        reason = "near_vwap"

        if abs_deviation >= self.config.deviation_threshold:
            if deviation < 0:
                # Price below VWAP - bullish (mean reversion buy)
                strength = self._calculate_strength(abs_deviation)
                reason = "below_vwap"
            else:
                # Price above VWAP - bearish (mean reversion sell)
                strength = -self._calculate_strength(abs_deviation)
                reason = "above_vwap"

            # Calculate confidence based on deviation magnitude and trade count
            confidence = self._calculate_confidence(abs_deviation, trade_count)

        # Get additional context from trade stats
        volume = trade_stats.get("volume")
        avg_size = trade_stats.get("avg_size")

        return SignalOutput(
            name="vwap_deviation",
            type=SignalType.MEAN_REVERSION,
            strength=strength,
            confidence=min(confidence, 1.0),
            metadata={
                "vwap": vwap_float,
                "current_price": current_price,
                "deviation_pct": deviation * 100,
                "trade_count": trade_count,
                "volume": str(volume) if volume else None,
                "avg_size": str(avg_size) if avg_size else None,
                "reason": reason,
            },
        )

    def _no_data_signal(self, reason: str, metadata: dict | None = None) -> SignalOutput:
        """Return a neutral signal when data is unavailable."""
        meta = {"reason": reason}
        if metadata:
            meta.update(metadata)
        return SignalOutput(
            name="vwap_deviation",
            type=SignalType.MEAN_REVERSION,
            strength=0.0,
            confidence=0.0,
            metadata=meta,
        )

    def _calculate_strength(self, abs_deviation: float) -> float:
        """Calculate signal strength from deviation magnitude.

        Scales from ~0.1 (at threshold) to 1 (at 2x strong threshold).
        """
        max_deviation = self.config.strong_deviation_threshold * 2

        if abs_deviation < self.config.deviation_threshold:
            return 0.0

        # Linear scale from threshold to max
        # Start at 0.1 when at threshold, scale up to 1.0
        base_strength = 0.1
        range_strength = 0.9
        scale = min(
            (abs_deviation - self.config.deviation_threshold)
            / (max_deviation - self.config.deviation_threshold),
            1.0,
        )
        return base_strength + scale * range_strength

    def _calculate_confidence(self, abs_deviation: float, trade_count: int) -> float:
        """Calculate confidence based on deviation and data quality.

        Higher confidence for:
        - Larger deviations (clearer signal)
        - More trades (more reliable VWAP)
        """
        # Base from deviation magnitude
        if abs_deviation >= self.config.strong_deviation_threshold:
            deviation_conf = 0.5
        else:
            # Scale between 0.2 and 0.5
            progress = (abs_deviation - self.config.deviation_threshold) / (
                self.config.strong_deviation_threshold - self.config.deviation_threshold
            )
            deviation_conf = 0.2 + progress * 0.3

        # Contribution from trade count (more trades = more reliable)
        count_conf = min(trade_count / 100, 1.0) * 0.3

        # Base confidence
        base = 0.2

        return base + deviation_conf + count_conf
