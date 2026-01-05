"""
Orderbook Imbalance signal generator using depth analysis.

Analyzes the balance between bid and ask depth to identify
supply/demand imbalances that may predict price movement.
"""

from dataclasses import dataclass
from decimal import Decimal

from gpt_trader.features.live_trade.signals.protocol import (
    SignalGenerator,
    StrategyContext,
)
from gpt_trader.features.live_trade.signals.types import SignalOutput, SignalType


@dataclass
class OrderbookImbalanceSignalConfig:
    """Configuration for OrderbookImbalanceSignal.

    Attributes:
        levels: Number of orderbook levels to analyze.
        imbalance_threshold: Minimum imbalance for signal generation.
        strong_imbalance_threshold: Threshold for high-confidence signals.
    """

    levels: int = 5
    imbalance_threshold: float = 0.2
    strong_imbalance_threshold: float = 0.5


class OrderbookImbalanceSignal(SignalGenerator):
    """Generates signals based on orderbook depth imbalance.

    Calculates the ratio of bid depth to ask depth:
    - More bid depth than ask depth → Bullish (buying support)
    - More ask depth than bid depth → Bearish (selling pressure)

    Imbalance formula: (bid_depth - ask_depth) / (bid_depth + ask_depth)
    Range: -1.0 (all ask) to +1.0 (all bid)
    """

    def __init__(self, config: OrderbookImbalanceSignalConfig | None = None) -> None:
        self.config = config or OrderbookImbalanceSignalConfig()

    def generate(self, context: StrategyContext) -> SignalOutput:
        """Generate orderbook imbalance signal."""
        # Check for market data availability
        if context.market_data is None:
            return self._no_data_signal("no_market_data")

        orderbook = context.market_data.orderbook_snapshot
        if orderbook is None:
            return self._no_data_signal("no_orderbook")

        # Get depth at configured levels
        try:
            bid_depth, ask_depth = orderbook.get_depth(self.config.levels)
        except Exception:
            return self._no_data_signal("depth_calculation_error")

        # Validate depths
        if bid_depth is None or ask_depth is None:
            return self._no_data_signal("invalid_depth")

        bid_depth_float = float(bid_depth) if isinstance(bid_depth, Decimal) else bid_depth
        ask_depth_float = float(ask_depth) if isinstance(ask_depth, Decimal) else ask_depth

        total_depth = bid_depth_float + ask_depth_float
        if total_depth <= 0:
            return self._no_data_signal("zero_depth")

        # Calculate imbalance: -1 (all ask) to +1 (all bid)
        imbalance = (bid_depth_float - ask_depth_float) / total_depth

        # Calculate signal strength and confidence
        strength = 0.0
        confidence = 0.3  # Base confidence for microstructure signals
        reason = "balanced"

        abs_imbalance = abs(imbalance)

        if abs_imbalance >= self.config.imbalance_threshold:
            # Significant imbalance detected
            if imbalance > 0:
                # More bid depth - bullish
                strength = self._scale_strength(imbalance)
                reason = "bid_heavy"
            else:
                # More ask depth - bearish
                strength = self._scale_strength(imbalance)
                reason = "ask_heavy"

            # Higher confidence for stronger imbalances
            if abs_imbalance >= self.config.strong_imbalance_threshold:
                confidence = 0.7
            else:
                # Scale between 0.3 and 0.7 based on imbalance
                confidence = 0.3 + (
                    (abs_imbalance - self.config.imbalance_threshold)
                    / (self.config.strong_imbalance_threshold - self.config.imbalance_threshold)
                ) * 0.4

        # Include spread as context (wide spread reduces reliability)
        spread_bps = None
        if orderbook.spread_bps is not None:
            spread_bps = float(orderbook.spread_bps)
            # Wide spreads reduce confidence
            if spread_bps > 20:
                confidence *= 0.7
            elif spread_bps > 10:
                confidence *= 0.85

        return SignalOutput(
            name="orderbook_imbalance",
            type=SignalType.MICROSTRUCTURE,
            strength=strength,
            confidence=min(confidence, 1.0),
            metadata={
                "imbalance": imbalance,
                "bid_depth": bid_depth_float,
                "ask_depth": ask_depth_float,
                "levels_analyzed": self.config.levels,
                "spread_bps": spread_bps,
                "reason": reason,
            },
        )

    def _no_data_signal(self, reason: str) -> SignalOutput:
        """Return a neutral signal when data is unavailable."""
        return SignalOutput(
            name="orderbook_imbalance",
            type=SignalType.MICROSTRUCTURE,
            strength=0.0,
            confidence=0.0,
            metadata={"reason": reason},
        )

    def _scale_strength(self, imbalance: float) -> float:
        """Scale imbalance to signal strength.

        The imbalance is already in [-1, 1] range, but we apply
        a threshold-based scaling to filter noise.
        """
        # Direct pass-through since imbalance is already normalized
        return max(-1.0, min(1.0, imbalance))
