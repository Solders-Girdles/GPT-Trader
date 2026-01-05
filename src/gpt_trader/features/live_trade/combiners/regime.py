"""
Regime-aware signal combiner using ADX.
"""

from dataclasses import dataclass, field
from decimal import Decimal

from gpt_trader.features.live_trade.indicators import average_directional_index
from gpt_trader.features.live_trade.signals.protocol import (
    SignalCombiner,
    StrategyContext,
)
from gpt_trader.features.live_trade.signals.types import SignalOutput, SignalType


@dataclass
class RegimeCombinerConfig:
    """Configuration for RegimeAwareCombiner."""

    adx_period: int = 14
    trending_threshold: int = 25
    ranging_threshold: int = 20

    # Weights for Trending Regime
    trending_weights: dict[SignalType, float] = field(
        default_factory=lambda: {
            SignalType.TREND: 1.0,
            SignalType.MEAN_REVERSION: 0.0,
            SignalType.VOLATILITY: 0.5,
            SignalType.SENTIMENT: 0.5,
            SignalType.ORDER_FLOW: 0.8,  # High weight - confirms trend direction
            SignalType.MICROSTRUCTURE: 0.5,  # Medium weight - execution quality
        }
    )

    # Weights for Ranging Regime
    ranging_weights: dict[SignalType, float] = field(
        default_factory=lambda: {
            SignalType.TREND: 0.0,
            SignalType.MEAN_REVERSION: 1.0,
            SignalType.VOLATILITY: 0.5,
            SignalType.SENTIMENT: 0.5,
            SignalType.ORDER_FLOW: 0.6,  # Still useful for entry timing
            SignalType.MICROSTRUCTURE: 0.7,  # Higher weight - important for mean reversion
        }
    )


class RegimeAwareCombiner(SignalCombiner):
    """Combines signals based on market regime (Trending vs Ranging)."""

    def __init__(self, config: RegimeCombinerConfig) -> None:
        self.config = config
        self._current_regime = "neutral"  # "trending", "ranging", "neutral"

    def combine(self, signals: list[SignalOutput], context: StrategyContext) -> SignalOutput:
        """Combine signals using regime-based weights."""

        # 1. Determine Regime
        adx = self._calculate_adx(context)
        self._update_regime(adx)

        # 2. Select Weights
        if self._current_regime == "trending":
            weights = self.config.trending_weights
        elif self._current_regime == "ranging":
            weights = self.config.ranging_weights
        else:
            # Neutral/Transition: Blend or default?
            # Let's use 50/50 blend for now
            weights = {
                t: (
                    self.config.trending_weights.get(t, 0.0)
                    + self.config.ranging_weights.get(t, 0.0)
                )
                / 2
                for t in SignalType
            }

        # 3. Aggregate Signals
        total_weighted_signal = 0.0
        total_weight = 0.0

        signal_details = {}

        for signal in signals:
            weight = weights.get(signal.type, 0.0)
            # Adjust weight by signal confidence?
            # effective_weight = weight * signal.confidence

            # For now, just use base weight.
            # But we should probably use confidence too.
            effective_weight = weight

            total_weighted_signal += signal.strength * effective_weight
            total_weight += effective_weight

            signal_details[signal.name] = {
                "raw": signal.strength,
                "weight": effective_weight,
                "contribution": signal.strength * effective_weight,
            }

        # 4. Normalize
        if total_weight > 0:
            net_strength = total_weighted_signal / total_weight
        else:
            net_strength = 0.0

        # Calculate net confidence (weighted average)
        if total_weight > 0:
            net_confidence = (
                sum(s.confidence * weights.get(s.type, 0.0) for s in signals) / total_weight
            )
        else:
            net_confidence = 0.0

        return SignalOutput(
            name="ensemble_net",
            type=SignalType.OTHER,
            strength=net_strength,
            confidence=net_confidence,
            metadata={
                "regime": self._current_regime,
                "adx": float(adx) if adx is not None else None,
                "components": signal_details,
            },
        )

    def _calculate_adx(self, context: StrategyContext) -> Decimal | None:
        """Calculate ADX from context candles."""
        if not context.candles or len(context.candles) < self.config.adx_period * 2:
            return None

        # Extract Highs, Lows, Closes
        # Assuming candles have these attributes and are sorted oldest to newest
        try:
            highs = [c.high for c in context.candles]
            lows = [c.low for c in context.candles]
            closes = [c.close for c in context.candles]
            return average_directional_index(highs, lows, closes, self.config.adx_period)
        except AttributeError:
            return None

    def _update_regime(self, adx: Decimal | None) -> None:
        """Update current regime based on ADX with hysteresis."""
        if adx is None:
            # Keep previous or default to neutral
            return

        adx_val = float(adx)

        if self._current_regime == "trending":
            if adx_val < self.config.ranging_threshold:
                self._current_regime = "ranging"
            # Else stay trending
        elif self._current_regime == "ranging":
            if adx_val > self.config.trending_threshold:
                self._current_regime = "trending"
            # Else stay ranging
        else:
            # Neutral/Initial
            if adx_val > self.config.trending_threshold:
                self._current_regime = "trending"
            elif adx_val < self.config.ranging_threshold:
                self._current_regime = "ranging"
