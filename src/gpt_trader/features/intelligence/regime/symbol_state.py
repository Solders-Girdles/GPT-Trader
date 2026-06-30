"""Per-symbol regime state for the market regime detector.

Holds the rolling price/indicator state for a single symbol and computes the
component scores (trend, volatility, momentum, drawdown, ATR/ADX percentiles,
squeeze) that MarketRegimeDetector consumes. Extracted from detector.py so the
stateful per-symbol accumulator is separated from classification orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from gpt_trader.features.intelligence.regime.indicators import (
    OnlineATR,
    OnlineBollingerBands,
    OnlineTrendStrength,
)
from gpt_trader.features.intelligence.regime.models import RegimeConfig, RegimeType
from gpt_trader.features.live_trade.stateful_indicators import (
    OnlineEMA,
    RollingWindow,
    WelfordState,
)


@dataclass
class _SymbolRegimeState:
    """Per-symbol stateful indicators for regime detection.

    All updates are O(1) using incremental algorithms.
    """

    config: RegimeConfig

    # Trend indicators (EMAs)
    short_ema: OnlineEMA = field(init=False)
    long_ema: OnlineEMA = field(init=False)
    ema_slope_window: RollingWindow = field(init=False)  # For slope calculation

    # Volatility tracking
    returns_welford: WelfordState = field(default_factory=WelfordState)
    volatility_history: RollingWindow = field(init=False)  # Historical vol distribution

    # Momentum tracking
    price_window: RollingWindow = field(init=False)

    # Advanced indicators
    atr: OnlineATR = field(init=False)
    bollinger: OnlineBollingerBands = field(init=False)
    trend_strength: OnlineTrendStrength = field(init=False)
    atr_history: RollingWindow = field(init=False)  # For ATR percentile calculation

    # Price tracking
    prev_price: Decimal | None = None
    peak_price: Decimal | None = None  # For drawdown calculation
    tick_count: int = 0

    # Regime persistence
    current_regime: RegimeType = RegimeType.UNKNOWN
    regime_ticks: int = 0
    pending_regime: RegimeType | None = None
    pending_ticks: int = 0

    def __post_init__(self) -> None:
        """Initialize indicators based on config."""
        self.short_ema = OnlineEMA(period=self.config.short_ema_period)
        self.long_ema = OnlineEMA(period=self.config.long_ema_period)
        self.ema_slope_window = RollingWindow(max_size=5)  # 5-tick slope window
        self.volatility_history = RollingWindow(max_size=self.config.volatility_lookback)
        self.price_window = RollingWindow(max_size=self.config.momentum_period)

        # Advanced indicators
        self.atr = OnlineATR(period=self.config.atr_period)
        self.bollinger = OnlineBollingerBands(period=self.config.short_ema_period)
        self.trend_strength = OnlineTrendStrength(period=self.config.atr_period)
        self.atr_history = RollingWindow(max_size=self.config.volatility_lookback)

    def update(self, price: Decimal) -> None:
        """Update all indicators with new price. O(1) complexity."""
        self.tick_count += 1

        # Update EMAs
        self.short_ema.update(price)
        long_val = self.long_ema.update(price)

        # Track EMA slope (for trend direction)
        if long_val is not None:
            self.ema_slope_window.add(long_val)

        # Update volatility (absolute return)
        if self.prev_price is not None and self.prev_price > 0:
            ret = (price - self.prev_price) / self.prev_price
            self.returns_welford.update(ret)

            # Store current volatility reading (using return magnitude)
            abs_ret = abs(ret)
            self.volatility_history.add(abs_ret)

        # Update momentum window
        self.price_window.add(price)

        # Update advanced indicators
        atr_value = self.atr.update(price)
        if atr_value is not None:
            self.atr_history.add(atr_value)
        self.bollinger.update(price)
        self.trend_strength.update(price)

        # Update peak for drawdown calculation
        if self.peak_price is None or price > self.peak_price:
            self.peak_price = price

        self.prev_price = price

    def get_trend_score(self) -> float:
        """Calculate trend score from -1 (bearish) to +1 (bullish).

        Uses:
        1. Price position relative to long EMA
        2. Short EMA vs long EMA
        3. EMA slope direction
        """
        if not self.long_ema.initialized or self.long_ema.value is None:
            return 0.0

        long_ema = float(self.long_ema.value)
        if long_ema == 0:
            return 0.0

        scores: list[float] = []

        # 1. Price vs long EMA (weight: 0.4)
        if self.prev_price is not None:
            price_deviation = (float(self.prev_price) - long_ema) / long_ema
            price_score = max(-1.0, min(1.0, price_deviation / self.config.trend_threshold))
            scores.append(price_score * 0.4)

        # 2. Short EMA vs Long EMA (weight: 0.4)
        if self.short_ema.initialized and self.short_ema.value is not None:
            ema_diff = (float(self.short_ema.value) - long_ema) / long_ema
            ema_score = max(-1.0, min(1.0, ema_diff / self.config.trend_threshold))
            scores.append(ema_score * 0.4)

        # 3. EMA slope (weight: 0.2)
        if len(self.ema_slope_window) >= 2:
            oldest = float(self.ema_slope_window[0])
            newest = float(self.ema_slope_window[-1])
            if oldest > 0:
                slope = (newest - oldest) / oldest
                slope_score = max(-1.0, min(1.0, slope / (self.config.trend_threshold / 5)))
                scores.append(slope_score * 0.2)

        return sum(scores) if scores else 0.0

    def get_volatility_percentile(self) -> float:
        """Calculate current volatility percentile vs historical.

        Returns value from 0.0 (lowest) to 1.0 (highest).
        """
        if self.returns_welford.count < 10:
            return 0.5  # Default to middle

        current_vol = float(self.returns_welford.std_dev)
        if current_vol == 0:
            return 0.5

        # Compare to historical distribution
        if len(self.volatility_history) < 10:
            return 0.5

        # Count how many historical values are below current
        count_below = sum(1 for v in self.volatility_history.values if float(v) < current_vol)

        return count_below / len(self.volatility_history)

    def get_momentum_score(self) -> float:
        """Calculate momentum score from -1 to +1.

        Uses rate of change over momentum window.
        """
        if len(self.price_window) < 2:
            return 0.0

        oldest = float(self.price_window[0])
        newest = float(self.price_window[-1])

        if oldest == 0:
            return 0.0

        roc = (newest - oldest) / oldest

        # Normalize to -1 to +1 range (assuming 5% max expected change)
        return max(-1.0, min(1.0, roc / 0.05))

    def get_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if self.peak_price is None or self.prev_price is None:
            return 0.0
        if self.peak_price == 0:
            return 0.0

        return float((self.peak_price - self.prev_price) / self.peak_price)

    def get_atr_percentile(self) -> float:
        """Get current ATR percentile vs historical.

        Returns value from 0.0 (lowest vol) to 1.0 (highest vol).
        More robust than return-based volatility for regime detection.
        """
        if not self.atr.is_ready or self.atr.value is None:
            return 0.5  # Default to middle

        current_atr = float(self.atr.value)
        if len(self.atr_history) < 10:
            return 0.5

        # Count how many historical ATR values are below current
        count_below = sum(1 for v in self.atr_history.values if float(v) < current_atr)
        return count_below / len(self.atr_history)

    def get_adx(self) -> float | None:
        """Get current ADX (trend strength) value.

        Returns 0-100 scale:
        - 0-25: Weak/no trend (ranging)
        - 25-50: Trending
        - 50-75: Strong trend
        - 75-100: Very strong trend
        """
        if not self.trend_strength.is_ready:
            return None

        result = self.trend_strength.update(self.prev_price or Decimal("0"))
        return result.get("adx")

    def get_squeeze_score(self) -> float:
        """Get Bollinger Band squeeze score.

        Returns 0.0-1.0:
        - High values indicate consolidation (squeeze)
        - Low values indicate expansion
        - Squeeze often precedes breakouts
        """
        if not self.bollinger.is_ready:
            return 0.5

        return self.bollinger.get_squeeze_score()

    def is_warmed_up(self) -> bool:
        """Check if enough data for valid regime detection."""
        return (
            self.long_ema.initialized
            and len(self.volatility_history) >= 20
            and self.tick_count >= self.config.long_ema_period
        )

    def serialize(self) -> dict[str, Any]:
        """Serialize state for persistence."""
        return {
            "short_ema": self.short_ema.serialize(),
            "long_ema": self.long_ema.serialize(),
            "ema_slope_window": self.ema_slope_window.serialize(),
            "returns_welford": self.returns_welford.serialize(),
            "volatility_history": self.volatility_history.serialize(),
            "price_window": self.price_window.serialize(),
            # Advanced indicators
            "atr": self.atr.serialize(),
            "bollinger": self.bollinger.serialize(),
            "trend_strength": self.trend_strength.serialize(),
            "atr_history": self.atr_history.serialize(),
            # Price tracking
            "prev_price": str(self.prev_price) if self.prev_price else None,
            "peak_price": str(self.peak_price) if self.peak_price else None,
            "tick_count": self.tick_count,
            "current_regime": self.current_regime.name,
            "regime_ticks": self.regime_ticks,
            "pending_regime": self.pending_regime.name if self.pending_regime else None,
            "pending_ticks": self.pending_ticks,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any], config: RegimeConfig) -> _SymbolRegimeState:
        """Restore state from serialized data."""
        state = cls(config=config)

        state.short_ema = OnlineEMA.deserialize(data["short_ema"])
        state.long_ema = OnlineEMA.deserialize(data["long_ema"])
        state.ema_slope_window = RollingWindow.deserialize(data["ema_slope_window"])
        state.returns_welford = WelfordState.deserialize(data["returns_welford"])
        state.volatility_history = RollingWindow.deserialize(data["volatility_history"])
        state.price_window = RollingWindow.deserialize(data["price_window"])

        # Restore advanced indicators (with backward compatibility)
        if "atr" in data:
            state.atr = OnlineATR.deserialize(data["atr"])
        if "bollinger" in data:
            state.bollinger = OnlineBollingerBands.deserialize(data["bollinger"])
        if "trend_strength" in data:
            state.trend_strength = OnlineTrendStrength.deserialize(data["trend_strength"])
        if "atr_history" in data:
            state.atr_history = RollingWindow.deserialize(data["atr_history"])

        state.prev_price = Decimal(data["prev_price"]) if data["prev_price"] else None
        state.peak_price = Decimal(data["peak_price"]) if data["peak_price"] else None
        state.tick_count = data["tick_count"]
        state.current_regime = RegimeType[data["current_regime"]]
        state.regime_ticks = data["regime_ticks"]
        state.pending_regime = (
            RegimeType[data["pending_regime"]] if data["pending_regime"] else None
        )
        state.pending_ticks = data["pending_ticks"]

        return state
