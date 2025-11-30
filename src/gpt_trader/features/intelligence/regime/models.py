"""
Regime detection data models.

Defines the core types for market regime classification:
- RegimeType: Enum of possible market regimes
- RegimeState: Current regime with confidence and metrics
- RegimeConfig: Configuration parameters for detection
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class RegimeType(Enum):
    """Market regime classifications.

    Regimes are defined by two dimensions:
    - Trend: BULL (uptrend), BEAR (downtrend), SIDEWAYS (range-bound)
    - Volatility: QUIET (low volatility), VOLATILE (high volatility)

    Special regimes:
    - CRISIS: Extreme conditions requiring reduced exposure
    - UNKNOWN: Insufficient data for classification
    """

    BULL_QUIET = auto()  # Uptrend, low volatility
    BULL_VOLATILE = auto()  # Uptrend, high volatility
    BEAR_QUIET = auto()  # Downtrend, low volatility
    BEAR_VOLATILE = auto()  # Downtrend, high volatility
    SIDEWAYS_QUIET = auto()  # Range-bound, low volatility
    SIDEWAYS_VOLATILE = auto()  # Range-bound, high volatility
    CRISIS = auto()  # Extreme conditions
    UNKNOWN = auto()  # Insufficient data


@dataclass
class RegimeState:
    """Current regime detection state with confidence and metrics.

    Attributes:
        regime: Detected market regime
        confidence: Confidence in regime classification (0.0 to 1.0)
        trend_score: Trend direction score (-1 bearish to +1 bullish)
        volatility_percentile: Current volatility vs historical (0.0 to 1.0)
        momentum_score: Price momentum (-1 to +1)
        regime_age_ticks: Number of ticks in current regime
        transition_probability: Likelihood of regime change (0.0 to 1.0)
    """

    regime: RegimeType
    confidence: float
    trend_score: float
    volatility_percentile: float
    momentum_score: float
    regime_age_ticks: int = 0
    transition_probability: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization/logging."""
        return {
            "regime": self.regime.name,
            "confidence": round(self.confidence, 4),
            "trend_score": round(self.trend_score, 4),
            "volatility_percentile": round(self.volatility_percentile, 4),
            "momentum_score": round(self.momentum_score, 4),
            "regime_age_ticks": self.regime_age_ticks,
            "transition_probability": round(self.transition_probability, 4),
        }

    @classmethod
    def unknown(cls) -> RegimeState:
        """Create an unknown regime state (insufficient data)."""
        return cls(
            regime=RegimeType.UNKNOWN,
            confidence=0.0,
            trend_score=0.0,
            volatility_percentile=0.5,
            momentum_score=0.0,
            regime_age_ticks=0,
            transition_probability=0.0,
        )

    def is_bullish(self) -> bool:
        """Check if regime is bullish (BULL_QUIET or BULL_VOLATILE)."""
        return self.regime in (RegimeType.BULL_QUIET, RegimeType.BULL_VOLATILE)

    def is_bearish(self) -> bool:
        """Check if regime is bearish (BEAR_QUIET or BEAR_VOLATILE)."""
        return self.regime in (RegimeType.BEAR_QUIET, RegimeType.BEAR_VOLATILE)

    def is_volatile(self) -> bool:
        """Check if regime is volatile."""
        return self.regime in (
            RegimeType.BULL_VOLATILE,
            RegimeType.BEAR_VOLATILE,
            RegimeType.SIDEWAYS_VOLATILE,
            RegimeType.CRISIS,
        )

    def is_crisis(self) -> bool:
        """Check if regime is crisis mode."""
        return self.regime == RegimeType.CRISIS


@dataclass
class RegimeConfig:
    """Configuration for market regime detection.

    All parameters are tunable via YAML profiles.
    """

    # EMA periods for trend detection
    short_ema_period: int = 20
    long_ema_period: int = 50

    # Volatility calculation
    atr_period: int = 14
    volatility_lookback: int = 100  # Historical volatility distribution size

    # Momentum calculation
    momentum_period: int = 10

    # Classification thresholds
    trend_threshold: float = 0.02  # EMA slope threshold for trend vs sideways
    volatility_high_percentile: float = 0.70  # Above this = volatile
    volatility_low_percentile: float = 0.30  # Below this = quiet

    # Crisis detection
    crisis_volatility_multiplier: float = 3.0  # Vol > 3x normal = crisis
    crisis_drawdown_threshold: float = 0.10  # 10% drawdown = crisis

    # Regime persistence (smoothing)
    min_regime_ticks: int = 5  # Minimum ticks before regime change
    transition_smoothing: float = 0.7  # EMA factor for regime confidence

    # Crisis behavior
    crisis_position_multiplier: float = 0.2  # Scale down to 20% in crisis

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "short_ema_period": self.short_ema_period,
            "long_ema_period": self.long_ema_period,
            "atr_period": self.atr_period,
            "volatility_lookback": self.volatility_lookback,
            "momentum_period": self.momentum_period,
            "trend_threshold": self.trend_threshold,
            "volatility_high_percentile": self.volatility_high_percentile,
            "volatility_low_percentile": self.volatility_low_percentile,
            "crisis_volatility_multiplier": self.crisis_volatility_multiplier,
            "crisis_drawdown_threshold": self.crisis_drawdown_threshold,
            "min_regime_ticks": self.min_regime_ticks,
            "transition_smoothing": self.transition_smoothing,
            "crisis_position_multiplier": self.crisis_position_multiplier,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegimeConfig:
        """Create config from dictionary."""
        return cls(
            short_ema_period=data.get("short_ema_period", 20),
            long_ema_period=data.get("long_ema_period", 50),
            atr_period=data.get("atr_period", 14),
            volatility_lookback=data.get("volatility_lookback", 100),
            momentum_period=data.get("momentum_period", 10),
            trend_threshold=data.get("trend_threshold", 0.02),
            volatility_high_percentile=data.get("volatility_high_percentile", 0.70),
            volatility_low_percentile=data.get("volatility_low_percentile", 0.30),
            crisis_volatility_multiplier=data.get("crisis_volatility_multiplier", 3.0),
            crisis_drawdown_threshold=data.get("crisis_drawdown_threshold", 0.10),
            min_regime_ticks=data.get("min_regime_ticks", 5),
            transition_smoothing=data.get("transition_smoothing", 0.7),
            crisis_position_multiplier=data.get("crisis_position_multiplier", 0.2),
        )


__all__ = [
    "RegimeConfig",
    "RegimeState",
    "RegimeType",
]
