"""
Market regime detection for strategy validation.

Identifies different market regimes (trend, range, high-vol) to validate
strategies across varied conditions.
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

import pandas as pd

from bot_v2.features.live_trade.strategies.indicators import calculate_adx, calculate_atr
from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="regime_detection")


class MarketRegime(Enum):
    """Market regime types."""

    TREND = "trend"
    RANGE = "range"
    HIGH_VOL = "high_vol"
    UNKNOWN = "unknown"


@dataclass
class RegimeMetrics:
    """Metrics characterizing a market regime."""

    regime: MarketRegime
    start_idx: int
    end_idx: int
    avg_adx: float
    avg_atr: float
    avg_return: float
    volatility: float
    bars_count: int


class RegimeDetector:
    """
    Detects market regimes in historical data.

    Regimes:
    - TREND: Strong directional movement (ADX > 25, consistent returns)
    - RANGE: Sideways/choppy (ADX < 20, low returns, high reversals)
    - HIGH_VOL: High volatility regardless of direction (ATR > 1.5x average)
    """

    def __init__(
        self,
        *,
        adx_trend_threshold: float = 25.0,
        adx_range_threshold: float = 20.0,
        atr_volatility_multiplier: float = 1.5,
        adx_period: int = 14,
        atr_period: int = 14,
        min_regime_bars: int = 20,  # Minimum bars for a regime
    ):
        """
        Initialize regime detector.

        Args:
            adx_trend_threshold: ADX threshold for trending (default 25)
            adx_range_threshold: ADX threshold for ranging (default 20)
            atr_volatility_multiplier: ATR multiplier for high-vol (default 1.5)
            adx_period: ADX calculation period
            atr_period: ATR calculation period
            min_regime_bars: Minimum bars to classify a regime
        """
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_range_threshold = adx_range_threshold
        self.atr_volatility_multiplier = atr_volatility_multiplier
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.min_regime_bars = min_regime_bars

    def detect_regimes(self, data: pd.DataFrame) -> list[RegimeMetrics]:
        """
        Detect regimes in historical data.

        Args:
            data: DataFrame with columns: open, high, low, close

        Returns:
            List of RegimeMetrics for each detected regime
        """
        if len(data) < max(self.adx_period, self.atr_period) + self.min_regime_bars:
            logger.warning("Insufficient data for regime detection")
            return []

        # Calculate indicators
        adx_values = self._calculate_adx_series(data)
        atr_values = self._calculate_atr_series(data)

        # Calculate rolling average ATR for volatility threshold
        avg_atr = sum(atr_values) / len(atr_values) if atr_values else 0.0

        # Classify each bar
        regime_labels: list[MarketRegime] = []

        for i in range(len(data)):
            if i < max(self.adx_period, self.atr_period):
                regime_labels.append(MarketRegime.UNKNOWN)
                continue

            adx = adx_values[i] if i < len(adx_values) else None
            atr = atr_values[i] if i < len(atr_values) else None

            # High volatility takes precedence
            if atr and avg_atr > 0 and atr > avg_atr * self.atr_volatility_multiplier:
                regime = MarketRegime.HIGH_VOL
            # Trending market
            elif adx and adx > self.adx_trend_threshold:
                regime = MarketRegime.TREND
            # Ranging market
            elif adx and adx < self.adx_range_threshold:
                regime = MarketRegime.RANGE
            else:
                regime = MarketRegime.UNKNOWN

            regime_labels.append(regime)

        # Group consecutive regimes
        regimes = self._group_regimes(data, regime_labels, adx_values, atr_values)

        logger.info(
            "Regime detection complete | regimes=%d | trend=%d | range=%d | high_vol=%d",
            len(regimes),
            sum(1 for r in regimes if r.regime == MarketRegime.TREND),
            sum(1 for r in regimes if r.regime == MarketRegime.RANGE),
            sum(1 for r in regimes if r.regime == MarketRegime.HIGH_VOL),
        )

        return regimes

    def _calculate_adx_series(self, data: pd.DataFrame) -> list[float]:
        """Calculate ADX for entire dataset."""
        adx_values: list[float] = []

        for i in range(len(data)):
            if i < self.adx_period + 1:
                adx_values.append(0.0)
                continue

            highs = [Decimal(str(h)) for h in data["high"].iloc[:i + 1]]
            lows = [Decimal(str(l)) for l in data["low"].iloc[:i + 1]]
            closes = [Decimal(str(c)) for c in data["close"].iloc[:i + 1]]

            adx = calculate_adx(highs, lows, closes, period=self.adx_period)
            adx_values.append(float(adx) if adx else 0.0)

        return adx_values

    def _calculate_atr_series(self, data: pd.DataFrame) -> list[float]:
        """Calculate ATR for entire dataset."""
        atr_values: list[float] = []

        for i in range(len(data)):
            if i < self.atr_period + 1:
                atr_values.append(0.0)
                continue

            highs = [Decimal(str(h)) for h in data["high"].iloc[:i + 1]]
            lows = [Decimal(str(l)) for l in data["low"].iloc[:i + 1]]
            closes = [Decimal(str(c)) for c in data["close"].iloc[:i + 1]]

            atr = calculate_atr(highs, lows, closes, period=self.atr_period)
            atr_values.append(float(atr) if atr else 0.0)

        return atr_values

    def _group_regimes(
        self,
        data: pd.DataFrame,
        regime_labels: list[MarketRegime],
        adx_values: list[float],
        atr_values: list[float],
    ) -> list[RegimeMetrics]:
        """Group consecutive bars into regime periods."""
        regimes: list[RegimeMetrics] = []
        current_regime = None
        start_idx = 0

        for i, regime in enumerate(regime_labels):
            if regime == MarketRegime.UNKNOWN:
                continue

            if regime != current_regime:
                # End previous regime if long enough
                if current_regime and (i - start_idx) >= self.min_regime_bars:
                    metrics = self._calculate_regime_metrics(
                        data, current_regime, start_idx, i, adx_values, atr_values
                    )
                    regimes.append(metrics)

                # Start new regime
                current_regime = regime
                start_idx = i

        # Handle final regime
        if current_regime and (len(regime_labels) - start_idx) >= self.min_regime_bars:
            metrics = self._calculate_regime_metrics(
                data, current_regime, start_idx, len(regime_labels), adx_values, atr_values
            )
            regimes.append(metrics)

        return regimes

    def _calculate_regime_metrics(
        self,
        data: pd.DataFrame,
        regime: MarketRegime,
        start_idx: int,
        end_idx: int,
        adx_values: list[float],
        atr_values: list[float],
    ) -> RegimeMetrics:
        """Calculate metrics for a regime period."""
        # Calculate average ADX and ATR
        avg_adx = sum(adx_values[start_idx:end_idx]) / (end_idx - start_idx)
        avg_atr = sum(atr_values[start_idx:end_idx]) / (end_idx - start_idx)

        # Calculate returns
        period_data = data.iloc[start_idx:end_idx]
        returns = period_data["close"].pct_change().dropna()
        avg_return = float(returns.mean()) if len(returns) > 0 else 0.0
        volatility = float(returns.std()) if len(returns) > 0 else 0.0

        return RegimeMetrics(
            regime=regime,
            start_idx=start_idx,
            end_idx=end_idx,
            avg_adx=avg_adx,
            avg_atr=avg_atr,
            avg_return=avg_return,
            volatility=volatility,
            bars_count=end_idx - start_idx,
        )


def split_data_by_regime(
    data: pd.DataFrame,
    detector: RegimeDetector | None = None,
) -> dict[MarketRegime, pd.DataFrame]:
    """
    Split data into regime-specific subsets.

    Args:
        data: Historical OHLC data
        detector: RegimeDetector instance (creates default if None)

    Returns:
        Dict mapping regime -> data subset
    """
    if detector is None:
        detector = RegimeDetector()

    regimes = detector.detect_regimes(data)

    # Group data by regime
    regime_data: dict[MarketRegime, list[pd.DataFrame]] = {
        MarketRegime.TREND: [],
        MarketRegime.RANGE: [],
        MarketRegime.HIGH_VOL: [],
    }

    for regime_metrics in regimes:
        regime = regime_metrics.regime
        subset = data.iloc[regime_metrics.start_idx:regime_metrics.end_idx]

        if regime in regime_data:
            regime_data[regime].append(subset)

    # Concatenate subsets
    result: dict[MarketRegime, pd.DataFrame] = {}
    for regime, subsets in regime_data.items():
        if subsets:
            result[regime] = pd.concat(subsets, ignore_index=False)

    return result


def get_regime_statistics(regimes: list[RegimeMetrics]) -> dict[str, any]:
    """
    Calculate statistics across regimes.

    Args:
        regimes: List of regime metrics

    Returns:
        Dict with regime statistics
    """
    if not regimes:
        return {}

    trend_regimes = [r for r in regimes if r.regime == MarketRegime.TREND]
    range_regimes = [r for r in regimes if r.regime == MarketRegime.RANGE]
    high_vol_regimes = [r for r in regimes if r.regime == MarketRegime.HIGH_VOL]

    total_bars = sum(r.bars_count for r in regimes)

    return {
        "total_regimes": len(regimes),
        "total_bars": total_bars,
        "trend": {
            "count": len(trend_regimes),
            "bars": sum(r.bars_count for r in trend_regimes),
            "pct": sum(r.bars_count for r in trend_regimes) / total_bars if total_bars > 0 else 0,
            "avg_adx": sum(r.avg_adx for r in trend_regimes) / len(trend_regimes) if trend_regimes else 0,
        },
        "range": {
            "count": len(range_regimes),
            "bars": sum(r.bars_count for r in range_regimes),
            "pct": sum(r.bars_count for r in range_regimes) / total_bars if total_bars > 0 else 0,
            "avg_adx": sum(r.avg_adx for r in range_regimes) / len(range_regimes) if range_regimes else 0,
        },
        "high_vol": {
            "count": len(high_vol_regimes),
            "bars": sum(r.bars_count for r in high_vol_regimes),
            "pct": sum(r.bars_count for r in high_vol_regimes) / total_bars if total_bars > 0 else 0,
            "avg_atr": sum(r.avg_atr for r in high_vol_regimes) / len(high_vol_regimes) if high_vol_regimes else 0,
        },
    }


__all__ = [
    "MarketRegime",
    "RegimeMetrics",
    "RegimeDetector",
    "split_data_by_regime",
    "get_regime_statistics",
]
