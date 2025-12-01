"""
Batch regime detection for historical backtesting.

Provides efficient batch processing of historical price data for regime
classification, enabling fast backtesting without per-bar overhead.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from ..regime.detector import MarketRegimeDetector
from ..regime.models import RegimeConfig, RegimeType


@dataclass
class RegimeSnapshot:
    """A single regime detection result at a point in time.

    Captures the full state of regime detection for historical analysis.
    """

    timestamp: datetime
    price: Decimal
    regime: RegimeType
    confidence: float
    volatility_percentile: float
    trend_percentile: float

    # Advanced indicators (if available)
    atr_value: float | None = None
    atr_percentile: float | None = None
    adx_value: float | None = None
    squeeze_score: float | None = None

    # Transition info
    transition_probability: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "price": str(self.price),
            "regime": self.regime.name,
            "confidence": round(self.confidence, 4),
            "volatility_percentile": round(self.volatility_percentile, 4),
            "trend_percentile": round(self.trend_percentile, 4),
            "atr_value": round(self.atr_value, 6) if self.atr_value else None,
            "atr_percentile": round(self.atr_percentile, 4) if self.atr_percentile else None,
            "adx_value": round(self.adx_value, 4) if self.adx_value else None,
            "squeeze_score": round(self.squeeze_score, 4) if self.squeeze_score else None,
            "transition_probability": self.transition_probability,
        }


@dataclass
class RegimeHistory:
    """Complete regime history for a symbol during backtest period.

    Provides analysis methods for understanding regime behavior.
    """

    symbol: str
    snapshots: list[RegimeSnapshot] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.snapshots)

    def __iter__(self) -> Iterator[RegimeSnapshot]:
        return iter(self.snapshots)

    def get_regime_at(self, timestamp: datetime) -> RegimeSnapshot | None:
        """Get regime snapshot at or before a given timestamp.

        Uses binary search for efficient lookup.
        """
        if not self.snapshots:
            return None

        # Binary search for nearest timestamp
        left, right = 0, len(self.snapshots) - 1
        result = None

        while left <= right:
            mid = (left + right) // 2
            if self.snapshots[mid].timestamp <= timestamp:
                result = self.snapshots[mid]
                left = mid + 1
            else:
                right = mid - 1

        return result

    def get_regime_distribution(self) -> dict[str, float]:
        """Calculate percentage of time spent in each regime.

        Returns:
            Dict mapping regime names to percentage (0-100)
        """
        if not self.snapshots:
            return {}

        counts: dict[str, int] = {}
        for snapshot in self.snapshots:
            name = snapshot.regime.name
            counts[name] = counts.get(name, 0) + 1

        total = len(self.snapshots)
        return {name: (count / total) * 100 for name, count in counts.items()}

    def get_regime_transitions(self) -> list[tuple[datetime, RegimeType, RegimeType]]:
        """Get all regime transitions.

        Returns:
            List of (timestamp, from_regime, to_regime) tuples
        """
        transitions = []

        for i in range(1, len(self.snapshots)):
            prev = self.snapshots[i - 1]
            curr = self.snapshots[i]

            if prev.regime != curr.regime:
                transitions.append((curr.timestamp, prev.regime, curr.regime))

        return transitions

    def get_crisis_periods(self) -> list[tuple[datetime, datetime]]:
        """Get all crisis periods.

        Returns:
            List of (start_time, end_time) tuples for crisis periods
        """
        periods = []
        crisis_start = None

        for snapshot in self.snapshots:
            if snapshot.regime == RegimeType.CRISIS:
                if crisis_start is None:
                    crisis_start = snapshot.timestamp
            else:
                if crisis_start is not None:
                    periods.append((crisis_start, snapshot.timestamp))
                    crisis_start = None

        # Handle ongoing crisis at end
        if crisis_start is not None and self.snapshots:
            periods.append((crisis_start, self.snapshots[-1].timestamp))

        return periods

    def get_average_confidence(self) -> float:
        """Get average regime detection confidence."""
        if not self.snapshots:
            return 0.0
        return sum(s.confidence for s in self.snapshots) / len(self.snapshots)

    def get_volatility_summary(self) -> dict[str, float]:
        """Get volatility statistics.

        Returns:
            Dict with mean, min, max volatility percentiles
        """
        if not self.snapshots:
            return {"mean": 0.0, "min": 0.0, "max": 0.0}

        volatilities = [s.volatility_percentile for s in self.snapshots]
        return {
            "mean": sum(volatilities) / len(volatilities),
            "min": min(volatilities),
            "max": max(volatilities),
        }

    def to_dataframe_rows(self) -> list[dict[str, Any]]:
        """Convert to list of dicts suitable for DataFrame creation.

        Example:
            import pandas as pd
            df = pd.DataFrame(history.to_dataframe_rows())
        """
        return [s.to_dict() for s in self.snapshots]

    def summary(self) -> dict[str, Any]:
        """Get summary statistics for the regime history."""
        distribution = self.get_regime_distribution()
        transitions = self.get_regime_transitions()
        crisis_periods = self.get_crisis_periods()

        return {
            "symbol": self.symbol,
            "total_bars": len(self.snapshots),
            "regime_distribution": distribution,
            "total_transitions": len(transitions),
            "crisis_periods": len(crisis_periods),
            "average_confidence": round(self.get_average_confidence(), 4),
            "volatility": self.get_volatility_summary(),
        }


class BatchRegimeDetector:
    """Batch regime detection for efficient backtesting.

    Processes historical price data in batch to generate regime history,
    much faster than per-bar online detection for backtesting scenarios.

    Features:
    - Vectorized processing for efficiency
    - Warmup period handling
    - Full regime history with all indicators
    - Transition analysis

    Example:
        detector = BatchRegimeDetector()

        # Process historical prices
        history = detector.process(
            symbol="BTC-USD",
            prices=[Decimal("50000"), ...],
            timestamps=[datetime(...), ...],
        )

        # Analyze results
        print(f"Regime distribution: {history.get_regime_distribution()}")
        print(f"Crisis periods: {history.get_crisis_periods()}")
    """

    def __init__(
        self,
        config: RegimeConfig | None = None,
        warmup_bars: int = 50,
    ):
        """Initialize batch detector.

        Args:
            config: Regime detection configuration
            warmup_bars: Number of bars to skip at start (for indicator warmup)
        """
        self.config = config or RegimeConfig()
        self.warmup_bars = warmup_bars

        # Create underlying detector
        self._detector = MarketRegimeDetector(self.config)

    def process(
        self,
        symbol: str,
        prices: Sequence[Decimal],
        timestamps: Sequence[datetime] | None = None,
    ) -> RegimeHistory:
        """Process historical prices and generate regime history.

        Args:
            symbol: Trading symbol
            prices: Sequence of historical prices (chronological order)
            timestamps: Optional timestamps for each price

        Returns:
            RegimeHistory with all snapshots
        """
        history = RegimeHistory(symbol=symbol)

        # Generate timestamps if not provided
        if timestamps is None:
            # Use sequential integers as placeholder timestamps
            base = datetime(2020, 1, 1)
            from datetime import timedelta

            timestamps = [base + timedelta(minutes=i) for i in range(len(prices))]

        # Process each price through detector
        for i, (price, timestamp) in enumerate(zip(prices, timestamps)):
            # Update detector
            regime_state = self._detector.update(symbol, price)

            # Skip warmup period
            if i < self.warmup_bars:
                continue

            # Get additional indicator values using public API
            indicators = self._detector.get_indicator_values(symbol)
            atr_value = indicators.get("atr")
            atr_percentile = indicators.get("atr_percentile")
            adx_value = indicators.get("adx")
            squeeze_score = indicators.get("squeeze_score")

            # Get transition probabilities
            transition_probs = None
            forecast = self._detector.get_transition_forecast(symbol)
            if forecast:
                transition_probs = {k: round(v, 4) for k, v in forecast.items()}

            # Create snapshot
            # Convert trend_score (-1 to 1) to percentile (0 to 1)
            trend_percentile = (regime_state.trend_score + 1.0) / 2.0

            snapshot = RegimeSnapshot(
                timestamp=timestamp,
                price=price,
                regime=regime_state.regime,
                confidence=regime_state.confidence,
                volatility_percentile=regime_state.volatility_percentile,
                trend_percentile=trend_percentile,
                atr_value=atr_value,
                atr_percentile=atr_percentile,
                adx_value=adx_value,
                squeeze_score=squeeze_score,
                transition_probability=transition_probs,
            )

            history.snapshots.append(snapshot)

        return history

    def process_candles(
        self,
        symbol: str,
        candles: Sequence[Any],
        price_field: str = "close",
    ) -> RegimeHistory:
        """Process historical candles and generate regime history.

        Convenience method for processing candle data.

        Args:
            symbol: Trading symbol
            candles: Sequence of candle objects
            price_field: Attribute name for price (default: "close")

        Returns:
            RegimeHistory with all snapshots
        """
        prices = []
        timestamps = []

        for candle in candles:
            # Get price
            price = getattr(candle, price_field, None)
            if price is None:
                continue
            prices.append(Decimal(str(price)) if not isinstance(price, Decimal) else price)

            # Get timestamp
            ts = getattr(candle, "ts", None) or getattr(candle, "timestamp", None)
            if ts is not None:
                timestamps.append(ts)

        # Process with timestamps if available
        if len(timestamps) == len(prices):
            return self.process(symbol, prices, timestamps)
        else:
            return self.process(symbol, prices)

    def reset(self) -> None:
        """Reset detector state for a new batch."""
        self._detector = MarketRegimeDetector(self.config)


__all__ = [
    "BatchRegimeDetector",
    "RegimeHistory",
    "RegimeSnapshot",
]
