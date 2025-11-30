"""
Advanced regime detection indicators using O(1) incremental algorithms.

Provides additional technical indicators beyond basic EMAs for more
accurate regime classification:
- Online ATR (Average True Range) for volatility measurement
- Bollinger Band width for squeeze detection
- ADX-style trend strength indicator
- Regime transition matrix for probabilistic forecasting
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from gpt_trader.features.live_trade.stateful_indicators import (
    OnlineEMA,
    RollingWindow,
    WelfordState,
)


@dataclass
class OnlineATR:
    """Online Average True Range using Wilder's smoothing.

    ATR measures volatility using the True Range:
    TR = max(high - low, |high - prev_close|, |low - prev_close|)

    For tick data without high/low, we approximate using price changes.

    All updates are O(1) after initialization.
    """

    period: int = 14
    _atr: Decimal | None = None
    _prev_price: Decimal | None = None
    _warmup_trs: list[Decimal] = field(default_factory=list)
    _initialized: bool = False

    def update(
        self, price: Decimal, high: Decimal | None = None, low: Decimal | None = None
    ) -> Decimal | None:
        """Update ATR with new price data.

        Args:
            price: Current close/tick price
            high: High of period (optional, uses price if None)
            low: Low of period (optional, uses price if None)

        Returns:
            Current ATR value or None if warming up
        """
        if self._prev_price is None:
            self._prev_price = price
            return None

        # Calculate True Range
        # For tick data, we approximate TR as absolute price change
        if high is not None and low is not None:
            tr1 = high - low
            tr2 = abs(high - self._prev_price)
            tr3 = abs(low - self._prev_price)
            true_range = max(tr1, tr2, tr3)
        else:
            # Approximate with price change magnitude
            true_range = abs(price - self._prev_price)

        self._prev_price = price

        if not self._initialized:
            self._warmup_trs.append(true_range)
            if len(self._warmup_trs) >= self.period:
                # Initialize with simple average
                self._atr = sum(self._warmup_trs[: self.period], Decimal("0")) / Decimal(
                    self.period
                )
                self._initialized = True
                # Process remaining warmup values
                for tr in self._warmup_trs[self.period :]:
                    self._apply_wilder_smoothing(tr)
                self._warmup_trs = []
            return self._atr

        # Wilder's smoothing: ATR = ((period-1) * prev_ATR + TR) / period
        self._apply_wilder_smoothing(true_range)
        return self._atr

    def _apply_wilder_smoothing(self, true_range: Decimal) -> None:
        """Apply Wilder's exponential smoothing."""
        if self._atr is not None:
            period_dec = Decimal(self.period)
            self._atr = ((period_dec - 1) * self._atr + true_range) / period_dec

    @property
    def value(self) -> Decimal | None:
        """Current ATR value."""
        return self._atr

    @property
    def is_ready(self) -> bool:
        """Whether ATR has enough data."""
        return self._initialized

    def serialize(self) -> dict[str, Any]:
        """Serialize state for persistence."""
        return {
            "period": self.period,
            "atr": str(self._atr) if self._atr is not None else None,
            "prev_price": str(self._prev_price) if self._prev_price is not None else None,
            "warmup_trs": [str(tr) for tr in self._warmup_trs],
            "initialized": self._initialized,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> OnlineATR:
        """Restore from serialized data."""
        atr = cls(period=data["period"])
        atr._atr = Decimal(data["atr"]) if data["atr"] is not None else None
        atr._prev_price = Decimal(data["prev_price"]) if data["prev_price"] is not None else None
        atr._warmup_trs = [Decimal(tr) for tr in data["warmup_trs"]]
        atr._initialized = data["initialized"]
        return atr


@dataclass
class OnlineBollingerBands:
    """Online Bollinger Bands using rolling statistics.

    Tracks:
    - Middle band (SMA)
    - Upper/Lower bands (SMA +/- k * std_dev)
    - Band width (useful for squeeze detection)
    - %B (price position within bands)

    Uses Welford's algorithm for numerically stable O(1) updates.
    """

    period: int = 20
    num_std: float = 2.0

    _prices: RollingWindow = field(init=False)
    _welford: WelfordState = field(default_factory=WelfordState)
    _current_price: Decimal | None = None

    def __post_init__(self) -> None:
        self._prices = RollingWindow(max_size=self.period)

    def update(self, price: Decimal) -> dict[str, Decimal | None]:
        """Update Bollinger Bands with new price.

        Returns dict with:
        - middle: Middle band (SMA)
        - upper: Upper band
        - lower: Lower band
        - width: Band width ((upper - lower) / middle)
        - percent_b: %B ((price - lower) / (upper - lower))
        """
        self._current_price = price
        self._prices.add(price)

        # Need full period for meaningful bands
        if not self._prices.is_full:
            return {
                "middle": None,
                "upper": None,
                "lower": None,
                "width": None,
                "percent_b": None,
            }

        # Calculate mean (SMA)
        middle = self._prices.mean
        if middle is None:
            return {
                "middle": None,
                "upper": None,
                "lower": None,
                "width": None,
                "percent_b": None,
            }

        # Calculate standard deviation
        prices_list = list(self._prices.values)
        variance = sum((p - middle) ** 2 for p in prices_list) / Decimal(len(prices_list))
        std_dev = variance.sqrt() if variance > 0 else Decimal("0")

        # Calculate bands
        band_width_decimal = std_dev * Decimal(str(self.num_std))
        upper = middle + band_width_decimal
        lower = middle - band_width_decimal

        # Band width as percentage of middle
        width = (upper - lower) / middle if middle > 0 else Decimal("0")

        # %B (price position within bands)
        band_range = upper - lower
        if band_range > 0:
            percent_b = (price - lower) / band_range
        else:
            percent_b = Decimal("0.5")

        return {
            "middle": middle,
            "upper": upper,
            "lower": lower,
            "width": width,
            "percent_b": percent_b,
            "std_dev": std_dev,
        }

    @property
    def is_ready(self) -> bool:
        """Whether bands have enough data."""
        return self._prices.is_full

    def get_squeeze_score(self) -> float:
        """Calculate squeeze score (0 = no squeeze, 1 = extreme squeeze).

        Low band width relative to historical suggests a squeeze
        (consolidation before potential breakout).
        """
        result = self.update(self._current_price) if self._current_price else {}
        width = result.get("width")
        if width is None:
            return 0.5

        # Normalize width - lower width = higher squeeze score
        # Typical band width is 2-10%, squeeze is < 2%
        width_float = float(width)
        if width_float < 0.02:
            return 1.0
        elif width_float > 0.10:
            return 0.0
        else:
            return 1.0 - (width_float - 0.02) / 0.08

    def serialize(self) -> dict[str, Any]:
        """Serialize state for persistence."""
        return {
            "period": self.period,
            "num_std": self.num_std,
            "prices": self._prices.serialize(),
            "current_price": str(self._current_price) if self._current_price else None,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> OnlineBollingerBands:
        """Restore from serialized data."""
        bb = cls(period=data["period"], num_std=data["num_std"])
        bb._prices = RollingWindow.deserialize(data["prices"])
        bb._current_price = Decimal(data["current_price"]) if data["current_price"] else None
        return bb


@dataclass
class OnlineTrendStrength:
    """Online trend strength indicator (simplified ADX-style).

    Measures how strong the current trend is, regardless of direction.
    Useful for distinguishing trending vs ranging markets.

    Uses directional movement analysis:
    - +DM: Upward directional movement
    - -DM: Downward directional movement
    - DX: Directional index = |+DI - -DI| / (+DI + -DI)
    - ADX: Smoothed DX

    Higher values indicate stronger trend.
    """

    period: int = 14

    _plus_dm_ema: OnlineEMA = field(init=False)
    _minus_dm_ema: OnlineEMA = field(init=False)
    _tr_ema: OnlineEMA = field(init=False)
    _adx_ema: OnlineEMA = field(init=False)

    _prev_price: Decimal | None = None
    _prev_high: Decimal | None = None
    _prev_low: Decimal | None = None

    def __post_init__(self) -> None:
        self._plus_dm_ema = OnlineEMA(period=self.period)
        self._minus_dm_ema = OnlineEMA(period=self.period)
        self._tr_ema = OnlineEMA(period=self.period)
        self._adx_ema = OnlineEMA(period=self.period)

    def update(
        self,
        price: Decimal,
        high: Decimal | None = None,
        low: Decimal | None = None,
    ) -> dict[str, float | None]:
        """Update trend strength with new price.

        Args:
            price: Current price (used as high/low if not provided)
            high: Period high
            low: Period low

        Returns dict with:
        - plus_di: Positive directional indicator (0-100)
        - minus_di: Negative directional indicator (0-100)
        - dx: Directional index (0-100)
        - adx: Average directional index (0-100, trend strength)
        """
        # Use price as high/low for tick data
        current_high = high if high is not None else price
        current_low = low if low is not None else price

        if self._prev_price is None:
            self._prev_price = price
            self._prev_high = current_high
            self._prev_low = current_low
            return {"plus_di": None, "minus_di": None, "dx": None, "adx": None}

        # Calculate directional movement
        up_move = current_high - (self._prev_high or current_high)
        down_move = (self._prev_low or current_low) - current_low

        plus_dm = up_move if up_move > down_move and up_move > 0 else Decimal("0")
        minus_dm = down_move if down_move > up_move and down_move > 0 else Decimal("0")

        # True range
        tr1 = current_high - current_low
        tr2 = abs(current_high - self._prev_price)
        tr3 = abs(current_low - self._prev_price)
        true_range = max(tr1, tr2, tr3)

        # Update EMAs
        self._plus_dm_ema.update(plus_dm)
        self._minus_dm_ema.update(minus_dm)
        self._tr_ema.update(true_range)

        self._prev_price = price
        self._prev_high = current_high
        self._prev_low = current_low

        # Calculate DI values
        tr_smoothed = self._tr_ema.value
        if tr_smoothed is None or tr_smoothed == 0:
            return {"plus_di": None, "minus_di": None, "dx": None, "adx": None}

        plus_di_val = self._plus_dm_ema.value
        minus_di_val = self._minus_dm_ema.value

        if plus_di_val is None or minus_di_val is None:
            return {"plus_di": None, "minus_di": None, "dx": None, "adx": None}

        plus_di = float(plus_di_val / tr_smoothed * 100)
        minus_di = float(minus_di_val / tr_smoothed * 100)

        # DX = |+DI - -DI| / (+DI + -DI) * 100
        di_sum = plus_di + minus_di
        if di_sum == 0:
            dx = 0.0
        else:
            dx = abs(plus_di - minus_di) / di_sum * 100

        # ADX = smoothed DX
        adx_val = self._adx_ema.update(Decimal(str(dx)))
        adx = float(adx_val) if adx_val is not None else None

        return {
            "plus_di": plus_di,
            "minus_di": minus_di,
            "dx": dx,
            "adx": adx,
        }

    @property
    def is_ready(self) -> bool:
        """Whether indicator has enough data."""
        return self._adx_ema.initialized

    def serialize(self) -> dict[str, Any]:
        """Serialize state for persistence."""
        return {
            "period": self.period,
            "plus_dm_ema": self._plus_dm_ema.serialize(),
            "minus_dm_ema": self._minus_dm_ema.serialize(),
            "tr_ema": self._tr_ema.serialize(),
            "adx_ema": self._adx_ema.serialize(),
            "prev_price": str(self._prev_price) if self._prev_price else None,
            "prev_high": str(self._prev_high) if self._prev_high else None,
            "prev_low": str(self._prev_low) if self._prev_low else None,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> OnlineTrendStrength:
        """Restore from serialized data."""
        ts = cls(period=data["period"])
        ts._plus_dm_ema = OnlineEMA.deserialize(data["plus_dm_ema"])
        ts._minus_dm_ema = OnlineEMA.deserialize(data["minus_dm_ema"])
        ts._tr_ema = OnlineEMA.deserialize(data["tr_ema"])
        ts._adx_ema = OnlineEMA.deserialize(data["adx_ema"])
        ts._prev_price = Decimal(data["prev_price"]) if data["prev_price"] else None
        ts._prev_high = Decimal(data["prev_high"]) if data["prev_high"] else None
        ts._prev_low = Decimal(data["prev_low"]) if data["prev_low"] else None
        return ts


@dataclass
class RegimeTransitionMatrix:
    """Tracks historical regime transitions for probabilistic forecasting.

    Maintains a count matrix of transitions between regimes,
    enabling estimation of transition probabilities.

    P(next_regime | current_regime) = count[current][next] / sum(count[current])
    """

    # Transition counts: from_regime -> to_regime -> count
    _transitions: dict[str, dict[str, int]] = field(default_factory=dict)
    _last_regime: str | None = None
    _total_transitions: int = 0

    def record_transition(self, from_regime: str, to_regime: str) -> None:
        """Record a regime transition.

        Args:
            from_regime: Previous regime name
            to_regime: New regime name
        """
        if from_regime not in self._transitions:
            self._transitions[from_regime] = {}

        if to_regime not in self._transitions[from_regime]:
            self._transitions[from_regime][to_regime] = 0

        self._transitions[from_regime][to_regime] += 1
        self._total_transitions += 1
        self._last_regime = to_regime

    def update(self, new_regime: str) -> None:
        """Update with new regime, recording transition if changed.

        Args:
            new_regime: Current regime name
        """
        if self._last_regime is not None and new_regime != self._last_regime:
            self.record_transition(self._last_regime, new_regime)
        self._last_regime = new_regime

    def get_transition_probability(self, from_regime: str, to_regime: str) -> float:
        """Get estimated probability of transition.

        Args:
            from_regime: Current regime
            to_regime: Target regime

        Returns:
            Probability estimate (0.0 to 1.0)
        """
        if from_regime not in self._transitions:
            return 0.0

        from_counts = self._transitions[from_regime]
        total_from = sum(from_counts.values())

        if total_from == 0:
            return 0.0

        return from_counts.get(to_regime, 0) / total_from

    def get_most_likely_next(self, current_regime: str) -> tuple[str | None, float]:
        """Get most likely next regime and its probability.

        Args:
            current_regime: Current regime

        Returns:
            Tuple of (next_regime_name, probability)
        """
        if current_regime not in self._transitions:
            return None, 0.0

        from_counts = self._transitions[current_regime]
        if not from_counts:
            return None, 0.0

        best_regime = max(from_counts, key=from_counts.get)  # type: ignore
        total_from = sum(from_counts.values())

        return best_regime, from_counts[best_regime] / total_from

    def get_transition_distribution(self, current_regime: str) -> dict[str, float]:
        """Get full transition probability distribution.

        Args:
            current_regime: Current regime

        Returns:
            Dict mapping regime names to probabilities
        """
        if current_regime not in self._transitions:
            return {}

        from_counts = self._transitions[current_regime]
        total_from = sum(from_counts.values())

        if total_from == 0:
            return {}

        return {regime: count / total_from for regime, count in from_counts.items()}

    def serialize(self) -> dict[str, Any]:
        """Serialize state for persistence."""
        return {
            "transitions": self._transitions,
            "last_regime": self._last_regime,
            "total_transitions": self._total_transitions,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> RegimeTransitionMatrix:
        """Restore from serialized data."""
        matrix = cls()
        matrix._transitions = data.get("transitions", {})
        matrix._last_regime = data.get("last_regime")
        matrix._total_transitions = data.get("total_transitions", 0)
        return matrix


__all__ = [
    "OnlineATR",
    "OnlineBollingerBands",
    "OnlineTrendStrength",
    "RegimeTransitionMatrix",
]
