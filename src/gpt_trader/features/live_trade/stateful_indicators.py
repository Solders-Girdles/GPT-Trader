"""
Stateful technical indicators using incremental O(1) algorithms.

These indicators maintain internal state and update incrementally,
avoiding O(n) recalculation each cycle. Supports serialization for
crash recovery via rehydrate/serialize methods.

Key algorithms:
- Welford's algorithm for online mean/variance (O(1) per update)
- Exponential smoothing for RSI/EMA (O(1) per update)
- Circular buffer for rolling windows (O(1) amortized)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any


@dataclass
class WelfordState:
    """Welford's online algorithm for computing mean and variance.

    Provides numerically stable O(1) updates for running statistics.

    Attributes:
        count: Number of samples seen
        mean: Current running mean
        m2: Sum of squared differences from mean (for variance)
    """

    count: int = 0
    mean: Decimal = field(default_factory=lambda: Decimal("0"))
    m2: Decimal = field(default_factory=lambda: Decimal("0"))

    def update(self, value: Decimal) -> None:
        """Update statistics with a new value. O(1) complexity."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / Decimal(self.count)
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> Decimal:
        """Population variance. Returns 0 if count < 2."""
        if self.count < 2:
            return Decimal("0")
        return self.m2 / Decimal(self.count)

    @property
    def sample_variance(self) -> Decimal:
        """Sample variance (Bessel's correction). Returns 0 if count < 2."""
        if self.count < 2:
            return Decimal("0")
        return self.m2 / Decimal(self.count - 1)

    @property
    def std_dev(self) -> Decimal:
        """Population standard deviation."""
        return self.variance.sqrt() if self.variance > 0 else Decimal("0")

    @property
    def sample_std_dev(self) -> Decimal:
        """Sample standard deviation."""
        return self.sample_variance.sqrt() if self.sample_variance > 0 else Decimal("0")

    def serialize(self) -> dict[str, Any]:
        """Serialize state for persistence."""
        return {
            "count": self.count,
            "mean": str(self.mean),
            "m2": str(self.m2),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> WelfordState:
        """Restore state from serialized data."""
        return cls(
            count=data["count"],
            mean=Decimal(data["mean"]),
            m2=Decimal(data["m2"]),
        )


@dataclass
class RollingWindow:
    """Fixed-size circular buffer for windowed calculations.

    Maintains a rolling window of values with O(1) add/remove operations.
    Supports incremental sum tracking for O(1) mean calculation.

    Attributes:
        max_size: Maximum window size
        values: Deque of values (most recent last)
        running_sum: Current sum of all values in window
    """

    max_size: int
    values: deque[Decimal] = field(default_factory=deque)
    running_sum: Decimal = field(default_factory=lambda: Decimal("0"))

    def __post_init__(self) -> None:
        if not isinstance(self.values, deque):
            self.values = deque(self.values, maxlen=self.max_size)
        else:
            self.values = deque(self.values, maxlen=self.max_size)

    def add(self, value: Decimal) -> Decimal | None:
        """Add a value, returning the evicted value if window was full."""
        evicted = None
        if len(self.values) == self.max_size:
            evicted = self.values[0]
            self.running_sum -= evicted

        self.values.append(value)
        self.running_sum += value
        return evicted

    @property
    def mean(self) -> Decimal | None:
        """O(1) mean calculation using running sum."""
        if not self.values:
            return None
        return self.running_sum / Decimal(len(self.values))

    @property
    def is_full(self) -> bool:
        """Whether the window has reached max_size."""
        return len(self.values) == self.max_size

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> Decimal:
        return self.values[index]

    def serialize(self) -> dict[str, Any]:
        """Serialize state for persistence."""
        return {
            "max_size": self.max_size,
            "values": [str(v) for v in self.values],
            "running_sum": str(self.running_sum),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> RollingWindow:
        """Restore state from serialized data."""
        window = cls(max_size=data["max_size"])
        window.values = deque([Decimal(v) for v in data["values"]], maxlen=data["max_size"])
        window.running_sum = Decimal(data["running_sum"])
        return window


@dataclass
class OnlineEMA:
    """Online Exponential Moving Average with O(1) updates.

    Uses standard EMA formula: EMA = price * k + prev_EMA * (1 - k)
    where k = smoothing / (period + 1)

    Attributes:
        period: EMA period
        smoothing: Smoothing factor (default 2.0 for standard EMA)
        value: Current EMA value (None until initialized)
        initialized: Whether EMA has been seeded with initial SMA
        warmup_values: Values collected during warmup phase
    """

    period: int
    smoothing: Decimal = field(default_factory=lambda: Decimal("2"))
    value: Decimal | None = None
    initialized: bool = False
    warmup_values: list[Decimal] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._multiplier = self.smoothing / (Decimal(self.period) + Decimal("1"))

    def update(self, price: Decimal) -> Decimal | None:
        """Update EMA with new price. O(1) after initialization."""
        if not self.initialized:
            self.warmup_values.append(price)
            if len(self.warmup_values) >= self.period:
                # Initialize with SMA of warmup values
                self.value = sum(self.warmup_values[: self.period], Decimal("0")) / Decimal(
                    self.period
                )
                self.initialized = True
                # Process any remaining warmup values
                for p in self.warmup_values[self.period :]:
                    self.value = (p * self._multiplier) + (
                        self.value * (Decimal("1") - self._multiplier)
                    )
                self.warmup_values = []  # Free memory
            return self.value

        # Standard EMA update: O(1)
        self.value = (price * self._multiplier) + (self.value * (Decimal("1") - self._multiplier))
        return self.value

    def serialize(self) -> dict[str, Any]:
        """Serialize state for persistence."""
        return {
            "period": self.period,
            "smoothing": str(self.smoothing),
            "value": str(self.value) if self.value is not None else None,
            "initialized": self.initialized,
            "warmup_values": [str(v) for v in self.warmup_values],
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> OnlineEMA:
        """Restore state from serialized data."""
        ema = cls(
            period=data["period"],
            smoothing=Decimal(data["smoothing"]),
        )
        ema.value = Decimal(data["value"]) if data["value"] is not None else None
        ema.initialized = data["initialized"]
        ema.warmup_values = [Decimal(v) for v in data["warmup_values"]]
        return ema


@dataclass
class OnlineSMA:
    """Online Simple Moving Average using rolling window.

    Maintains a circular buffer for O(1) amortized updates.

    Attributes:
        period: SMA period
        window: Rolling window of values
    """

    period: int
    window: RollingWindow = field(init=False)

    def __post_init__(self) -> None:
        self.window = RollingWindow(max_size=self.period)

    def update(self, price: Decimal) -> Decimal | None:
        """Update SMA with new price. O(1) amortized."""
        self.window.add(price)
        return self.window.mean

    @property
    def value(self) -> Decimal | None:
        """Current SMA value."""
        return self.window.mean

    @property
    def is_ready(self) -> bool:
        """Whether SMA has enough data points."""
        return self.window.is_full

    def serialize(self) -> dict[str, Any]:
        """Serialize state for persistence."""
        return {
            "period": self.period,
            "window": self.window.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> OnlineSMA:
        """Restore state from serialized data."""
        sma = cls(period=data["period"])
        sma.window = RollingWindow.deserialize(data["window"])
        return sma


@dataclass
class OnlineRSI:
    """Online Relative Strength Index using Wilder's smoothing.

    Maintains exponentially smoothed gain/loss averages for O(1) updates
    after initialization.

    RSI = 100 - (100 / (1 + RS))
    where RS = avg_gain / avg_loss

    Attributes:
        period: RSI period (default 14)
        avg_gain: Smoothed average gain
        avg_loss: Smoothed average loss
        prev_price: Previous price for change calculation
        initialized: Whether RSI has been seeded
        warmup_gains: Gains collected during warmup
        warmup_losses: Losses collected during warmup
        value: Current RSI value
    """

    period: int = 14
    avg_gain: Decimal = field(default_factory=lambda: Decimal("0"))
    avg_loss: Decimal = field(default_factory=lambda: Decimal("0"))
    prev_price: Decimal | None = None
    initialized: bool = False
    warmup_gains: list[Decimal] = field(default_factory=list)
    warmup_losses: list[Decimal] = field(default_factory=list)
    value: Decimal | None = None

    def update(self, price: Decimal) -> Decimal | None:
        """Update RSI with new price. O(1) after initialization."""
        if self.prev_price is None:
            self.prev_price = price
            return None

        # Calculate change
        change = price - self.prev_price
        self.prev_price = price

        gain = change if change > 0 else Decimal("0")
        loss = abs(change) if change < 0 else Decimal("0")

        if not self.initialized:
            self.warmup_gains.append(gain)
            self.warmup_losses.append(loss)

            if len(self.warmup_gains) >= self.period:
                # Initialize with simple average
                self.avg_gain = sum(self.warmup_gains[: self.period], Decimal("0")) / Decimal(
                    self.period
                )
                self.avg_loss = sum(self.warmup_losses[: self.period], Decimal("0")) / Decimal(
                    self.period
                )
                self.initialized = True

                # Process remaining warmup values
                for i in range(self.period, len(self.warmup_gains)):
                    self._apply_wilder_smoothing(self.warmup_gains[i], self.warmup_losses[i])

                self.warmup_gains = []
                self.warmup_losses = []
                self.value = self._calculate_rsi()

            return self.value

        # Wilder's smoothing: O(1)
        self._apply_wilder_smoothing(gain, loss)
        self.value = self._calculate_rsi()
        return self.value

    def _apply_wilder_smoothing(self, gain: Decimal, loss: Decimal) -> None:
        """Apply Wilder's exponential smoothing to gain/loss averages."""
        period_decimal = Decimal(self.period)
        self.avg_gain = (self.avg_gain * (period_decimal - 1) + gain) / period_decimal
        self.avg_loss = (self.avg_loss * (period_decimal - 1) + loss) / period_decimal

    def _calculate_rsi(self) -> Decimal:
        """Calculate RSI from current avg_gain and avg_loss."""
        if self.avg_loss == 0:
            return Decimal("100")
        relative_strength = self.avg_gain / self.avg_loss
        return Decimal("100") - (Decimal("100") / (Decimal("1") + relative_strength))

    def serialize(self) -> dict[str, Any]:
        """Serialize state for persistence."""
        return {
            "period": self.period,
            "avg_gain": str(self.avg_gain),
            "avg_loss": str(self.avg_loss),
            "prev_price": str(self.prev_price) if self.prev_price is not None else None,
            "initialized": self.initialized,
            "warmup_gains": [str(g) for g in self.warmup_gains],
            "warmup_losses": [str(lo) for lo in self.warmup_losses],
            "value": str(self.value) if self.value is not None else None,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> OnlineRSI:
        """Restore state from serialized data."""
        rsi = cls(period=data["period"])
        rsi.avg_gain = Decimal(data["avg_gain"])
        rsi.avg_loss = Decimal(data["avg_loss"])
        rsi.prev_price = Decimal(data["prev_price"]) if data["prev_price"] is not None else None
        rsi.initialized = data["initialized"]
        rsi.warmup_gains = [Decimal(g) for g in data["warmup_gains"]]
        rsi.warmup_losses = [Decimal(lo) for lo in data["warmup_losses"]]
        rsi.value = Decimal(data["value"]) if data["value"] is not None else None
        return rsi


@dataclass
class OnlineZScore:
    """Online Z-Score calculation using Welford's algorithm.

    Z-Score = (value - mean) / std_dev

    Useful for mean reversion strategies.

    Attributes:
        welford: Welford state for mean/variance
        lookback: Optional rolling window size (None = all history)
        window: Rolling window (if lookback specified)
    """

    welford: WelfordState = field(default_factory=WelfordState)
    lookback: int | None = None
    window: RollingWindow | None = None

    def __post_init__(self) -> None:
        if self.lookback is not None:
            self.window = RollingWindow(max_size=self.lookback)
            # Also create a windowed Welford for the lookback period
            self._window_welford = WelfordState()

    def update(self, value: Decimal) -> Decimal | None:
        """Update and return current Z-Score. O(1) complexity."""
        # Update global Welford (all-time statistics)
        self.welford.update(value)

        if self.lookback is not None and self.window is not None:
            # For windowed Z-Score, we need to track mean/variance of window
            self.window.add(value)

            # Recalculate windowed stats (could optimize with Welford removal)
            if self.window.is_full:
                values = list(self.window.values)
                mean = sum(values, Decimal("0")) / Decimal(len(values))
                variance = sum((v - mean) ** 2 for v in values) / Decimal(len(values))
                std_dev = variance.sqrt() if variance > 0 else Decimal("0")

                if std_dev == 0:
                    return Decimal("0")
                return (value - mean) / std_dev

            return None  # Not enough data for windowed Z-Score

        # Global Z-Score
        if self.welford.count < 2:
            return None

        std_dev = self.welford.std_dev
        if std_dev == 0:
            return Decimal("0")

        return (value - self.welford.mean) / std_dev

    @property
    def mean(self) -> Decimal:
        """Current mean (from Welford or window)."""
        if self.lookback is not None and self.window is not None and self.window.is_full:
            return self.window.mean or Decimal("0")
        return self.welford.mean

    def serialize(self) -> dict[str, Any]:
        """Serialize state for persistence."""
        return {
            "welford": self.welford.serialize(),
            "lookback": self.lookback,
            "window": self.window.serialize() if self.window is not None else None,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> OnlineZScore:
        """Restore state from serialized data."""
        zscore = cls(lookback=data["lookback"])
        zscore.welford = WelfordState.deserialize(data["welford"])
        if data["window"] is not None:
            zscore.window = RollingWindow.deserialize(data["window"])
        return zscore


@dataclass
class IndicatorBundle:
    """Bundle of commonly used indicators for a single symbol.

    Provides a convenient container for managing multiple indicators
    together with unified serialization.

    Attributes:
        symbol: Trading symbol
        short_sma: Short-period SMA
        long_sma: Long-period SMA
        rsi: RSI indicator
        price_history: Recent prices for crossover detection
    """

    symbol: str
    short_sma: OnlineSMA
    long_sma: OnlineSMA
    rsi: OnlineRSI
    price_history: RollingWindow

    @classmethod
    def create(
        cls,
        symbol: str,
        short_period: int = 5,
        long_period: int = 20,
        rsi_period: int = 14,
        history_size: int = 50,
    ) -> IndicatorBundle:
        """Create a new indicator bundle with specified parameters."""
        return cls(
            symbol=symbol,
            short_sma=OnlineSMA(period=short_period),
            long_sma=OnlineSMA(period=long_period),
            rsi=OnlineRSI(period=rsi_period),
            price_history=RollingWindow(max_size=history_size),
        )

    def update(self, price: Decimal) -> dict[str, Decimal | None]:
        """Update all indicators with new price. Returns current values."""
        self.price_history.add(price)
        return {
            "short_sma": self.short_sma.update(price),
            "long_sma": self.long_sma.update(price),
            "rsi": self.rsi.update(price),
            "price": price,
        }

    def serialize(self) -> dict[str, Any]:
        """Serialize all indicator state."""
        return {
            "symbol": self.symbol,
            "short_sma": self.short_sma.serialize(),
            "long_sma": self.long_sma.serialize(),
            "rsi": self.rsi.serialize(),
            "price_history": self.price_history.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> IndicatorBundle:
        """Restore indicator bundle from serialized data."""
        return cls(
            symbol=data["symbol"],
            short_sma=OnlineSMA.deserialize(data["short_sma"]),
            long_sma=OnlineSMA.deserialize(data["long_sma"]),
            rsi=OnlineRSI.deserialize(data["rsi"]),
            price_history=RollingWindow.deserialize(data["price_history"]),
        )


__all__ = [
    "WelfordState",
    "RollingWindow",
    "OnlineEMA",
    "OnlineSMA",
    "OnlineRSI",
    "OnlineZScore",
    "IndicatorBundle",
]
