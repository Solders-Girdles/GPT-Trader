"""
Technical indicators for trading strategy analysis.

Provides RSI, moving averages, crossover detection, and other technical indicators.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any


def mean_decimal(values: Sequence[Decimal]) -> Decimal:
    """Calculate the arithmetic mean of decimal values."""
    if not values:
        return Decimal("0")
    return sum(values, Decimal("0")) / len(values)


def simple_moving_average(values: Sequence[Decimal], period: int) -> Decimal | None:
    """Calculate Simple Moving Average for the given period.

    Args:
        values: Price series (most recent last)
        period: Number of periods for SMA

    Returns:
        SMA value or None if insufficient data
    """
    if len(values) < period:
        return None
    return mean_decimal(values[-period:])


def exponential_moving_average(
    values: Sequence[Decimal], period: int, smoothing: Decimal = Decimal("2")
) -> Decimal | None:
    """Calculate Exponential Moving Average for the given period.

    Args:
        values: Price series (most recent last)
        period: Number of periods for EMA
        smoothing: Smoothing factor (default 2 for standard EMA)

    Returns:
        EMA value or None if insufficient data
    """
    if len(values) < period:
        return None

    # Start with SMA for initial EMA value
    multiplier = smoothing / (Decimal(period) + Decimal("1"))
    ema = mean_decimal(values[:period])

    # Apply EMA formula for remaining values
    for price in values[period:]:
        ema = (price * multiplier) + (ema * (Decimal("1") - multiplier))

    return ema


def relative_strength_index(values: Sequence[Decimal], period: int = 14) -> Decimal | None:
    """Calculate Relative Strength Index using Wilder's smoothing method.

    The RSI measures momentum by comparing average gains to average losses
    over a specified period. Values range from 0 to 100:
    - RSI > 70: Potentially overbought (bearish signal)
    - RSI < 30: Potentially oversold (bullish signal)

    Args:
        values: Price series (most recent last), needs period + 1 values minimum
        period: Lookback period (default 14, standard RSI period)

    Returns:
        RSI value (0-100) or None if insufficient data
    """
    if len(values) < period + 1:
        return None

    # Calculate price changes
    changes: list[Decimal] = []
    for i in range(1, len(values)):
        changes.append(values[i] - values[i - 1])

    if len(changes) < period:
        return None

    # Separate gains and losses
    gains: list[Decimal] = []
    losses: list[Decimal] = []
    for change in changes:
        if change > 0:
            gains.append(change)
            losses.append(Decimal("0"))
        else:
            gains.append(Decimal("0"))
            losses.append(abs(change))

    # Calculate initial averages using simple average
    avg_gain = sum(gains[:period], Decimal("0")) / Decimal(period)
    avg_loss = sum(losses[:period], Decimal("0")) / Decimal(period)

    # Apply Wilder's smoothing for remaining values
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * Decimal(period - 1) + gains[i]) / Decimal(period)
        avg_loss = (avg_loss * Decimal(period - 1) + losses[i]) / Decimal(period)

    # Calculate RS and RSI
    if avg_loss == 0:
        return Decimal("100")  # No losses means maximum RSI

    relative_strength = avg_gain / avg_loss
    rsi = Decimal("100") - (Decimal("100") / (Decimal("1") + relative_strength))

    return rsi


@dataclass
class CrossoverSignal:
    """Result of a crossover detection."""

    crossed: bool
    direction: str  # "bullish" (fast crosses above slow) or "bearish" (fast crosses below slow)
    fast_value: Decimal
    slow_value: Decimal


def detect_crossover(
    fast_values: Sequence[Decimal],
    slow_values: Sequence[Decimal],
    lookback: int = 2,
) -> CrossoverSignal | None:
    """Detect if a crossover occurred between fast and slow moving averages.

    A crossover is detected when the fast MA crosses the slow MA within
    the lookback period.

    Args:
        fast_values: Fast MA values (most recent last)
        slow_values: Slow MA values (most recent last)
        lookback: Number of periods to check for crossover

    Returns:
        CrossoverSignal if crossover detected, None otherwise
    """
    if len(fast_values) < lookback or len(slow_values) < lookback:
        return None

    # Get recent values
    fast_current = fast_values[-1]
    slow_current = slow_values[-1]
    fast_prev = fast_values[-lookback]
    slow_prev = slow_values[-lookback]

    # Check for bullish crossover (fast crosses above slow)
    if fast_prev <= slow_prev and fast_current > slow_current:
        return CrossoverSignal(
            crossed=True,
            direction="bullish",
            fast_value=fast_current,
            slow_value=slow_current,
        )

    # Check for bearish crossover (fast crosses below slow)
    if fast_prev >= slow_prev and fast_current < slow_current:
        return CrossoverSignal(
            crossed=True,
            direction="bearish",
            fast_value=fast_current,
            slow_value=slow_current,
        )

    return CrossoverSignal(
        crossed=False,
        direction="none",
        fast_value=fast_current,
        slow_value=slow_current,
    )


def compute_ma_series(
    values: Sequence[Decimal], period: int, ma_type: str = "sma"
) -> list[Decimal]:
    """Compute a series of moving average values.

    Args:
        values: Price series
        period: MA period
        ma_type: "sma" for simple, "ema" for exponential

    Returns:
        List of MA values (same length as input, with None for insufficient data periods)
    """
    result: list[Decimal] = []

    for i in range(len(values)):
        if i < period - 1:
            # Not enough data yet - use simple average of available data
            result.append(mean_decimal(values[: i + 1]))
        else:
            window = values[: i + 1]
            if ma_type == "ema":
                ma = exponential_moving_average(window, period)
            else:
                ma = simple_moving_average(window, period)
            result.append(ma if ma is not None else values[i])

    return result


def to_decimal(value: Any) -> Decimal:
    """Convert any numeric value to Decimal."""
    return Decimal(str(value))


def true_range(high: Decimal, low: Decimal, prev_close: Decimal) -> Decimal:
    """Calculate True Range for ATR computation."""
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def average_true_range(
    highs: Sequence[Decimal],
    lows: Sequence[Decimal],
    closes: Sequence[Decimal],
    period: int = 14,
) -> Decimal | None:
    """Calculate Average True Range (ATR).

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        period: ATR period (default 14)

    Returns:
        ATR value or None if insufficient data
    """
    if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
        return None

    # Calculate true ranges
    true_ranges: list[Decimal] = []
    for i in range(1, len(closes)):
        tr = true_range(highs[i], lows[i], closes[i - 1])
        true_ranges.append(tr)

    if len(true_ranges) < period:
        return None

    # Use Wilder's smoothing (same as RSI)
    atr = sum(true_ranges[:period], Decimal("0")) / Decimal(period)
    for i in range(period, len(true_ranges)):
        atr = (atr * Decimal(period - 1) + true_ranges[i]) / Decimal(period)

    return atr


__all__ = [
    "mean_decimal",
    "simple_moving_average",
    "exponential_moving_average",
    "relative_strength_index",
    "detect_crossover",
    "compute_ma_series",
    "to_decimal",
    "true_range",
    "average_true_range",
    "CrossoverSignal",
]
