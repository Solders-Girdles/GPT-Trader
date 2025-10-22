"""
Technical indicators for strategy enhancement.

Provides ATR, ADX, and other indicators to improve baseline strategy performance.
"""

from decimal import Decimal
from typing import Sequence


def calculate_atr(
    highs: Sequence[Decimal],
    lows: Sequence[Decimal],
    closes: Sequence[Decimal],
    period: int = 14,
) -> Decimal | None:
    """
    Calculate Average True Range (ATR).

    ATR measures volatility by decomposing the entire range of an asset price
    for that period.

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

    # Calculate True Range for each bar
    true_ranges: list[Decimal] = []

    for i in range(1, len(highs)):
        high_low = highs[i] - lows[i]
        high_close = abs(highs[i] - closes[i - 1])
        low_close = abs(lows[i] - closes[i - 1])

        true_range = max(high_low, high_close, low_close)
        true_ranges.append(true_range)

    if len(true_ranges) < period:
        return None

    # Calculate initial ATR as simple average
    atr = sum(true_ranges[-period:]) / Decimal(period)

    return atr


def calculate_adx(
    highs: Sequence[Decimal],
    lows: Sequence[Decimal],
    closes: Sequence[Decimal],
    period: int = 14,
) -> Decimal | None:
    """
    Calculate Average Directional Index (ADX).

    ADX measures trend strength regardless of direction.
    Values > 25 indicate strong trend, < 20 indicate weak/ranging.

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        period: ADX period (default 14)

    Returns:
        ADX value or None if insufficient data
    """
    if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
        return None

    # Calculate +DM and -DM
    plus_dm_values: list[Decimal] = []
    minus_dm_values: list[Decimal] = []

    for i in range(1, len(highs)):
        high_diff = highs[i] - highs[i - 1]
        low_diff = lows[i - 1] - lows[i]

        plus_dm = Decimal("0")
        minus_dm = Decimal("0")

        if high_diff > low_diff and high_diff > Decimal("0"):
            plus_dm = high_diff
        if low_diff > high_diff and low_diff > Decimal("0"):
            minus_dm = low_diff

        plus_dm_values.append(plus_dm)
        minus_dm_values.append(minus_dm)

    if len(plus_dm_values) < period:
        return None

    # Calculate ATR for normalization
    atr = calculate_atr(highs, lows, closes, period)
    if atr is None or atr == Decimal("0"):
        return None

    # Calculate smoothed +DI and -DI
    plus_di = (sum(plus_dm_values[-period:]) / Decimal(period)) / atr * Decimal("100")
    minus_di = (sum(minus_dm_values[-period:]) / Decimal(period)) / atr * Decimal("100")

    # Calculate DX
    di_sum = plus_di + minus_di
    if di_sum == Decimal("0"):
        return None

    dx = abs(plus_di - minus_di) / di_sum * Decimal("100")

    # ADX is typically smoothed DX, but for simplicity we return DX
    # (proper ADX would require maintaining state for exponential smoothing)
    return dx


def calculate_rsi(
    closes: Sequence[Decimal],
    period: int = 14,
) -> Decimal | None:
    """
    Calculate Relative Strength Index (RSI).

    RSI oscillates between 0 and 100. Traditionally:
    - RSI > 70: Overbought
    - RSI < 30: Oversold

    Args:
        closes: Close prices
        period: RSI period (default 14)

    Returns:
        RSI value or None if insufficient data
    """
    if len(closes) < period + 1:
        return None

    # Calculate price changes
    gains: list[Decimal] = []
    losses: list[Decimal] = []

    for i in range(1, len(closes)):
        change = closes[i] - closes[i - 1]
        if change > Decimal("0"):
            gains.append(change)
            losses.append(Decimal("0"))
        else:
            gains.append(Decimal("0"))
            losses.append(abs(change))

    if len(gains) < period:
        return None

    # Calculate average gain and loss
    avg_gain = sum(gains[-period:]) / Decimal(period)
    avg_loss = sum(losses[-period:]) / Decimal(period)

    if avg_loss == Decimal("0"):
        return Decimal("100")

    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = Decimal("100") - (Decimal("100") / (Decimal("1") + rs))

    return rsi


def is_volatile_market(
    current_atr: Decimal | None,
    historical_closes: Sequence[Decimal],
    highs: Sequence[Decimal] | None = None,
    lows: Sequence[Decimal] | None = None,
    threshold_multiplier: Decimal = Decimal("1.5"),
    lookback: int = 50,
) -> bool:
    """
    Determine if market is in high volatility regime.

    Args:
        current_atr: Current ATR value
        historical_closes: Historical close prices for baseline
        highs: Historical highs (optional, for better ATR calculation)
        lows: Historical lows (optional, for better ATR calculation)
        threshold_multiplier: Multiplier for average ATR (default 1.5)
        lookback: Periods to calculate average ATR

    Returns:
        True if current volatility > threshold
    """
    if current_atr is None or len(historical_closes) < lookback:
        return False

    # Calculate historical ATR values if highs/lows provided
    if highs and lows and len(highs) >= lookback and len(lows) >= lookback:
        historical_atrs: list[Decimal] = []
        for i in range(14, min(len(highs), lookback)):
            atr = calculate_atr(
                highs[:i + 1],
                lows[:i + 1],
                historical_closes[:i + 1],
                period=14,
            )
            if atr:
                historical_atrs.append(atr)

        if historical_atrs:
            avg_atr = sum(historical_atrs) / Decimal(len(historical_atrs))
            return current_atr > avg_atr * threshold_multiplier

    # Fallback: use price changes as proxy for volatility
    returns = [
        abs(historical_closes[i] - historical_closes[i - 1]) / historical_closes[i - 1]
        for i in range(1, min(len(historical_closes), lookback))
    ]

    if not returns:
        return False

    avg_volatility = sum(returns) / Decimal(len(returns))
    current_return_vol = current_atr / historical_closes[-1]

    return current_return_vol > avg_volatility * threshold_multiplier


def is_trending_market(
    adx: Decimal | None,
    adx_threshold: Decimal = Decimal("25"),
) -> bool:
    """
    Determine if market is trending.

    Args:
        adx: Current ADX value
        adx_threshold: ADX threshold for trending (default 25)

    Returns:
        True if ADX indicates strong trend
    """
    if adx is None:
        return False

    return adx > adx_threshold


def is_ranging_market(
    adx: Decimal | None,
    adx_threshold: Decimal = Decimal("20"),
) -> bool:
    """
    Determine if market is ranging.

    Args:
        adx: Current ADX value
        adx_threshold: ADX threshold for ranging (default 20)

    Returns:
        True if ADX indicates ranging/choppy market
    """
    if adx is None:
        return False

    return adx < adx_threshold


def calculate_dynamic_stop(
    entry_price: Decimal,
    atr: Decimal | None,
    atr_multiplier: Decimal = Decimal("2.0"),
    default_stop_pct: Decimal = Decimal("0.02"),
) -> Decimal:
    """
    Calculate dynamic stop loss based on ATR.

    Args:
        entry_price: Position entry price
        atr: Current ATR value
        atr_multiplier: ATR multiplier for stop distance (default 2.0)
        default_stop_pct: Default stop if ATR unavailable (default 2%)

    Returns:
        Stop loss distance as percentage of entry price
    """
    if atr is None or atr == Decimal("0"):
        return default_stop_pct

    # Calculate stop distance in price units
    stop_distance = atr * atr_multiplier

    # Convert to percentage
    stop_pct = stop_distance / entry_price

    # Cap at reasonable limits (0.5% to 5%)
    min_stop = Decimal("0.005")  # 0.5%
    max_stop = Decimal("0.05")  # 5%

    return max(min_stop, min(stop_pct, max_stop))


__all__ = [
    "calculate_atr",
    "calculate_adx",
    "calculate_rsi",
    "is_volatile_market",
    "is_trending_market",
    "is_ranging_market",
    "calculate_dynamic_stop",
]
