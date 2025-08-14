"""
Enhanced technical indicators that utilize more available data features.
Designed to work with the expanded parameter space for surprising strategies.
"""

from __future__ import annotations

import pandas as pd


def enhanced_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Enhanced RSI with volume confirmation."""
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # Volume-weighted RSI
    if "Volume" in df.columns:
        volume_ma = df["Volume"].rolling(window=period).mean()
        volume_factor = (df["Volume"] / volume_ma).clip(0.5, 2.0)
        rsi = rsi * volume_factor

    return rsi


def bollinger_bands(
    df: pd.DataFrame, period: int = 20, std_dev: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands with volume adjustment."""
    sma = df["Close"].rolling(window=period).mean()
    std = df["Close"].rolling(window=period).std()

    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)

    # Volume-adjusted bands
    if "Volume" in df.columns:
        volume_ratio = df["Volume"] / df["Volume"].rolling(window=period).mean()
        volume_factor = volume_ratio.clip(0.8, 1.2)
        upper_band = upper_band * volume_factor
        lower_band = lower_band / volume_factor

    return upper_band, sma, lower_band


def volume_price_trend(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Volume Price Trend indicator."""
    if "Volume" not in df.columns:
        return pd.Series(0, index=df.index)

    price_change = df["Close"].pct_change()
    vpt = (price_change * df["Volume"]).cumsum()
    vpt_ma = vpt.rolling(window=period).mean()

    return (vpt - vpt_ma) / vpt_ma.rolling(window=period).std()


def money_flow_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index combining price and volume."""
    if "Volume" not in df.columns:
        return pd.Series(50, index=df.index)

    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    money_flow = typical_price * df["Volume"]

    positive_flow = (
        money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=period).sum()
    )
    negative_flow = (
        money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=period).sum()
    )

    mfi = 100 - (100 / (1 + positive_flow / negative_flow))
    return mfi


def enhanced_atr(df: pd.DataFrame, period: int = 14, volume_weight: bool = True) -> pd.Series:
    """Enhanced ATR with volume weighting."""
    high_low = df["High"] - df["Low"]
    high_close = abs(df["High"] - df["Close"].shift())
    low_close = abs(df["Low"] - df["Close"].shift())

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    if volume_weight and "Volume" in df.columns:
        volume_ma = df["Volume"].rolling(window=period).mean()
        volume_factor = (df["Volume"] / volume_ma).clip(0.5, 2.0)
        true_range = true_range * volume_factor

    atr = true_range.rolling(window=period).mean()
    return atr


def time_based_filters(
    df: pd.DataFrame, day_of_week: int | None = None, month: int | None = None
) -> pd.Series:
    """Time-based filters for trading."""
    filter_mask = pd.Series(True, index=df.index)

    if day_of_week is not None:
        day_filter = df.index.dayofweek == day_of_week
        filter_mask = filter_mask & day_filter

    if month is not None:
        month_filter = df.index.month == month
        filter_mask = filter_mask & month_filter

    return filter_mask


def correlation_filter(
    df: pd.DataFrame, market_df: pd.DataFrame, lookback: int = 60, threshold: float = 0.7
) -> pd.Series:
    """Correlation filter with market."""
    if df.empty or market_df.empty:
        return pd.Series(True, index=df.index)

    # Align data
    aligned_data = pd.concat([df["Close"], market_df["Close"]], axis=1).dropna()
    aligned_data.columns = ["asset", "market"]

    # Calculate rolling correlation
    correlation = aligned_data["asset"].rolling(window=lookback).corr(aligned_data["market"])

    # Filter based on correlation threshold
    correlation_filter = correlation.abs() > threshold

    # Reindex to original index
    result = pd.Series(False, index=df.index)
    result.loc[correlation_filter.index] = correlation_filter

    return result


def regime_filter(df: pd.DataFrame, lookback: int = 200) -> pd.Series:
    """Market regime filter based on trend."""
    if len(df) < lookback:
        return pd.Series(True, index=df.index)

    # Calculate trend using multiple timeframes
    short_ma = df["Close"].rolling(window=20).mean()
    long_ma = df["Close"].rolling(window=lookback).mean()

    # Volatility regime
    volatility = df["Close"].rolling(window=20).std()
    volatility_ma = volatility.rolling(window=lookback).mean()
    low_volatility = volatility < volatility_ma * 0.8
    high_volatility = volatility > volatility_ma * 1.2

    # Trend regime
    trend_up = short_ma > long_ma
    trend_down = short_ma < long_ma

    # Combined regime
    regime_ok = (trend_up & ~high_volatility) | (trend_down & low_volatility)

    return regime_ok


def volume_breakout(df: pd.DataFrame, period: int = 20, threshold: float = 1.5) -> pd.Series:
    """Volume breakout detection."""
    if "Volume" not in df.columns:
        return pd.Series(False, index=df.index)

    volume_ma = df["Volume"].rolling(window=period).mean()
    volume_std = df["Volume"].rolling(window=period).std()

    # Volume breakout
    volume_breakout = df["Volume"] > (volume_ma + threshold * volume_std)

    # Price confirmation
    price_change = df["Close"].pct_change()
    price_confirmation = price_change.abs() > price_change.rolling(window=period).std()

    return volume_breakout & price_confirmation


def enhanced_momentum(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Enhanced momentum indicator combining price and volume."""
    # Price momentum
    price_momentum = df["Close"].pct_change(period)

    # Volume momentum
    if "Volume" in df.columns:
        volume_momentum = df["Volume"].pct_change(period)
        volume_factor = volume_momentum.clip(-0.5, 1.0) + 1.0
    else:
        volume_factor = 1.0

    # Enhanced momentum
    enhanced_momentum = price_momentum * volume_factor

    return enhanced_momentum


def volatility_regime(
    df: pd.DataFrame, short_period: int = 20, long_period: int = 100
) -> pd.Series:
    """Volatility regime detection."""
    # Calculate volatility
    returns = df["Close"].pct_change()
    short_vol = returns.rolling(window=short_period).std()
    long_vol = returns.rolling(window=long_period).std()

    # Volatility regime
    vol_ratio = short_vol / long_vol

    # Define regimes
    low_vol_regime = vol_ratio < 0.8
    (vol_ratio >= 0.8) & (vol_ratio <= 1.2)
    high_vol_regime = vol_ratio > 1.2

    # Return regime indicator (0=low, 1=normal, 2=high)
    regime = pd.Series(1, index=df.index)  # Default to normal
    regime[low_vol_regime] = 0
    regime[high_vol_regime] = 2

    return regime


def enhanced_support_resistance(df: pd.DataFrame, period: int = 20) -> tuple[pd.Series, pd.Series]:
    """Enhanced support and resistance levels."""
    # Dynamic support and resistance
    rolling_high = df["High"].rolling(window=period).max()
    rolling_low = df["Low"].rolling(window=period).min()

    # Volume-weighted levels
    if "Volume" in df.columns:
        volume_ma = df["Volume"].rolling(window=period).mean()
        volume_factor = (df["Volume"] / volume_ma).clip(0.5, 2.0)

        # Adjust levels based on volume
        resistance = rolling_high * (1 + (volume_factor - 1) * 0.1)
        support = rolling_low * (1 - (volume_factor - 1) * 0.1)
    else:
        resistance = rolling_high
        support = rolling_low

    return support, resistance


def multi_timeframe_signal(
    df: pd.DataFrame, short_period: int = 10, medium_period: int = 20, long_period: int = 50
) -> pd.Series:
    """Multi-timeframe signal generation."""
    # Calculate moving averages for different timeframes
    short_ma = df["Close"].rolling(window=short_period).mean()
    medium_ma = df["Close"].rolling(window=medium_period).mean()
    long_ma = df["Close"].rolling(window=long_period).mean()

    # Generate signals based on alignment
    bullish_alignment = (short_ma > medium_ma) & (medium_ma > long_ma)
    bearish_alignment = (short_ma < medium_ma) & (medium_ma < long_ma)

    # Signal strength based on alignment
    signal = pd.Series(0, index=df.index)
    signal[bullish_alignment] = 1
    signal[bearish_alignment] = -1

    return signal


def enhanced_breakout_detection(
    df: pd.DataFrame, lookback: int = 55, atr_period: int = 20, atr_multiplier: float = 2.0
) -> pd.Series:
    """Enhanced breakout detection with volume confirmation."""
    # Donchian channels
    upper = df["High"].rolling(window=lookback).max()
    lower = df["Low"].rolling(window=lookback).min()

    # ATR for volatility adjustment
    enhanced_atr(df, atr_period)

    # Volume confirmation
    if "Volume" in df.columns:
        volume_ma = df["Volume"].rolling(window=lookback).mean()
        volume_confirmation = df["Volume"] > volume_ma * 1.2
    else:
        volume_confirmation = pd.Series(True, index=df.index)

    # Breakout signals
    breakout_up = (df["Close"] > upper.shift(1)) & volume_confirmation
    breakout_down = (df["Close"] < lower.shift(1)) & volume_confirmation

    # Signal generation
    signal = pd.Series(0, index=df.index)
    signal[breakout_up] = 1
    signal[breakout_down] = -1

    return signal


def calculate_all_enhanced_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Calculate all enhanced indicators based on parameters."""
    indicators = pd.DataFrame(index=df.index)

    # Core indicators
    indicators["atr"] = enhanced_atr(df, params.get("atr_period", 20))

    # RSI
    if params.get("use_rsi_filter", False):
        indicators["rsi"] = enhanced_rsi(df, params.get("rsi_period", 14))

    # Bollinger Bands
    if params.get("use_bollinger_filter", False):
        upper, middle, lower = bollinger_bands(
            df, params.get("bollinger_period", 20), params.get("bollinger_std", 2.0)
        )
        indicators["bb_upper"] = upper
        indicators["bb_middle"] = middle
        indicators["bb_lower"] = lower

    # Volume indicators
    if params.get("use_volume_filter", True):
        indicators["volume_ma"] = (
            df["Volume"].rolling(window=params.get("volume_ma_period", 20)).mean()
        )
        indicators["volume_ratio"] = df["Volume"] / indicators["volume_ma"]
        indicators["volume_breakout"] = volume_breakout(
            df, params.get("volume_ma_period", 20), params.get("volume_threshold", 1.5)
        )

    # Time filters
    if params.get("use_time_filter", False):
        indicators["time_filter"] = time_based_filters(
            df, params.get("day_of_week_filter"), params.get("month_filter")
        )

    # Enhanced breakout
    indicators["breakout_signal"] = enhanced_breakout_detection(
        df,
        params.get("donchian_lookback", 55),
        params.get("atr_period", 20),
        params.get("atr_k", 2.0),
    )

    # Momentum
    indicators["momentum"] = enhanced_momentum(df, params.get("rsi_period", 14))

    # Volatility regime
    indicators["volatility_regime"] = volatility_regime(df)

    # Support/Resistance
    support, resistance = enhanced_support_resistance(df, params.get("donchian_lookback", 55))
    indicators["support"] = support
    indicators["resistance"] = resistance

    # Multi-timeframe
    indicators["multi_tf_signal"] = multi_timeframe_signal(df)

    return indicators
