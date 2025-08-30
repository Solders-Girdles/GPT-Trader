"""
Local technical indicator calculations.

Complete isolation - no external dependencies.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return data.rolling(window=period).mean()


def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return data.ewm(span=period, adjust=False).mean()


def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index.
    
    Args:
        data: Price series
        period: RSI period
        
    Returns:
        RSI series
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD indicator.
    
    Args:
        data: Price series
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
        
    Returns:
        (MACD line, Signal line, Histogram)
    """
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(data: pd.Series, period: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.
    
    Args:
        data: Price series
        period: MA period
        num_std: Number of standard deviations
        
    Returns:
        (Upper band, Middle band, Lower band)
    """
    middle = calculate_sma(data, period)
    std = data.rolling(window=period).std()
    
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    
    return upper, middle, lower


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Average True Range.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period
        
    Returns:
        ATR series
    """
    # Calculate true range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR
    atr = tr.rolling(window=period).mean()
    
    return atr


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume.
    
    Args:
        close: Close prices
        volume: Volume series
        
    Returns:
        OBV series
    """
    obv = pd.Series(index=close.index, dtype=float)
    obv.iloc[0] = volume.iloc[0]
    
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                         k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic oscillator.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        k_period: %K period
        d_period: %D period
        
    Returns:
        (%K, %D)
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_period).mean()
    
    return k, d


def identify_support_resistance(data: pd.DataFrame, lookback: int = 20) -> Tuple[float, float, float, float, float]:
    """
    Identify support and resistance levels.
    
    Args:
        data: OHLC data
        lookback: Lookback period
        
    Returns:
        (immediate_support, strong_support, immediate_resistance, strong_resistance, pivot)
    """
    recent_data = data.tail(lookback)
    
    # Calculate pivot point
    last_high = data['high'].iloc[-1]
    last_low = data['low'].iloc[-1]
    last_close = data['close'].iloc[-1]
    pivot = (last_high + last_low + last_close) / 3
    
    # Find recent highs and lows
    recent_highs = recent_data['high'].nlargest(3)
    recent_lows = recent_data['low'].nsmallest(3)
    
    # Support levels
    immediate_support = recent_lows.iloc[0] if len(recent_lows) > 0 else last_low
    strong_support = recent_lows.mean() if len(recent_lows) > 1 else immediate_support
    
    # Resistance levels
    immediate_resistance = recent_highs.iloc[0] if len(recent_highs) > 0 else last_high
    strong_resistance = recent_highs.mean() if len(recent_highs) > 1 else immediate_resistance
    
    return immediate_support, strong_support, immediate_resistance, strong_resistance, pivot


def detect_trend(data: pd.Series, short_period: int = 20, long_period: int = 50) -> str:
    """
    Detect price trend.
    
    Args:
        data: Price series
        short_period: Short MA period
        long_period: Long MA period
        
    Returns:
        'bullish', 'bearish', or 'neutral'
    """
    short_ma = calculate_sma(data, short_period)
    long_ma = calculate_sma(data, long_period)
    
    if short_ma.iloc[-1] > long_ma.iloc[-1] * 1.02:
        return 'bullish'
    elif short_ma.iloc[-1] < long_ma.iloc[-1] * 0.98:
        return 'bearish'
    else:
        return 'neutral'


def calculate_volatility(returns: pd.Series, period: int = 20) -> str:
    """
    Calculate volatility regime.
    
    Args:
        returns: Returns series
        period: Lookback period
        
    Returns:
        'low', 'medium', or 'high'
    """
    vol = returns.rolling(window=period).std()
    current_vol = vol.iloc[-1]
    
    # Annualized volatility
    annual_vol = current_vol * np.sqrt(252)
    
    if annual_vol < 0.15:
        return 'low'
    elif annual_vol < 0.25:
        return 'medium'
    else:
        return 'high'