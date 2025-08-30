"""
Market data fetching and indicator calculation - LOCAL to this slice.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional


def fetch_market_data(symbol: str, lookback_days: int) -> pd.DataFrame:
    """
    Fetch market data for analysis.
    
    LOCAL implementation - in production would call data provider.
    """
    # Generate synthetic market data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate realistic price data
    initial_price = 100
    returns = np.random.normal(0.001, 0.02, len(dates))  # 0.1% mean, 2% std daily
    prices = initial_price * np.exp(np.cumsum(returns))
    
    # Generate volume
    base_volume = 10_000_000
    volume = base_volume * np.random.uniform(0.5, 2.0, len(dates))
    
    data = pd.DataFrame({
        'date': dates,
        'open': prices * np.random.uniform(0.98, 1.02, len(dates)),
        'high': prices * np.random.uniform(1.0, 1.03, len(dates)),
        'low': prices * np.random.uniform(0.97, 1.0, len(dates)),
        'close': prices,
        'volume': volume
    })
    
    return data


def calculate_indicators(data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate technical indicators from market data.
    
    Returns dict of indicator values.
    """
    indicators = {}
    
    # Price-based indicators
    close_prices = data['close'].values
    
    # Volatility (annualized)
    returns = np.diff(close_prices) / close_prices[:-1]
    indicators['volatility'] = np.std(returns) * np.sqrt(252) * 100  # Annualized %
    
    # Trend strength (using linear regression slope)
    x = np.arange(len(close_prices))
    slope = np.polyfit(x, close_prices, 1)[0]
    trend_strength = (slope / np.mean(close_prices)) * 100 * len(close_prices)
    indicators['trend_strength'] = np.clip(trend_strength, -100, 100)
    
    # Volume ratio (current vs average)
    current_volume = data['volume'].iloc[-5:].mean()  # Last 5 days
    avg_volume = data['volume'].mean()
    indicators['volume_ratio'] = current_volume / avg_volume if avg_volume > 0 else 1.0
    
    # Price momentum (rate of change)
    if len(close_prices) >= 10:
        momentum = ((close_prices[-1] / close_prices[-10]) - 1) * 100
    else:
        momentum = 0
    indicators['momentum'] = momentum
    
    # RSI
    indicators['rsi'] = calculate_rsi(close_prices)
    
    # Bollinger Band position
    indicators['bb_position'] = calculate_bb_position(close_prices)
    
    # Moving average signals
    indicators['ma_signal'] = calculate_ma_signal(close_prices)
    
    # ATR (Average True Range)
    indicators['atr'] = calculate_atr(data)
    
    # Support/Resistance distance
    indicators['support_distance'] = calculate_support_distance(close_prices)
    indicators['resistance_distance'] = calculate_resistance_distance(close_prices)
    
    return indicators


def calculate_rsi(prices: np.ndarray, period: int = 14) -> float:
    """Calculate RSI indicator."""
    if len(prices) < period + 1:
        return 50.0  # Neutral
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_bb_position(prices: np.ndarray, period: int = 20) -> float:
    """
    Calculate position within Bollinger Bands.
    
    Returns -1 (lower band) to 1 (upper band).
    """
    if len(prices) < period:
        return 0.0
    
    recent_prices = prices[-period:]
    mean = np.mean(recent_prices)
    std = np.std(recent_prices)
    
    if std == 0:
        return 0.0
    
    upper_band = mean + (2 * std)
    lower_band = mean - (2 * std)
    current_price = prices[-1]
    
    # Normalize position to -1 to 1
    position = (current_price - lower_band) / (upper_band - lower_band) * 2 - 1
    return np.clip(position, -1, 1)


def calculate_ma_signal(prices: np.ndarray) -> float:
    """
    Calculate moving average crossover signal.
    
    Returns strength of bullish/bearish signal.
    """
    if len(prices) < 50:
        return 0.0
    
    ma_10 = np.mean(prices[-10:])
    ma_30 = np.mean(prices[-30:])
    ma_50 = np.mean(prices[-50:])
    
    # Calculate signal strength
    short_signal = (ma_10 - ma_30) / ma_30 * 100 if ma_30 > 0 else 0
    long_signal = (ma_30 - ma_50) / ma_50 * 100 if ma_50 > 0 else 0
    
    combined_signal = (short_signal + long_signal) / 2
    return np.clip(combined_signal, -10, 10)


def calculate_atr(data: pd.DataFrame, period: int = 14) -> float:
    """Calculate Average True Range."""
    if len(data) < period + 1:
        return 0.0
    
    high = data['high'].values
    low = data['low'].values
    close = data['close'].values
    
    tr_list = []
    for i in range(1, len(data)):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        tr_list.append(tr)
    
    if len(tr_list) >= period:
        atr = np.mean(tr_list[-period:])
        # Normalize by current price
        current_price = close[-1]
        return (atr / current_price) * 100 if current_price > 0 else 0
    
    return 0.0


def calculate_support_distance(prices: np.ndarray) -> float:
    """Calculate distance to nearest support level."""
    if len(prices) < 20:
        return 0.0
    
    current_price = prices[-1]
    recent_lows = []
    
    # Find local minima
    for i in range(1, len(prices) - 1):
        if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
            recent_lows.append(prices[i])
    
    if recent_lows:
        # Find nearest support below current price
        supports = [low for low in recent_lows if low < current_price]
        if supports:
            nearest_support = max(supports)
            distance = (current_price - nearest_support) / current_price * 100
            return distance
    
    # If no support found, use 10% below
    return 10.0


def calculate_resistance_distance(prices: np.ndarray) -> float:
    """Calculate distance to nearest resistance level."""
    if len(prices) < 20:
        return 0.0
    
    current_price = prices[-1]
    recent_highs = []
    
    # Find local maxima
    for i in range(1, len(prices) - 1):
        if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
            recent_highs.append(prices[i])
    
    if recent_highs:
        # Find nearest resistance above current price
        resistances = [high for high in recent_highs if high > current_price]
        if resistances:
            nearest_resistance = min(resistances)
            distance = (nearest_resistance - current_price) / current_price * 100
            return distance
    
    # If no resistance found, use 10% above
    return 10.0