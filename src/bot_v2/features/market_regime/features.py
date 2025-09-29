"""
Feature engineering for market regime detection - LOCAL to this slice.
"""

import numpy as np
import pandas as pd

from .types import RegimeFeatures


def extract_regime_features(data: pd.DataFrame) -> RegimeFeatures:
    """
    Extract comprehensive features for regime detection.

    Args:
        data: DataFrame with OHLCV data

    Returns:
        RegimeFeatures object with all calculated features
    """
    close_prices = data["close"].values
    high_prices = data["high"].values
    low_prices = data["low"].values
    volume = data["volume"].values

    # Calculate returns
    returns = np.diff(close_prices) / close_prices[:-1]

    # Price features
    returns_1d = returns[-1] if len(returns) > 0 else 0
    returns_5d = (close_prices[-1] / close_prices[-6] - 1) if len(close_prices) > 5 else 0
    returns_20d = (close_prices[-1] / close_prices[-21] - 1) if len(close_prices) > 20 else 0
    returns_60d = (close_prices[-1] / close_prices[-61] - 1) if len(close_prices) > 60 else 0

    # Volatility features
    realized_vol_10d = calculate_realized_volatility(returns[-10:]) if len(returns) >= 10 else 0
    realized_vol_30d = calculate_realized_volatility(returns[-30:]) if len(returns) >= 30 else 0
    vol_of_vol = calculate_volatility_of_volatility(returns) if len(returns) >= 20 else 0

    # Trend features
    ma_5_20_spread = calculate_ma_spread(close_prices, 5, 20)
    ma_20_60_spread = calculate_ma_spread(close_prices, 20, 60)
    trend_strength = calculate_trend_strength(close_prices)

    # Market structure features
    volume_ratio = calculate_volume_ratio(volume)
    high_low_ratio = calculate_high_low_ratio(high_prices, low_prices)
    correlation_market = calculate_market_correlation(returns) if len(returns) >= 30 else 0.7

    # Risk indicators (simplified - would use real data in production)
    vix_level = estimate_vix_level(realized_vol_30d)
    put_call_ratio = None  # Would need options data
    safe_haven_flow = None  # Would need cross-asset data

    return RegimeFeatures(
        returns_1d=returns_1d,
        returns_5d=returns_5d,
        returns_20d=returns_20d,
        returns_60d=returns_60d,
        realized_vol_10d=realized_vol_10d,
        realized_vol_30d=realized_vol_30d,
        vol_of_vol=vol_of_vol,
        ma_5_20_spread=ma_5_20_spread,
        ma_20_60_spread=ma_20_60_spread,
        trend_strength=trend_strength,
        volume_ratio=volume_ratio,
        high_low_ratio=high_low_ratio,
        correlation_market=correlation_market,
        vix_level=vix_level,
        put_call_ratio=put_call_ratio,
        safe_haven_flow=safe_haven_flow,
    )


def calculate_indicators(data: pd.DataFrame) -> dict[str, float]:
    """
    Calculate additional technical indicators for regime analysis.

    Returns dictionary of indicator values.
    """
    close_prices = data["close"].values
    high_prices = data["high"].values
    low_prices = data["low"].values
    volume = data["volume"].values

    indicators = {}

    # Basic indicators
    indicators["current_price"] = close_prices[-1]
    indicators["price_change_1d"] = (
        (close_prices[-1] / close_prices[-2] - 1) if len(close_prices) > 1 else 0
    )

    # Moving averages
    if len(close_prices) >= 20:
        indicators["sma_5"] = np.mean(close_prices[-5:])
        indicators["sma_10"] = np.mean(close_prices[-10:])
        indicators["sma_20"] = np.mean(close_prices[-20:])

        # MA positioning
        indicators["price_above_sma20"] = close_prices[-1] > indicators["sma_20"]
        indicators["sma5_above_sma20"] = indicators["sma_5"] > indicators["sma_20"]

    # Volatility indicators
    if len(close_prices) >= 20:
        returns = np.diff(close_prices) / close_prices[:-1]
        indicators["volatility_20d"] = np.std(returns[-20:]) * np.sqrt(252) * 100
        indicators["volatility_percentile"] = calculate_volatility_percentile(returns)

    # Momentum indicators
    if len(close_prices) >= 14:
        indicators["rsi"] = calculate_rsi(close_prices)
        indicators["momentum_10d"] = (close_prices[-1] / close_prices[-11] - 1) * 100

    # Volume indicators
    if len(volume) >= 20:
        indicators["volume_sma20"] = np.mean(volume[-20:])
        indicators["volume_ratio"] = volume[-1] / indicators["volume_sma20"]
        indicators["volume_trend"] = calculate_volume_trend(volume)

    # Support/Resistance
    if len(close_prices) >= 20:
        indicators["distance_to_high20"] = (
            (np.max(high_prices[-20:]) - close_prices[-1]) / close_prices[-1] * 100
        )
        indicators["distance_to_low20"] = (
            (close_prices[-1] - np.min(low_prices[-20:])) / close_prices[-1] * 100
        )

    # Market structure
    indicators["daily_range"] = (high_prices[-1] - low_prices[-1]) / close_prices[-1] * 100

    return indicators


# Feature calculation functions


def calculate_realized_volatility(returns: np.ndarray, annualize: bool = True) -> float:
    """Calculate realized volatility from returns."""
    if len(returns) < 2:
        return 0.0

    vol = np.std(returns)
    if annualize:
        vol *= np.sqrt(252)  # Annualize

    return vol * 100  # Return as percentage


def calculate_volatility_of_volatility(returns: np.ndarray, window: int = 10) -> float:
    """Calculate volatility of volatility (VoV)."""
    if len(returns) < window * 2:
        return 0.0

    # Calculate rolling volatilities
    rolling_vols = []
    for i in range(window, len(returns)):
        vol = np.std(returns[i - window : i])
        rolling_vols.append(vol)

    if len(rolling_vols) < 2:
        return 0.0

    # VoV is volatility of the volatilities
    vov = np.std(rolling_vols)
    return vov


def calculate_ma_spread(prices: np.ndarray, short_period: int, long_period: int) -> float:
    """Calculate spread between two moving averages."""
    if len(prices) < long_period:
        return 0.0

    ma_short = np.mean(prices[-short_period:])
    ma_long = np.mean(prices[-long_period:])

    spread = (ma_short - ma_long) / ma_long
    return spread


def calculate_trend_strength(prices: np.ndarray, window: int = 20) -> float:
    """Calculate trend strength using linear regression slope."""
    if len(prices) < window:
        return 0.0

    recent_prices = prices[-window:]
    x = np.arange(len(recent_prices))

    # Linear regression
    slope = np.polyfit(x, recent_prices, 1)[0]

    # Normalize by average price
    avg_price = np.mean(recent_prices)
    trend_strength = (slope / avg_price) * 100 * window  # Scale by window

    return np.clip(trend_strength, -100, 100)


def calculate_volume_ratio(volume: np.ndarray, window: int = 20) -> float:
    """Calculate current volume vs average volume ratio."""
    if len(volume) < window:
        return 1.0

    current_vol = volume[-1]
    avg_vol = np.mean(volume[-window:])

    if avg_vol > 0:
        return current_vol / avg_vol
    return 1.0


def calculate_high_low_ratio(
    high_prices: np.ndarray, low_prices: np.ndarray, window: int = 20
) -> float:
    """Calculate average high-low range ratio."""
    if len(high_prices) < window:
        return 0.0

    recent_highs = high_prices[-window:]
    recent_lows = low_prices[-window:]

    ranges = (recent_highs - recent_lows) / ((recent_highs + recent_lows) / 2)
    avg_range = np.mean(ranges)

    return avg_range


def calculate_market_correlation(returns: np.ndarray, window: int = 30) -> float:
    """
    Calculate correlation with market (simplified - uses random walk).

    In production, would correlate with actual market index returns.
    """
    if len(returns) < window:
        return 0.7  # Default correlation

    # Simulate market returns (in production, use real SPY/market data)
    market_returns = np.random.normal(0.001, 0.015, len(returns))

    if len(returns) >= window:
        recent_returns = returns[-window:]
        recent_market = market_returns[-window:]

        correlation = np.corrcoef(recent_returns, recent_market)[0, 1]
        return correlation if not np.isnan(correlation) else 0.7

    return 0.7


def calculate_rsi(prices: np.ndarray, period: int = 14) -> float:
    """Calculate RSI (Relative Strength Index)."""
    if len(prices) < period + 1:
        return 50.0

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


def calculate_volatility_percentile(returns: np.ndarray, window: int = 252) -> float:
    """Calculate current volatility percentile vs historical."""
    if len(returns) < window:
        return 50.0

    # Calculate rolling volatilities
    current_vol = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)

    historical_vols = []
    for i in range(20, len(returns), 5):  # Every 5 days
        vol = np.std(returns[i - 20 : i])
        historical_vols.append(vol)

    if len(historical_vols) < 10:
        return 50.0

    # Calculate percentile
    percentile = (np.sum(np.array(historical_vols) < current_vol) / len(historical_vols)) * 100
    return percentile


def calculate_volume_trend(volume: np.ndarray, window: int = 20) -> float:
    """Calculate volume trend using linear regression."""
    if len(volume) < window:
        return 0.0

    recent_volume = volume[-window:]
    x = np.arange(len(recent_volume))

    # Linear regression on volume
    slope = np.polyfit(x, recent_volume, 1)[0]

    # Normalize by average volume
    avg_volume = np.mean(recent_volume)
    if avg_volume > 0:
        trend = (slope / avg_volume) * 100
    else:
        trend = 0.0

    return trend


def estimate_vix_level(realized_volatility: float) -> float:
    """
    Estimate VIX-like level from realized volatility.

    In production, would use actual VIX data.
    """
    # Rough approximation: VIX typically trades at premium to realized vol
    estimated_vix = realized_volatility * 1.2 + np.random.normal(0, 2)
    return max(10, min(estimated_vix, 80))  # Reasonable VIX range


def engineer_regime_features(base_features: RegimeFeatures) -> np.ndarray:
    """
    Engineer additional features from base features.

    Creates interaction terms and transformations.
    """
    engineered = []

    # Volatility regime indicators
    vol_low = 1.0 if base_features.realized_vol_30d < 15 else 0.0
    vol_medium = 1.0 if 15 <= base_features.realized_vol_30d < 25 else 0.0
    vol_high = 1.0 if base_features.realized_vol_30d >= 25 else 0.0

    engineered.extend([vol_low, vol_medium, vol_high])

    # Trend regime indicators
    trend_bull = 1.0 if base_features.returns_20d > 0.05 else 0.0
    trend_bear = 1.0 if base_features.returns_20d < -0.05 else 0.0
    trend_sideways = 1.0 if abs(base_features.returns_20d) <= 0.05 else 0.0

    engineered.extend([trend_bull, trend_bear, trend_sideways])

    # Interaction features
    vol_trend_interaction = base_features.realized_vol_30d * abs(base_features.returns_20d)
    ma_convergence = abs(base_features.ma_5_20_spread) + abs(base_features.ma_20_60_spread)
    momentum_volume = base_features.returns_5d * base_features.volume_ratio

    engineered.extend([vol_trend_interaction, ma_convergence, momentum_volume])

    # Risk stress indicators
    stress_indicator = (
        base_features.realized_vol_30d / 30 + (1 - base_features.correlation_market) / 2
    )
    regime_conflict = abs(base_features.ma_5_20_spread - base_features.ma_20_60_spread)

    engineered.extend([stress_indicator, regime_conflict])

    return np.array(engineered)
