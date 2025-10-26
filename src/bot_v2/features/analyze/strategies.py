"""
Local strategy analysis for the analyze feature.

Complete duplication from other slices - intentional for isolation!
"""

from math import isclose

import pandas as pd

from bot_v2.features.analyze.types import StrategySignals


def analyze_with_strategies(data: pd.DataFrame) -> list[StrategySignals]:
    """
    Analyze data with multiple strategies.

    Args:
        data: OHLC data

    Returns:
        List of strategy signals
    """
    signals = []

    # Simple MA Strategy
    if ma_signal := analyze_ma_strategy(data):
        signals.append(ma_signal)

    # Momentum Strategy
    if momentum_signal := analyze_momentum_strategy(data):
        signals.append(momentum_signal)

    # Mean Reversion Strategy
    if mean_rev_signal := analyze_mean_reversion_strategy(data):
        signals.append(mean_rev_signal)

    # Volatility Strategy
    if vol_signal := analyze_volatility_strategy(data):
        signals.append(vol_signal)

    # Breakout Strategy
    if breakout_signal := analyze_breakout_strategy(data):
        signals.append(breakout_signal)

    return signals


def analyze_ma_strategy(data: pd.DataFrame, fast: int = 10, slow: int = 30) -> StrategySignals:
    """Analyze using MA crossover strategy."""
    if len(data) < slow:
        return StrategySignals(
            strategy_name="Simple MA", signal=0, confidence=0.0, reason="Insufficient data"
        )

    close = data["close"]
    fast_ma = close.rolling(window=fast).mean()
    slow_ma = close.rolling(window=slow).mean()

    # Current and previous values
    curr_fast = fast_ma.iloc[-1]
    curr_slow = slow_ma.iloc[-1]
    prev_fast = fast_ma.iloc[-2]
    prev_slow = slow_ma.iloc[-2]

    # Detect crossover
    if curr_fast > curr_slow and prev_fast <= prev_slow:
        signal = 1
        reason = f"Golden cross: {fast}MA crossed above {slow}MA"
        confidence = 0.7
    elif curr_fast < curr_slow and prev_fast >= prev_slow:
        signal = -1
        reason = f"Death cross: {fast}MA crossed below {slow}MA"
        confidence = 0.7
    else:
        signal = 0
        if isclose(curr_fast, curr_slow, rel_tol=1e-9, abs_tol=1e-9):
            reason = f"Neutral: {fast}MA equals {slow}MA"
        elif curr_fast > curr_slow:
            reason = f"Bullish: {fast}MA above {slow}MA"
        else:
            reason = f"Bearish: {fast}MA below {slow}MA"
        confidence = 0.4

    return StrategySignals(
        strategy_name="Simple MA", signal=signal, confidence=confidence, reason=reason
    )


def analyze_momentum_strategy(
    data: pd.DataFrame, lookback: int = 20, threshold: float = 0.02
) -> StrategySignals:
    """Analyze using momentum strategy."""
    if len(data) < lookback + 1:
        return StrategySignals(
            strategy_name="Momentum", signal=0, confidence=0.0, reason="Insufficient data"
        )

    close = data["close"]
    momentum = (close.iloc[-1] - close.iloc[-lookback - 1]) / close.iloc[-lookback - 1]

    if momentum > threshold:
        signal = 1
        reason = f"Strong positive momentum: {momentum:.2%}"
        if threshold > 0 and abs(momentum) <= threshold * 3:
            confidence = 0.5
        else:
            excess = max(0.0, abs(momentum) - threshold * 3)
            confidence = min(0.9, 0.5 + excess * 5)
    elif momentum < -threshold:
        signal = -1
        reason = f"Strong negative momentum: {momentum:.2%}"
        if threshold > 0 and abs(momentum) <= threshold * 3:
            confidence = 0.5
        else:
            excess = max(0.0, abs(momentum) - threshold * 3)
            confidence = min(0.9, 0.5 + excess * 5)
    else:
        signal = 0
        reason = f"Neutral momentum: {momentum:.2%}"
        confidence = 0.3

    return StrategySignals(
        strategy_name="Momentum", signal=signal, confidence=confidence, reason=reason
    )


def analyze_mean_reversion_strategy(
    data: pd.DataFrame, period: int = 20, num_std: float = 2.0
) -> StrategySignals:
    """Analyze using mean reversion strategy."""
    if len(data) < period:
        return StrategySignals(
            strategy_name="Mean Reversion", signal=0, confidence=0.0, reason="Insufficient data"
        )

    close = data["close"].iloc[-period:]
    mean = close.mean()
    std = close.std()

    upper_band = mean + (std * num_std)
    lower_band = mean - (std * num_std)
    current_price = close.iloc[-1]

    # Calculate position within bands
    band_position = (current_price - lower_band) / (upper_band - lower_band)

    if current_price <= lower_band:
        signal = 1
        reason = f"Price at lower Bollinger Band ({band_position:.1%} position)"
        confidence = 0.8
    elif current_price >= upper_band:
        signal = -1
        reason = f"Price at upper Bollinger Band ({band_position:.1%} position)"
        confidence = 0.8
    else:
        signal = 0
        reason = f"Price within Bollinger Bands ({band_position:.1%} position)"
        confidence = 0.4

    return StrategySignals(
        strategy_name="Mean Reversion", signal=signal, confidence=confidence, reason=reason
    )


def analyze_volatility_strategy(
    data: pd.DataFrame, period: int = 20, vol_threshold: float = 0.02
) -> StrategySignals:
    """Analyze using volatility strategy."""
    if len(data) < period + 1:
        return StrategySignals(
            strategy_name="Volatility", signal=0, confidence=0.0, reason="Insufficient data"
        )

    returns = data["close"].pct_change()
    current_vol = returns.iloc[-period:].std()

    # Check volatility regime
    if current_vol >= vol_threshold:
        signal = 0
        reason = f"High volatility ({current_vol:.2%}), staying out"
        confidence = 0.7
    else:
        # Low volatility - use momentum
        momentum = (data["close"].iloc[-1] - data["close"].iloc[-period]) / data["close"].iloc[
            -period
        ]

        if momentum > 0.01:
            signal = 1
            reason = f"Low volatility ({current_vol:.2%}) with positive momentum"
            confidence = 0.6
        elif momentum < -0.01:
            signal = -1
            reason = f"Low volatility ({current_vol:.2%}) with negative momentum"
            confidence = 0.6
        else:
            signal = 0
            reason = f"Low volatility ({current_vol:.2%}) but no clear direction"
            confidence = 0.3

    return StrategySignals(
        strategy_name="Volatility", signal=signal, confidence=confidence, reason=reason
    )


def analyze_breakout_strategy(data: pd.DataFrame, lookback: int = 20) -> StrategySignals:
    """Analyze using breakout strategy."""
    if len(data) < lookback + 1:
        return StrategySignals(
            strategy_name="Breakout", signal=0, confidence=0.0, reason="Insufficient data"
        )

    # Get recent high/low
    recent_high = data["high"].iloc[-lookback - 1 : -1].max()
    recent_low = data["low"].iloc[-lookback - 1 : -1].min()
    current_price = data["close"].iloc[-1]
    prev_price = data["close"].iloc[-2]

    # Check for breakout
    if current_price > recent_high and prev_price <= recent_high:
        signal = 1
        breakout_strength = (current_price - recent_high) / recent_high
        reason = f"Upward breakout above {lookback}-day high ({breakout_strength:.2%} above)"
        confidence = min(0.9, 0.6 + breakout_strength * 10)
    elif current_price < recent_low and prev_price >= recent_low:
        signal = -1
        breakout_strength = (recent_low - current_price) / recent_low
        reason = f"Downward breakout below {lookback}-day low ({breakout_strength:.2%} below)"
        confidence = min(0.9, 0.6 + breakout_strength * 10)
    else:
        # Check position within range
        range_position = (current_price - recent_low) / (recent_high - recent_low)
        signal = 0
        reason = f"Within {lookback}-day range ({range_position:.1%} position)"
        confidence = 0.3

    return StrategySignals(
        strategy_name="Breakout", signal=signal, confidence=confidence, reason=reason
    )
