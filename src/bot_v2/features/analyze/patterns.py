"""
Local pattern detection for market analysis.

Complete isolation - no external dependencies.
"""

import numpy as np
import pandas as pd

from bot_v2.features.analyze.types import PricePattern


def detect_patterns(data: pd.DataFrame) -> list[PricePattern]:
    """
    Detect price patterns in data.

    Args:
        data: OHLC data

    Returns:
        List of detected patterns
    """
    patterns = []

    # Detect various patterns
    if double_top := detect_double_top(data):
        patterns.append(double_top)

    if double_bottom := detect_double_bottom(data):
        patterns.append(double_bottom)

    if head_shoulders := detect_head_shoulders(data):
        patterns.append(head_shoulders)

    if triangle := detect_triangle(data):
        patterns.append(triangle)

    if flag := detect_flag(data):
        patterns.append(flag)

    return patterns


def detect_double_top(data: pd.DataFrame, lookback: int = 20) -> PricePattern | None:
    """
    Detect double top pattern.

    Args:
        data: OHLC data
        lookback: Period to analyze

    Returns:
        Pattern if detected, None otherwise
    """
    if len(data) < lookback:
        return None

    recent = data.tail(lookback)
    highs = recent["high"]

    # Find two peaks
    peaks = []
    for i in range(1, len(highs) - 1):
        if highs.iloc[i] > highs.iloc[i - 1] and highs.iloc[i] > highs.iloc[i + 1]:
            peaks.append((i, highs.iloc[i]))

    if len(peaks) >= 2:
        # Check if peaks are similar height
        peak1, peak2 = peaks[-2], peaks[-1]
        if abs(peak1[1] - peak2[1]) / peak1[1] < 0.03:  # Within 3%
            # Found double top
            neckline = recent["low"].iloc[peak1[0] : peak2[0] + 1].min()
            target = neckline - (peak2[1] - neckline)

            return PricePattern(
                pattern_type="Double Top",
                confidence=0.7,
                target_price=target,
                stop_loss=peak2[1] * 1.02,
                description="Bearish reversal pattern detected",
            )

    return None


def detect_double_bottom(data: pd.DataFrame, lookback: int = 20) -> PricePattern | None:
    """
    Detect double bottom pattern.

    Args:
        data: OHLC data
        lookback: Period to analyze

    Returns:
        Pattern if detected, None otherwise
    """
    if len(data) < lookback:
        return None

    recent = data.tail(lookback)
    lows = recent["low"]

    # Find two troughs
    troughs = []
    for i in range(1, len(lows) - 1):
        if lows.iloc[i] < lows.iloc[i - 1] and lows.iloc[i] < lows.iloc[i + 1]:
            troughs.append((i, lows.iloc[i]))

    if len(troughs) >= 2:
        # Check if troughs are similar depth
        trough1, trough2 = troughs[-2], troughs[-1]
        if abs(trough1[1] - trough2[1]) / trough1[1] < 0.03:  # Within 3%
            # Found double bottom
            neckline = recent["high"].iloc[trough1[0] : trough2[0] + 1].max()
            target = neckline + (neckline - trough2[1])

            return PricePattern(
                pattern_type="Double Bottom",
                confidence=0.7,
                target_price=target,
                stop_loss=trough2[1] * 0.98,
                description="Bullish reversal pattern detected",
            )

    return None


def detect_head_shoulders(data: pd.DataFrame, lookback: int = 30) -> PricePattern | None:
    """
    Detect head and shoulders pattern.

    Args:
        data: OHLC data
        lookback: Period to analyze

    Returns:
        Pattern if detected, None otherwise
    """
    if len(data) < lookback:
        return None

    recent = data.tail(lookback)
    highs = recent["high"]

    # Find three peaks
    peaks = []
    for i in range(2, len(highs) - 2):
        if (
            highs.iloc[i] > highs.iloc[i - 1]
            and highs.iloc[i] > highs.iloc[i + 1]
            and highs.iloc[i] > highs.iloc[i - 2]
            and highs.iloc[i] > highs.iloc[i + 2]
        ):
            peaks.append((i, highs.iloc[i]))

    if len(peaks) >= 3:
        # Check for head and shoulders pattern
        left_shoulder = peaks[-3]
        head = peaks[-2]
        right_shoulder = peaks[-1]

        # Head should be highest
        if head[1] > left_shoulder[1] and head[1] > right_shoulder[1]:
            # Shoulders should be similar height
            if abs(left_shoulder[1] - right_shoulder[1]) / left_shoulder[1] < 0.05:
                # Found head and shoulders
                neckline = recent["low"].iloc[left_shoulder[0] : right_shoulder[0] + 1].mean()
                target = neckline - (head[1] - neckline)

                return PricePattern(
                    pattern_type="Head and Shoulders",
                    confidence=0.8,
                    target_price=target,
                    stop_loss=head[1] * 1.02,
                    description="Strong bearish reversal pattern",
                )

    return None


def detect_triangle(data: pd.DataFrame, lookback: int = 20) -> PricePattern | None:
    """
    Detect triangle pattern (ascending, descending, or symmetrical).

    Args:
        data: OHLC data
        lookback: Period to analyze

    Returns:
        Pattern if detected, None otherwise
    """
    if len(data) < lookback:
        return None

    recent = data.tail(lookback)

    # Calculate trendlines
    highs = recent["high"].to_numpy(dtype=float)
    lows = recent["low"].to_numpy(dtype=float)
    x = np.arange(len(highs), dtype=float)

    # Fit lines to highs and lows
    high_slope = np.polyfit(x, highs, 1)[0]
    low_slope = np.polyfit(x, lows, 1)[0]

    # Classify triangle type
    if high_slope > 0.001 and abs(low_slope) < 0.001:
        pattern_type = "Ascending Triangle"
        description = "Bullish continuation pattern"
        target_mult = 1.05
    elif low_slope < -0.001 and abs(high_slope) < 0.001:
        pattern_type = "Descending Triangle"
        description = "Bearish continuation pattern"
        target_mult = 0.95
    elif abs(high_slope + low_slope) < 0.001:
        pattern_type = "Symmetrical Triangle"
        description = "Continuation pattern"
        target_mult = 1.03 if recent["close"].iloc[-1] > recent["close"].iloc[0] else 0.97
    else:
        return None

    current_price = recent["close"].iloc[-1]

    return PricePattern(
        pattern_type=pattern_type,
        confidence=0.6,
        target_price=current_price * target_mult,
        stop_loss=current_price * (0.98 if target_mult > 1 else 1.02),
        description=description,
    )


def detect_flag(data: pd.DataFrame, lookback: int = 15) -> PricePattern | None:
    """
    Detect flag pattern.

    Args:
        data: OHLC data
        lookback: Period to analyze

    Returns:
        Pattern if detected, None otherwise
    """
    if len(data) < lookback + 10:
        return None

    # Look for strong move followed by consolidation
    pre_flag = data.iloc[-lookback - 10 : -lookback]
    flag = data.tail(lookback)

    # Calculate move before flag
    pre_move = (pre_flag["close"].iloc[-1] - pre_flag["close"].iloc[0]) / pre_flag["close"].iloc[0]

    # Check for strong move (> 5%)
    if abs(pre_move) > 0.05:
        # Check for consolidation in flag
        flag_range = (flag["high"].max() - flag["low"].min()) / flag["close"].mean()

        if flag_range < 0.03:  # Tight consolidation
            # Determine direction
            if pre_move > 0:
                pattern_type = "Bull Flag"
                description = "Bullish continuation pattern"
                target = flag["close"].iloc[-1] * (1 + abs(pre_move))
                stop = flag["low"].min()
            else:
                pattern_type = "Bear Flag"
                description = "Bearish continuation pattern"
                target = flag["close"].iloc[-1] * (1 - abs(pre_move))
                stop = flag["high"].max()

            return PricePattern(
                pattern_type=pattern_type,
                confidence=0.65,
                target_price=target,
                stop_loss=stop,
                description=description,
            )

    return None
