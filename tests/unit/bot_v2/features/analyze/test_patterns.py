from __future__ import annotations

import numpy as np
import pandas as pd

from bot_v2.features.analyze import patterns


def make_ohlc(high: list[float], low: list[float], close: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"high": high, "low": low, "close": close})


def test_detect_double_top_identifies_pattern():
    data = make_ohlc(
        high=[10, 15, 13, 15.2, 12, 11],
        low=[9, 13, 12, 13.5, 11, 10],
        close=[9.5, 14, 12.5, 14.8, 11.5, 10.5],
    )

    pattern = patterns.detect_double_top(data, lookback=6)

    assert pattern is not None
    assert pattern.pattern_type == "Double Top"
    assert pattern.target_price is not None
    assert pattern.stop_loss is not None


def test_detect_double_bottom_identifies_pattern():
    data = make_ohlc(
        high=[12, 11, 12, 11.5, 13, 12.5],
        low=[9, 7, 9.5, 7.1, 9.2, 9],
        close=[10, 8, 10, 8.1, 10.5, 10.2],
    )

    pattern = patterns.detect_double_bottom(data, lookback=6)

    assert pattern is not None
    assert pattern.pattern_type == "Double Bottom"


def test_detect_head_shoulders_returns_none_for_insufficient_data():
    data = make_ohlc(high=[10, 11, 12, 11.5, 11.8], low=[9, 9.5, 9.8, 9.4, 9.6], close=[9.5] * 5)

    assert patterns.detect_head_shoulders(data, lookback=5) is None


def test_detect_triangle_classifies_ascending_pattern():
    high = np.linspace(10, 12, 8)
    low = np.full(8, 9.0)
    close = np.linspace(9.5, 11.5, 8)
    data = make_ohlc(high.tolist(), low.tolist(), close.tolist())

    pattern = patterns.detect_triangle(data, lookback=8)

    assert pattern is not None
    assert pattern.pattern_type == "Ascending Triangle"
    assert pattern.target_price is not None


def test_detect_triangle_returns_none_when_slopes_inconsistent():
    high = np.linspace(10, 10.7, 8)
    low = np.linspace(9.2, 9.8, 8)
    close = np.linspace(9.5, 9.0, 8)
    data = make_ohlc(high.tolist(), low.tolist(), close.tolist())

    assert patterns.detect_triangle(data, lookback=8) is None


def test_detect_flag_identifies_bullish_flag():
    pre_flag = np.linspace(100, 110, 10)
    flag_close = np.linspace(110, 112, 15)
    close = np.concatenate([pre_flag, flag_close])
    high = close + 0.2
    low = close - 0.2
    data = make_ohlc(high.tolist(), low.tolist(), close.tolist())

    pattern = patterns.detect_flag(data, lookback=15)

    assert pattern is not None
    assert pattern.pattern_type == "Bull Flag"
    assert pattern.target_price and pattern.target_price > flag_close[-1]


def test_detect_flag_returns_none_when_insufficient_length():
    data = make_ohlc([10, 11, 12], [9, 9.5, 10], [10, 10.5, 11])

    assert patterns.detect_flag(data, lookback=5) is None


def test_detect_patterns_aggregates_results():
    filler = make_ohlc([10] * 15, [9] * 15, [9.5] * 15)
    pattern_segment = make_ohlc(
        high=[10, 15, 13, 15.2, 12, 11],
        low=[9, 13, 12, 13.5, 11, 10],
        close=[9.5, 14, 12.5, 14.8, 11.5, 10.5],
    )
    data = pd.concat([filler, pattern_segment], ignore_index=True)

    detected = patterns.detect_patterns(data)

    assert any(p.pattern_type == "Double Top" for p in detected)
