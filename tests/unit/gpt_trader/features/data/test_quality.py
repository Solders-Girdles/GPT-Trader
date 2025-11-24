from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from gpt_trader.features.data.quality import DataQualityChecker


def _build_frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    data = pd.DataFrame(
        {
            "open": [100, 101, 102, 103],
            "high": [101, 102, 103, 104],
            "low": [99, 100, 101, 102],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [1000, 1100, 1200, 1300],
        },
        index=index,
    )
    return data


def test_check_quality_records_history() -> None:
    checker = DataQualityChecker()
    frame = _build_frame()
    quality = checker.check_quality(frame)

    assert 0 < quality.completeness <= 1
    assert checker.quality_history[-1] == quality


def test_check_quality_handles_empty_frame() -> None:
    checker = DataQualityChecker()
    empty = pd.DataFrame()
    quality = checker.check_quality(empty)
    assert quality.completeness == 0
    assert not checker.quality_history, "empty frames should not extend history"


def test_validate_ohlcv_detects_issues() -> None:
    checker = DataQualityChecker()
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    frame = pd.DataFrame(
        {
            "open": [100, 100, -5],
            "high": [99, 100, 101],  # First row high < low
            "low": [100, 95, 90],
            "close": [101, 97, 95],
            "volume": [1000, -10, 1200],
        },
        index=index,
    )

    issues = checker.validate_ohlcv(frame)
    assert any("High < Low" in issue for issue in issues)
    assert any("Negative open" in issue for issue in issues)
    assert any("Negative volume" in issue for issue in issues)


def test_quality_trend_returns_average() -> None:
    checker = DataQualityChecker()
    frame = _build_frame()

    # First check with current timestamps
    checker.check_quality(frame)

    # Older data to pull timeliness score down
    old_index = pd.date_range(datetime.now() - timedelta(days=90), periods=4, freq="D")
    stale_frame = frame.copy()
    stale_frame.index = old_index
    checker.check_quality(stale_frame)

    trend = checker.get_quality_trend()
    assert 0.0 <= trend["timeliness"] <= 1.0
