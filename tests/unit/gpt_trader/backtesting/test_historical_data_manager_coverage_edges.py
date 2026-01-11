"""Edge coverage for HistoricalDataManager coverage index behavior."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

from gpt_trader.backtesting.data.fetcher import CoinbaseHistoricalFetcher
from gpt_trader.backtesting.data.manager import HistoricalDataManager


def _make_manager(tmp_path: Path) -> HistoricalDataManager:
    fetcher = Mock(spec=CoinbaseHistoricalFetcher)
    return HistoricalDataManager(
        fetcher=fetcher,
        cache_dir=tmp_path,
        validate_quality=False,
    )


def test_identify_gaps_returns_full_range_when_no_coverage(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    start = datetime(2024, 1, 1, 0, 0, 0)
    end = start + timedelta(hours=1)

    assert manager._identify_gaps("BTC-USD", "ONE_HOUR", start, end) == [(start, end)]


def test_identify_gaps_returns_empty_when_range_covered(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    start = datetime(2024, 1, 1, 0, 0, 0)
    end = start + timedelta(hours=1)
    manager._coverage_index = {
        "BTC-USD": {"ONE_HOUR": [(start - timedelta(minutes=5), end + timedelta(minutes=5))]}
    }

    assert manager._identify_gaps("BTC-USD", "ONE_HOUR", start, end) == []


def test_load_coverage_index_invalid_json(tmp_path: Path) -> None:
    index_path = tmp_path / "_coverage_index.json"
    index_path.write_text("not-json")

    manager = _make_manager(tmp_path)

    assert manager._coverage_index == {}


def test_update_coverage_writes_index_file(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    start = datetime(2024, 1, 1, 0, 0, 0)
    end = start + timedelta(minutes=5)

    manager._update_coverage("BTC-USD", "ONE_MINUTE", start, end)

    data = json.loads((tmp_path / "_coverage_index.json").read_text())
    assert data["BTC-USD"]["ONE_MINUTE"] == [[start.isoformat(), end.isoformat()]]
