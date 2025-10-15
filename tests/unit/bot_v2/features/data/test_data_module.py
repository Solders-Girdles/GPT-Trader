from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

import bot_v2.features.data.data as data_module

from bot_v2.features.data.types import DataQuery, DataSource, DataType


class StorageStub:
    def __init__(self) -> None:
        self.store_calls: list[dict[str, object]] = []
        self.fetch_return: pd.DataFrame | None = None
        self.delete_before_return = 0
        self.last_fetch_query: DataQuery | None = None
        self.delete_cutoff: datetime | None = None

    def store(
        self,
        *,
        symbol: str,
        data: pd.DataFrame,
        data_type: DataType,
        source: DataSource,
        metadata: dict | None = None,
    ) -> bool:
        self.store_calls.append(
            {
                "symbol": symbol,
                "data": data,
                "data_type": data_type,
                "source": source,
                "metadata": metadata,
            }
        )
        return True

    def fetch(self, query: DataQuery) -> pd.DataFrame | None:
        self.last_fetch_query = query
        return self.fetch_return

    def delete_before(self, cutoff: datetime) -> int:
        self.delete_cutoff = cutoff
        return self.delete_before_return

    def get_stats(self) -> dict[str, object]:
        now = datetime.utcnow()
        return {
            "total_records": 10,
            "total_size_mb": 1.5,
            "oldest_record": now - timedelta(days=10),
            "newest_record": now,
            "symbols_count": 3,
        }


class CacheStub:
    def __init__(self) -> None:
        self.data: dict[str, pd.DataFrame] = {}
        self.put_calls: list[tuple[str, int | None]] = []
        self.total_hits = 0
        self.total_misses = 0
        self._clear_expired_called = False

    def put(self, key: str, data: pd.DataFrame, ttl_seconds: int = 3600) -> bool:
        self.data[key] = data.copy()
        self.put_calls.append((key, ttl_seconds))
        return True

    def get(self, key: str) -> pd.DataFrame | None:
        return self.data.get(key)

    def clear_expired(self) -> int:
        self._clear_expired_called = True
        cleared = len(self.data)
        self.data.clear()
        return cleared

    def get_stats(self) -> dict[str, object]:
        return {
            "entries": len(self.data),
            "size_mb": 0.1,
            "hit_rate": 0.5,
            "total_hits": 5,
            "total_misses": 5,
            "expired_entries": 0,
        }


class QualityStub:
    def __init__(self) -> None:
        self.acceptable = True
        self.calls: list[pd.DataFrame] = []

    class _Result:
        def __init__(self, acceptable: bool) -> None:
            self._acceptable = acceptable

        def overall_score(self) -> float:
            return 0.42

        def is_acceptable(self, threshold: float = 0.8) -> bool:
            return self._acceptable

    def check_quality(self, data: pd.DataFrame) -> _Result:
        self.calls.append(data)
        return self._Result(self.acceptable)


@pytest.fixture
def data_stubs(monkeypatch: pytest.MonkeyPatch):
    storage = StorageStub()
    cache = CacheStub()
    quality = QualityStub()

    monkeypatch.setattr(data_module, "_storage", storage)
    monkeypatch.setattr(data_module, "_cache", cache)
    monkeypatch.setattr(data_module, "_quality_checker", quality)

    # Silence console helpers
    for attr in (
        "console_success",
        "console_warning",
        "console_error",
        "console_cache",
        "console_storage",
        "console_data",
    ):
        monkeypatch.setattr(data_module, attr, lambda *_, **__: None)

    return {"storage": storage, "cache": cache, "quality": quality}


def _make_frame(days: int = 3) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=days, freq="D")
    return pd.DataFrame(
        {
            "open": [100 + i for i in range(days)],
            "high": [101 + i for i in range(days)],
            "low": [99 + i for i in range(days)],
            "close": [100.5 + i for i in range(days)],
            "volume": [1000 + 10 * i for i in range(days)],
        },
        index=index,
    )


def test_store_data_updates_cache_with_warning(data_stubs) -> None:
    quality: QualityStub = data_stubs["quality"]
    quality.acceptable = False

    frame = _make_frame()
    assert data_module.store_data("BTC-USD", frame)

    storage: StorageStub = data_stubs["storage"]
    cache: CacheStub = data_stubs["cache"]

    assert storage.store_calls[0]["symbol"] == "BTC-USD"
    assert cache.put_calls, "cache should be updated"


def test_fetch_data_returns_cache_hit(data_stubs) -> None:
    cache: CacheStub = data_stubs["cache"]
    frame = _make_frame()
    query = DataQuery(
        symbols=["ETH-USD"],
        start_date=frame.index.min(),
        end_date=frame.index.max(),
        data_type=DataType.OHLCV,
        source=DataSource.YAHOO,
    )
    cache.put(query.get_cache_key(), frame)

    result = data_module.fetch_data(query)
    assert result is not None
    pd.testing.assert_frame_equal(result, frame)


def test_fetch_data_loads_from_storage_and_updates_cache(data_stubs) -> None:
    storage: StorageStub = data_stubs["storage"]
    cache: CacheStub = data_stubs["cache"]

    frame = _make_frame()
    storage.fetch_return = frame

    query = DataQuery(
        symbols=["BTC-USD"],
        start_date=frame.index.min(),
        end_date=frame.index.max(),
        data_type=DataType.OHLCV,
    )
    result = data_module.fetch_data(query)

    assert result is not None
    pd.testing.assert_frame_equal(result, frame)
    assert cache.put_calls, "storage results should populate cache"
    assert storage.last_fetch_query is query


def test_fetch_data_downloads_when_storage_misses(
    data_stubs, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage: StorageStub = data_stubs["storage"]
    storage.fetch_return = None

    downloaded_frame = _make_frame()

    def _download_stub(symbols, start, end, interval):
        return {symbol: downloaded_frame for symbol in symbols}

    monkeypatch.setattr(data_module, "download_from_yahoo", _download_stub)

    query = DataQuery(
        symbols=["SOL-USD"],
        start_date=downloaded_frame.index.min(),
        end_date=downloaded_frame.index.max(),
        data_type=DataType.OHLCV,
        source=DataSource.YAHOO,
    )

    result = data_module.fetch_data(query)

    assert result is not None
    pd.testing.assert_frame_equal(result, downloaded_frame)


def test_cache_data_delegates_to_cache(data_stubs) -> None:
    cache: CacheStub = data_stubs["cache"]
    frame = _make_frame()
    assert data_module.cache_data("key", frame, ttl_seconds=30)
    assert cache.put_calls[0] == ("key", 30)


def test_clean_old_data_clears_cache(data_stubs) -> None:
    storage: StorageStub = data_stubs["storage"]
    storage.delete_before_return = 7

    deleted = data_module.clean_old_data(days_to_keep=30)
    assert deleted == 7
    assert data_stubs["cache"]._clear_expired_called


def test_get_storage_stats_combines_sources(data_stubs) -> None:
    stats = data_module.get_storage_stats()
    assert stats.total_records == 10
    assert stats.cache_entries == 0
    assert stats.symbols_count == 3


def test_export_data_writes_csv(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _make_frame()

    monkeypatch.setattr(
        data_module,
        "fetch_data",
        lambda query, use_cache=True: frame,
    )

    query = DataQuery(
        symbols=["BTC-USD"],
        start_date=frame.index.min(),
        end_date=frame.index.max(),
    )

    export_dir = tmp_path / "exports"
    success = data_module.export_data(query, format="csv", path=str(export_dir))
    assert success

    exported_files = list(export_dir.glob("*.csv"))
    assert exported_files, "CSV export should create a file"


def test_import_data_roundtrips_csv(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _make_frame()
    filepath = tmp_path / "input.csv"
    frame.to_csv(filepath)

    store_calls: list[dict[str, object]] = []

    def _store_stub(**kwargs):
        store_calls.append(kwargs)
        return True

    monkeypatch.setattr(data_module, "store_data", _store_stub)

    assert data_module.import_data(str(filepath), symbol="BTC-USD")
    assert store_calls, "store_data should be called for imported data"


def test_import_data_rejects_unknown_format(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data_module, "store_data", lambda *_, **__: True)
    assert not data_module.import_data("data.unsupported", symbol="BTC-USD")
