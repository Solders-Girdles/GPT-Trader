"""Ensure DataStorage surfaces backend failures via logging."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from bot_v2.features.data.storage import DataStorage
from bot_v2.features.data.types import DataQuery, DataType


@pytest.fixture
def storage(tmp_path) -> DataStorage:
    store = DataStorage(base_path=str(tmp_path))
    return store


def test_fetch_logs_load_failures(storage: DataStorage, monkeypatch, caplog):
    query = DataQuery(
        symbols=["BTC-USD"],
        start_date=datetime.utcnow() - timedelta(days=1),
        end_date=datetime.utcnow(),
        data_type=DataType.OHLCV,
    )
    storage.index = {"BTC-USD_ohlcv": "missing.pkl"}

    def _raise(*_args, **_kwargs):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(pd, "read_pickle", _raise)

    with caplog.at_level("WARNING"):
        frame = storage.fetch(query)

    assert frame is None
    assert any("Failed to load" in message for message in caplog.messages)


def test_delete_before_logs_prune_failures(storage: DataStorage, monkeypatch, caplog):
    storage.index = {"BTC-USD_ohlcv": "missing.pkl"}

    def _raise(*_args, **_kwargs):
        raise OSError("cannot open")

    monkeypatch.setattr(pd, "read_pickle", _raise)

    with caplog.at_level("WARNING"):
        deleted = storage.delete_before(datetime.utcnow())

    assert deleted == 0
    assert any("Failed to prune" in message for message in caplog.messages)


def test_get_stats_logs_collection_failures(storage: DataStorage, monkeypatch, caplog):
    storage.index = {"BTC-USD_ohlcv": "missing.pkl"}

    def _raise(*_args, **_kwargs):
        raise OSError("stats")

    monkeypatch.setattr(pd, "read_pickle", _raise)

    with caplog.at_level("WARNING"):
        stats = storage.get_stats()

    assert stats["total_records"] == 0
    assert any("Failed to load" in message for message in caplog.messages)
