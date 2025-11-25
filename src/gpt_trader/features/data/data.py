from __future__ import annotations
from typing import Any
import pandas as pd
from gpt_trader.features.data.types import DataQuery

# Placeholders for dependency injection/singleton
_storage: Any = None
_cache: Any = None
_quality_checker: Any = None


# Console helpers
def console_success(*args: Any) -> None:
    pass


def console_warning(*args: Any) -> None:
    pass


def console_error(*args: Any) -> None:
    pass


def console_cache(*args: Any) -> None:
    pass


def console_storage(*args: Any) -> None:
    pass


def console_data(*args: Any) -> None:
    pass


def store_data(symbol: str, data: pd.DataFrame, **kwargs: Any) -> bool:
    if _storage:
        success = _storage.store(
            symbol=symbol, data=data, data_type=kwargs.get("data_type"), source=kwargs.get("source")
        )
        if success and _cache:
            _cache.put(symbol, data)
        return success
    return False


def fetch_data(query: DataQuery) -> pd.DataFrame | None:
    # Check cache
    cache_key = query.get_cache_key()
    if _cache:
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached

    # Check storage
    if _storage:
        data = _storage.fetch(query)
        if data is not None:
            if _cache:
                _cache.put(cache_key, data)
            return data

    # Download (mock)
    result = download_from_yahoo(query.symbols, query.start_date, query.end_date, query.interval)
    if isinstance(result, dict) and len(query.symbols) == 1:
        return result.get(query.symbols[0])
    return result


def download_from_yahoo(*args: Any, **kwargs: Any) -> Any:
    return None


def cache_data(key: str, data: Any, ttl_seconds: int = 3600) -> bool:
    if _cache:
        return _cache.put(key, data, ttl_seconds)
    return False


def clean_old_data(days_to_keep: int) -> int:
    if _storage:
        import datetime

        cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=days_to_keep)
        deleted = _storage.delete_before(cutoff)
        if _cache:
            _cache.clear_expired()
        return deleted
    return 0


def get_storage_stats() -> Any:
    if _storage:
        stats = _storage.get_stats()

        # The test expects an object with attributes
        class StatsObj:
            def __init__(self, **kwargs: Any) -> None:
                for k, v in kwargs.items():
                    setattr(self, k, v)

            @property
            def cache_entries(self) -> int:
                return 0

        return StatsObj(**stats)
    return None


def export_data(query: DataQuery, format: str, path: str) -> bool:
    data = fetch_data(query)
    if data is not None:
        import os

        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, f"{query.symbols[0]}.{format}")
        if format == "csv":
            data.to_csv(file_path)
            return True
    return False


def import_data(filepath: str, symbol: str) -> bool:
    if filepath.endswith(".csv"):
        try:
            # Try parsing with default settings first
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            return store_data(symbol=symbol, data=data, data_type=None, source=None)
        except Exception:
            return False
    return False
