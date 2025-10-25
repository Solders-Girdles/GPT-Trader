"""Registry implementations supporting risk manager state tracking."""

from __future__ import annotations

from collections.abc import Callable, Iterator, MutableMapping
from datetime import datetime
from threading import RLock

from bot_v2.utilities.datetime_helpers import normalize_to_utc, utc_now


class MarkTimestampRegistry(MutableMapping[str, datetime | None]):
    """Thread-safe registry for tracking latest mark timestamps per symbol."""

    def __init__(self, now_provider: Callable[[], datetime] | None = None) -> None:
        self._data: dict[str, datetime | None] = {}
        self._lock = RLock()
        self._now_provider = now_provider or utc_now

    def __getitem__(self, key: str) -> datetime | None:
        with self._lock:
            return self._data[key]

    def __setitem__(self, key: str, value: datetime | None) -> None:
        with self._lock:
            self._data[key] = value

    def __delitem__(self, key: str) -> None:
        with self._lock:
            del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._data

    def keys(self) -> tuple[str, ...]:  # type: ignore[override]
        with self._lock:
            return tuple(self._data.keys())

    def items(self) -> tuple[tuple[str, datetime | None], ...]:  # type: ignore[override]
        with self._lock:
            return tuple(self._data.items())

    def values(self) -> tuple[datetime | None, ...]:  # type: ignore[override]
        with self._lock:
            return tuple(self._data.values())

    def get(self, key: str, default: datetime | None = None) -> datetime | None:  # type: ignore[override]
        with self._lock:
            return self._data.get(key, default)

    def clear(self) -> None:  # type: ignore[override]
        with self._lock:
            self._data.clear()

    def snapshot(self) -> dict[str, datetime | None]:
        """Return a shallow copy of the registry."""
        with self._lock:
            return dict(self._data)

    def update_timestamp(self, symbol: str, timestamp: datetime | None = None) -> datetime:
        """Record the latest mark timestamp for a symbol."""
        ts_source = timestamp or self._now_provider()
        normalized = normalize_to_utc(ts_source)
        with self._lock:
            self._data[symbol] = normalized
        return normalized


__all__ = ["MarkTimestampRegistry"]
