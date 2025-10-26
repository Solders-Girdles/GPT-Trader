from __future__ import annotations

from datetime import datetime, timezone


def ensure_utc_aware(dt: datetime) -> datetime:
    """Convert naive datetimes to UTC-aware and normalize aware datetimes to UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def utc_now() -> datetime:
    """Return current UTC time as an aware datetime."""
    return datetime.now(tz=timezone.utc)


__all__ = ["ensure_utc_aware", "utc_now"]
