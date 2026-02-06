"""Timezone-aware datetime utilities for consistent UTC timestamp handling.

All functions return timezone-aware datetime objects to prevent naive datetime bugs.
Use these helpers instead of datetime.utcnow() or datetime.now() to ensure
timestamps in logs, metrics, caches, and persistence include timezone information.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def utc_now() -> datetime:
    """Return current UTC time as a timezone-aware datetime.

    Returns:
        datetime: Timezone-aware datetime object in UTC

    Example:
        >>> now = utc_now()
        >>> now.tzinfo is not None
        True
    """
    return datetime.now(UTC)


def utc_now_iso() -> str:
    """Return current UTC time as ISO 8601 string with +00:00 suffix.

    Returns:
        str: ISO 8601 formatted timestamp with timezone (e.g., "2024-01-15T10:30:45.123456+00:00")

    Example:
        >>> timestamp = utc_now_iso()
        >>> timestamp.endswith('+00:00')
        True
    """
    return utc_now().isoformat()


def utc_timestamp() -> float:
    """Return current UTC time as a Unix timestamp (seconds since epoch).

    Returns:
        float: Unix timestamp with microsecond precision

    Example:
        >>> ts = utc_timestamp()
        >>> ts > 1700000000  # After Nov 2023
        True
    """
    return utc_now().timestamp()


def to_iso_utc(dt: datetime) -> str:
    """Convert a datetime to ISO 8601 string, ensuring UTC timezone.

    Args:
        dt: Datetime object (naive or aware)

    Returns:
        str: ISO 8601 formatted timestamp in UTC with +00:00 suffix

    Example:
        >>> from datetime import datetime, UTC
        >>> dt = datetime(2024, 1, 15, 10, 30, 45, tzinfo=UTC)
        >>> to_iso_utc(dt)
        '2024-01-15T10:30:45+00:00'
    """
    if dt.tzinfo is None:
        # Assume naive datetimes are already in UTC
        dt = dt.replace(tzinfo=UTC)
    elif dt.tzinfo != UTC:
        # Convert to UTC if in a different timezone
        dt = dt.astimezone(UTC)
    return dt.isoformat()


def normalize_to_utc(dt: datetime) -> datetime:
    """Return a timezone-aware UTC datetime from naive or aware inputs."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    if dt.tzinfo == UTC:
        return dt
    return dt.astimezone(UTC)


def parse_iso_to_epoch(value: str) -> float:
    """Parse an ISO 8601 timestamp string to an epoch float (UTC).

    Handles trailing "Z" by converting to "+00:00" and normalizes naive
    datetimes as UTC to avoid local-time ambiguity.
    """
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("Empty timestamp")
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    dt = datetime.fromisoformat(cleaned)
    return normalize_to_utc(dt).timestamp()


def to_epoch_seconds(value: Any) -> float | None:
    """Coerce a timestamp-ish value into epoch seconds (or None).

    Args:
        value: Timestamp expressed as numeric seconds, datetime, or ISO string.

    Returns:
        Seconds since the epoch, or None if the input is missing/unsupported.
    """

    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, datetime):
        return normalize_to_utc(value).timestamp()

    if isinstance(value, str):
        try:
            return parse_iso_to_epoch(value)
        except ValueError:
            return None

    return None


def age_since_timestamp_seconds(
    value: Any,
    *,
    now_seconds: float | None = None,
    missing_value: float = float("inf"),
) -> tuple[float | None, float]:
    """Return a (timestamp, age) pair for a timestamp-ish value.

    Converts the input to epoch seconds, then returns the computed age
    using ``now_seconds`` (defaulting to :func:`utc_timestamp`).
    If the timestamp is missing or falsy (e.g., 0), the returned age
    uses ``missing_value`` and the timestamp is None.
    """

    timestamp = to_epoch_seconds(value)
    if now_seconds is None:
        now_seconds = utc_timestamp()

    if not timestamp:
        return None, missing_value

    return timestamp, now_seconds - timestamp


__all__ = [
    "utc_now",
    "utc_now_iso",
    "utc_timestamp",
    "to_iso_utc",
    "normalize_to_utc",
    "parse_iso_to_epoch",
    "to_epoch_seconds",
    "age_since_timestamp_seconds",
]
