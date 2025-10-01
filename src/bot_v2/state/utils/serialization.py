"""Shared serialization utilities for state management.

Consolidates common serialization patterns used across checkpoint, backup, and recovery systems.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any


def serialize_datetime(dt: datetime) -> str:
    """Serialize datetime to ISO format string."""
    return dt.isoformat()


def deserialize_datetime(dt_str: str) -> datetime:
    """Deserialize ISO format string to datetime."""
    return datetime.fromisoformat(dt_str)


def serialize_decimal(value: Decimal) -> str:
    """Serialize Decimal to string."""
    return str(value)


def serialize_enum(value: Enum) -> str:
    """Serialize Enum to its value."""
    return value.value


def calculate_data_hash(data: dict[str, Any]) -> str:
    """Calculate SHA-256 hash of dictionary data.

    Args:
        data: Dictionary to hash

    Returns:
        Hex string of SHA-256 hash
    """
    data_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(data_str.encode()).hexdigest()


def compress_data(data: bytes, compression_level: int = 6) -> bytes:
    """Compress data using gzip.

    Args:
        data: Raw bytes to compress
        compression_level: Compression level (1-9, default 6)

    Returns:
        Compressed bytes
    """
    return gzip.compress(data, compresslevel=compression_level)


def decompress_data(compressed: bytes) -> bytes:
    """Decompress gzip compressed data.

    Args:
        compressed: Compressed bytes

    Returns:
        Decompressed bytes
    """
    return gzip.decompress(compressed)


class SerializableMixin:
    """Mixin to add serialization methods to dataclasses.

    Example:
        @dataclass
        class MyModel(SerializableMixin):
            id: str
            timestamp: datetime
            status: StatusEnum
            amount: Decimal
    """

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Handles datetime, Decimal, and Enum types automatically.
        """
        result: dict[str, Any] = {}

        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue  # Skip private attributes

            if isinstance(value, datetime):
                result[key] = serialize_datetime(value)
            elif isinstance(value, Decimal):
                result[key] = serialize_decimal(value)
            elif isinstance(value, Enum):
                result[key] = serialize_enum(value)
            elif isinstance(value, dict):
                # Recursively handle nested dicts
                result[key] = {
                    k: serialize_datetime(v) if isinstance(v, datetime) else v
                    for k, v in value.items()
                }
            elif isinstance(value, list):
                # Handle lists of serializable objects
                result[key] = [
                    item.to_dict() if hasattr(item, "to_dict") else item for item in value
                ]
            else:
                result[key] = value

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Any:
        """Create instance from dictionary.

        Note: Subclasses should override this to handle their specific types.
        This base implementation provides minimal support.
        """
        # Get field types from class annotations
        annotations = getattr(cls, "__annotations__", {})

        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key in annotations:
                field_type = annotations[key]

                # Handle datetime fields
                if field_type == datetime or (
                    hasattr(field_type, "__origin__")
                    and datetime in getattr(field_type, "__args__", ())
                ):
                    if isinstance(value, str):
                        kwargs[key] = deserialize_datetime(value)
                    else:
                        kwargs[key] = value
                else:
                    kwargs[key] = value
            else:
                kwargs[key] = value

        return cls(**kwargs)


def serialize_to_json(data: dict[str, Any], sort_keys: bool = True) -> bytes:
    """Serialize dictionary to JSON bytes.

    Args:
        data: Dictionary to serialize
        sort_keys: Whether to sort keys (default True for consistency)

    Returns:
        JSON encoded bytes
    """
    return json.dumps(data, default=str, sort_keys=sort_keys).encode("utf-8")


def deserialize_from_json(data: bytes) -> dict[str, Any]:
    """Deserialize JSON bytes to dictionary.

    Args:
        data: JSON encoded bytes

    Returns:
        Deserialized dictionary
    """
    return json.loads(data.decode("utf-8"))


def prepare_compressed_payload(
    data: dict[str, Any], enable_compression: bool = True, compression_level: int = 6
) -> tuple[bytes, int, int]:
    """Prepare data for storage with optional compression.

    Args:
        data: Dictionary to serialize
        enable_compression: Whether to compress (default True)
        compression_level: Gzip compression level (1-9, default 6)

    Returns:
        Tuple of (payload_bytes, original_size, compressed_size)
    """
    serialized = serialize_to_json(data)
    original_size = len(serialized)

    if enable_compression:
        compressed = compress_data(serialized, compression_level)
        return compressed, original_size, len(compressed)
    else:
        return serialized, original_size, original_size
