"""Shared utilities for state management subsystems."""

from bot_v2.state.utils.adapters import (
    DefaultPostgresAdapter,
    DefaultRedisAdapter,
    DefaultS3Adapter,
    PostgresAdapter,
    RedisAdapter,
    S3Adapter,
)
from bot_v2.state.utils.serialization import (
    SerializableMixin,
    calculate_data_hash,
    compress_data,
    decompress_data,
    deserialize_datetime,
    deserialize_from_json,
    prepare_compressed_payload,
    serialize_datetime,
    serialize_decimal,
    serialize_enum,
    serialize_to_json,
)
from bot_v2.state.utils.storage import (
    AtomicFileStorage,
    ensure_directory,
    get_file_age_seconds,
)

__all__ = [
    # Adapters
    "RedisAdapter",
    "PostgresAdapter",
    "S3Adapter",
    "DefaultRedisAdapter",
    "DefaultPostgresAdapter",
    "DefaultS3Adapter",
    # Serialization
    "SerializableMixin",
    "serialize_datetime",
    "deserialize_datetime",
    "serialize_decimal",
    "serialize_enum",
    "calculate_data_hash",
    "compress_data",
    "decompress_data",
    "serialize_to_json",
    "deserialize_from_json",
    "prepare_compressed_payload",
    # Storage
    "AtomicFileStorage",
    "ensure_directory",
    "get_file_age_seconds",
]
