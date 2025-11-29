"""
Persistence layer for GPT-Trader.

Provides durable storage for:
- Events (telemetry, metrics, errors)
- Orders (state tracking for crash recovery)
- Durability utilities (atomic writes, checksums)
"""

from gpt_trader.persistence.database import DatabaseEngine
from gpt_trader.persistence.durability import (
    CorruptionError,
    PersistenceError,
    RecoveryError,
    WriteError,
    WriteResult,
    atomic_write_file,
    atomic_write_json,
    check_sqlite_integrity,
    compute_checksum,
    read_json_with_checksum,
    repair_sqlite_database,
    verify_checksum,
)
from gpt_trader.persistence.event_store import EventStore
from gpt_trader.persistence.orders_store import OrderRecord, OrdersStore, OrderStatus

__all__ = [
    # Core stores
    "DatabaseEngine",
    "EventStore",
    "OrderRecord",
    "OrdersStore",
    "OrderStatus",
    # Durability utilities
    "CorruptionError",
    "PersistenceError",
    "RecoveryError",
    "WriteError",
    "WriteResult",
    "atomic_write_file",
    "atomic_write_json",
    "check_sqlite_integrity",
    "compute_checksum",
    "read_json_with_checksum",
    "repair_sqlite_database",
    "verify_checksum",
]
