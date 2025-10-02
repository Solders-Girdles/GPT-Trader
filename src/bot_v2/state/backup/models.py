"""Core models and configuration for backup operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class BackupType(Enum):
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"
    EMERGENCY = "emergency"


class BackupStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"
    CORRUPTED = "corrupted"


class StorageTier(Enum):
    LOCAL = "local"
    NETWORK = "network"
    CLOUD = "cloud"
    ARCHIVE = "archive"


@dataclass
class BackupMetadata:
    backup_id: str
    backup_type: BackupType
    timestamp: datetime
    size_bytes: int
    size_compressed: int
    checksum: str
    encryption_key_id: str | None
    storage_tier: StorageTier
    retention_days: int
    status: BackupStatus
    source_system: str = "bot_v2"
    backup_duration_seconds: float = 0.0
    data_sources: list[str] = field(default_factory=list)
    error_message: str | None = None
    verification_status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backup_id": self.backup_id,
            "backup_type": self.backup_type.value,
            "timestamp": self.timestamp.isoformat(),
            "size_bytes": self.size_bytes,
            "size_compressed": self.size_compressed,
            "checksum": self.checksum,
            "encryption_key_id": self.encryption_key_id,
            "storage_tier": self.storage_tier.value,
            "retention_days": self.retention_days,
            "status": self.status.value,
            "source_system": self.source_system,
            "backup_duration_seconds": self.backup_duration_seconds,
            "data_sources": self.data_sources,
            "error_message": self.error_message,
            "verification_status": self.verification_status,
        }


@dataclass
class BackupContext:
    """Shared mutable state for backup operations."""

    backup_history: list[BackupMetadata] = field(default_factory=list)
    backup_metadata: dict[str, BackupMetadata] = field(default_factory=dict)
    last_full_backup: datetime | None = None
    last_differential_backup: datetime | None = None
    last_full_state: dict[str, Any] | None = None
    last_backup_state: dict[str, Any] | None = None
    last_restored_payload: dict[str, Any] | None = None


@dataclass
class BackupConfig:
    backup_dir: str = "/tmp/bot_v2/backups"
    enable_encryption: bool = True
    enable_compression: bool = True

    full_backup_interval_hours: int = 24
    incremental_backup_interval_minutes: int = 15
    differential_backup_interval_hours: int = 6

    retention_full: int = 90
    retention_differential: int = 30
    retention_incremental: int = 7
    retention_emergency: int = 180

    local_storage_path: str = "/tmp/bot_v2/backups/local"
    network_storage_path: str | None = None
    s3_bucket: str | None = "bot-v2-backups"
    s3_region: str = "us-east-1"

    compression_level: int = 6
    parallel_uploads: bool = True
    chunk_size_mb: int = 5

    verify_after_backup: bool = True
    test_restore_frequency_days: int = 7
