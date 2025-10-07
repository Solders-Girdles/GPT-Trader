"""Backup metadata management."""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from bot_v2.state.backup.models import (
    BackupConfig,
    BackupContext,
    BackupMetadata,
    BackupStatus,
    BackupType,
    StorageTier,
)

logger = logging.getLogger(__name__)


class BackupMetadataManager:
    """Manages backup metadata persistence and history tracking."""

    def __init__(self, config: BackupConfig, context: BackupContext):
        self.config = config
        self.context = context

    def load_history(self) -> None:
        """Load backup history from metadata files."""
        try:
            metadata_dir = Path(self.config.backup_dir)

            for meta_file in metadata_dir.glob("*.meta"):
                with open(meta_file) as f:
                    data = json.load(f)

                    metadata = BackupMetadata(
                        backup_id=data["backup_id"],
                        backup_type=BackupType(data["backup_type"]),
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        size_bytes=data["size_bytes"],
                        size_compressed=data["size_compressed"],
                        checksum=data["checksum"],
                        encryption_key_id=data.get("encryption_key_id"),
                        storage_tier=StorageTier(data["storage_tier"]),
                        retention_days=data["retention_days"],
                        status=BackupStatus(data["status"]),
                    )

                    self.context.backup_history.append(metadata)
                    self.context.backup_metadata[metadata.backup_id] = metadata

            # Sort by timestamp
            self.context.backup_history.sort(key=lambda b: b.timestamp)

        except Exception as e:
            logger.error(f"Failed to load backup history: {e}")

    def save_metadata(self, metadata: BackupMetadata) -> None:
        """Save backup metadata to disk."""
        try:
            metadata_file = Path(self.config.backup_dir) / f"{metadata.backup_id}.meta"

            with open(metadata_file, "w") as f:
                json.dump(metadata.to_dict(), f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save backup metadata: {e}")

    def find_metadata(self, backup_id: str) -> BackupMetadata | None:
        """Find backup metadata by ID."""
        cached = self.context.backup_metadata.get(backup_id)
        if cached:
            if self._metadata_file_exists(backup_id):
                return cached
            self._remove_from_history(backup_id)

        for metadata in list(self.context.backup_history):
            if metadata.backup_id == backup_id:
                if self._metadata_file_exists(backup_id):
                    return metadata
                self._remove_from_history(backup_id)
                break

        # Try loading from disk
        metadata_file = Path(self.config.backup_dir) / f"{backup_id}.meta"

        if metadata_file.exists():
            with open(metadata_file) as f:
                data = json.load(f)
                return BackupMetadata(
                    backup_id=data["backup_id"],
                    backup_type=BackupType(data["backup_type"]),
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    size_bytes=data["size_bytes"],
                    size_compressed=data["size_compressed"],
                    checksum=data["checksum"],
                    encryption_key_id=data.get("encryption_key_id"),
                    storage_tier=StorageTier(data["storage_tier"]),
                    retention_days=data["retention_days"],
                    status=BackupStatus(data["status"]),
                )

        return None

    def add_to_history(
        self, metadata: BackupMetadata, backup_type: BackupType, start_time: datetime
    ) -> None:
        """Add backup metadata to history and persist it."""
        self.context.backup_history.append(metadata)
        self.context.backup_metadata[metadata.backup_id] = metadata
        self.save_metadata(metadata)

        if backup_type == BackupType.FULL:
            self.context.last_full_backup = start_time
        elif backup_type == BackupType.DIFFERENTIAL:
            self.context.last_differential_backup = start_time

    def _remove_from_history(self, backup_id: str) -> None:
        """Remove backup metadata from history."""
        self.context.backup_history = [
            metadata for metadata in self.context.backup_history if metadata.backup_id != backup_id
        ]
        self.context.backup_metadata.pop(backup_id, None)

    def _metadata_file_exists(self, backup_id: str) -> bool:
        """Check if metadata file exists on disk."""
        metadata_file = Path(self.config.backup_dir) / f"{backup_id}.meta"
        return metadata_file.exists()

    def get_last_backup_time(self) -> datetime:
        """Get timestamp of last successful backup."""
        if not self.context.backup_history:
            return datetime.now(timezone.utc) - timedelta(days=1)

        successful = [
            b
            for b in self.context.backup_history
            if b.status in [BackupStatus.COMPLETED, BackupStatus.VERIFIED]
        ]

        if successful:
            return max(b.timestamp for b in successful)

        return datetime.now(timezone.utc) - timedelta(days=1)

    def get_stats(self) -> dict[str, Any]:
        """Get backup statistics."""
        if not self.context.backup_history:
            return {"total_backups": 0, "total_size_bytes": 0, "compression_ratio": 0}

        successful = [
            b
            for b in self.context.backup_history
            if b.status in [BackupStatus.COMPLETED, BackupStatus.VERIFIED]
        ]

        total_original = sum(b.size_bytes for b in successful)
        total_compressed = sum(b.size_compressed for b in successful)

        return {
            "total_backups": len(self.context.backup_history),
            "successful_backups": len(successful),
            "total_size_bytes": total_original,
            "total_compressed_bytes": total_compressed,
            "compression_ratio": (
                (1 - total_compressed / total_original) * 100 if total_original > 0 else 0
            ),
            "last_full_backup": (
                self.context.last_full_backup.isoformat() if self.context.last_full_backup else None
            ),
            "last_backup": self.get_last_backup_time().isoformat(),
            "backups_by_type": {
                backup_type.value: len([b for b in successful if b.backup_type == backup_type])
                for backup_type in BackupType
            },
            "storage_distribution": {
                tier.value: len([b for b in successful if b.storage_tier == tier])
                for tier in StorageTier
            },
        }
