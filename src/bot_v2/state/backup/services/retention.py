"""Retention service for backup lifecycle management.

Handles cleanup of old backups based on retention policies.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

from bot_v2.state.backup.models import BackupMetadata, BackupType

logger = logging.getLogger(__name__)


class RetentionService:
    """Service for managing backup retention and cleanup."""

    def __init__(
        self,
        retention_incremental: int = 7,
        retention_differential: int = 30,
        retention_full: int = 90,
        retention_emergency: int = 30,
        retention_snapshot: int = 7,
    ) -> None:
        """
        Initialize retention service.

        Args:
            retention_incremental: Days to retain incremental backups
            retention_differential: Days to retain differential backups
            retention_full: Days to retain full backups
            retention_emergency: Days to retain emergency backups
            retention_snapshot: Days to retain snapshot backups
        """
        self.retention_map = {
            BackupType.INCREMENTAL: retention_incremental,
            BackupType.DIFFERENTIAL: retention_differential,
            BackupType.FULL: retention_full,
            BackupType.EMERGENCY: retention_emergency,
            BackupType.SNAPSHOT: retention_snapshot,
        }

    def get_retention_days(self, backup_type: BackupType) -> int:
        """
        Get retention period for backup type.

        Args:
            backup_type: Type of backup

        Returns:
            Retention period in days
        """
        return self.retention_map.get(backup_type, 30)

    def is_expired(self, metadata: BackupMetadata, current_time: datetime | None = None) -> bool:
        """
        Check if backup is expired based on retention policy.

        Args:
            metadata: Backup metadata
            current_time: Current time (defaults to now)

        Returns:
            True if backup should be deleted
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        retention_days = self.get_retention_days(metadata.backup_type)

        # Ensure timestamp has timezone
        timestamp = metadata.timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        age_days = (current_time - timestamp).days

        return age_days > retention_days

    def filter_expired(
        self,
        backups: list[BackupMetadata],
        current_time: datetime | None = None,
    ) -> list[BackupMetadata]:
        """
        Filter list to get expired backups.

        Args:
            backups: List of backup metadata
            current_time: Current time (defaults to now)

        Returns:
            List of expired backups
        """
        return [b for b in backups if self.is_expired(b, current_time)]

    def filter_active(
        self,
        backups: list[BackupMetadata],
        current_time: datetime | None = None,
    ) -> list[BackupMetadata]:
        """
        Filter list to get active (non-expired) backups.

        Args:
            backups: List of backup metadata
            current_time: Current time (defaults to now)

        Returns:
            List of active backups
        """
        return [b for b in backups if not self.is_expired(b, current_time)]

    def should_promote_to_archive(
        self,
        metadata: BackupMetadata,
        archive_threshold_days: int = 30,
        current_time: datetime | None = None,
    ) -> bool:
        """
        Determine if backup should be promoted to archive tier.

        Args:
            metadata: Backup metadata
            archive_threshold_days: Days before archiving
            current_time: Current time (defaults to now)

        Returns:
            True if should be archived
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        # Only archive full backups
        if metadata.backup_type != BackupType.FULL:
            return False

        timestamp = metadata.timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        age_days = (current_time - timestamp).days

        return age_days > archive_threshold_days

    def cleanup_metadata_files(self, backup_dir: Path, backup_ids: list[str]) -> int:
        """
        Remove metadata files for deleted backups.

        Args:
            backup_dir: Backup directory path
            backup_ids: List of backup IDs to clean up

        Returns:
            Number of metadata files removed
        """
        removed = 0

        for backup_id in backup_ids:
            meta_file = backup_dir / f"{backup_id}.meta"
            if meta_file.exists():
                try:
                    meta_file.unlink()
                    removed += 1
                except Exception as e:
                    logger.error(f"Failed to remove metadata file {meta_file}: {e}")

        return removed
