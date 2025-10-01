"""Storage tier determination strategy.

Determines appropriate storage tier based on backup type and configuration.
"""

import logging
from typing import Any

from bot_v2.state.backup.models import BackupType, StorageTier

logger = logging.getLogger(__name__)


class TierStrategy:
    """Strategy for determining storage tier for backups."""

    def __init__(
        self,
        has_network_storage: bool = False,
        has_cloud_storage: bool = False,
    ) -> None:
        """
        Initialize tier strategy.

        Args:
            has_network_storage: Whether network storage is available
            has_cloud_storage: Whether cloud storage (S3) is available
        """
        self.has_network_storage = has_network_storage
        self.has_cloud_storage = has_cloud_storage

    def determine_tier(self, backup_type: BackupType) -> StorageTier:
        """
        Determine appropriate storage tier for backup type.

        Strategy:
        - INCREMENTAL: LOCAL (fast, frequent)
        - DIFFERENTIAL: NETWORK (if available) or LOCAL
        - FULL: CLOUD (if available) or LOCAL
        - EMERGENCY: LOCAL (fastest access needed)
        - SNAPSHOT: LOCAL (fast access needed)

        Args:
            backup_type: Type of backup

        Returns:
            Appropriate storage tier
        """
        if backup_type == BackupType.INCREMENTAL:
            return StorageTier.LOCAL

        elif backup_type == BackupType.DIFFERENTIAL:
            return StorageTier.NETWORK if self.has_network_storage else StorageTier.LOCAL

        elif backup_type == BackupType.FULL:
            return StorageTier.CLOUD if self.has_cloud_storage else StorageTier.LOCAL

        elif backup_type == BackupType.EMERGENCY:
            # Fast access needed
            return StorageTier.LOCAL

        elif backup_type == BackupType.SNAPSHOT:
            # Fast access needed
            return StorageTier.LOCAL

        else:
            # Default to local
            logger.warning(f"Unknown backup type {backup_type}, defaulting to LOCAL")
            return StorageTier.LOCAL

    def should_archive(self, backup_type: BackupType, age_days: int) -> bool:
        """
        Determine if backup should be archived to slower/cheaper tier.

        Args:
            backup_type: Type of backup
            age_days: Age of backup in days

        Returns:
            True if should be archived
        """
        # Archive full backups older than 30 days
        if backup_type == BackupType.FULL and age_days > 30:
            return True

        # Archive differential backups older than 14 days
        if backup_type == BackupType.DIFFERENTIAL and age_days > 14:
            return True

        return False
