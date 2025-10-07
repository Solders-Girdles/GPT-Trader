"""
Retention Manager - Backup retention policy and cleanup orchestration

Extracted from BackupManager to separate retention concerns.
Handles expired backup deletion, metadata cleanup, and retention policy enforcement.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from bot_v2.state.backup.models import BackupConfig, BackupContext, BackupMetadata, BackupType
from bot_v2.state.backup.services import RetentionService, TransportService

if TYPE_CHECKING:  # pragma: no cover
    from bot_v2.monitoring.metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)


class RetentionManager:
    """
    Manages backup retention policy and cleanup operations.

    Responsibilities:
    - Expired backup identification (delegates to RetentionService)
    - Batch deletion orchestration (with sequential fallback)
    - Metadata and history cleanup
    - Retention policy queries
    """

    def __init__(
        self,
        retention_service: RetentionService,
        transport_service: TransportService,
        context: BackupContext,
        config: BackupConfig,
        metrics_collector: "MetricsCollector | None" = None,
    ) -> None:
        """
        Initialize retention manager.

        Args:
            retention_service: Service for retention policy logic
            transport_service: Service for storage operations
            context: Shared backup context
            config: Backup configuration
            metrics_collector: Optional metrics collector for telemetry
        """
        self.retention_service = retention_service
        self.transport_service = transport_service
        self.context = context
        self.config = config
        self.metrics_collector = metrics_collector

    async def cleanup(self, expired_backups: list[BackupMetadata] | None = None) -> int:
        """
        Remove expired backups based on retention policy using batch operations.

        Args:
            expired_backups: Optional pre-filtered list of expired backups.
                            If None, will filter using retention_service.

        Returns:
            Count of successfully removed backups
        """
        # Record cleanup operation start
        if self.metrics_collector:
            self.metrics_collector.record_counter("backup.retention.cleaned_total")

        removed_count = 0
        cleanup_success = False

        try:
            # Filter expired backups if not provided
            if expired_backups is None:
                current_time = datetime.now(timezone.utc)
                all_backups = list(self.context.backup_metadata.values())
                expired_backups = self.retention_service.filter_expired(all_backups, current_time)

            if not expired_backups:
                cleanup_success = True
                if self.metrics_collector:
                    self.metrics_collector.record_counter("backup.retention.cleaned_success")
                return 0

            # Use batch delete if available (especially efficient for S3/ARCHIVE tiers)
            if hasattr(self.transport_service, "batch_delete"):
                # Prepare batch delete: backup_ids and tier_map
                backup_ids = [m.backup_id for m in expired_backups]
                tier_map = {m.backup_id: m.storage_tier for m in expired_backups}

                # Batch delete from storage
                results = await self.transport_service.batch_delete(backup_ids, tier_map)

                # Update metadata for successful deletions
                for backup_id, success in results.items():
                    if success:
                        self.context.backup_metadata.pop(backup_id, None)
                        self.context.backup_history = [
                            entry
                            for entry in self.context.backup_history
                            if entry.backup_id != backup_id
                        ]
                        removed_count += 1

                logger.info(f"Batch cleaned up {removed_count} expired backups")

            else:
                # Fallback: Sequential deletion for compatibility
                for metadata in expired_backups:
                    if await self.transport_service.delete(
                        metadata.backup_id, metadata.storage_tier
                    ):
                        self.context.backup_metadata.pop(metadata.backup_id, None)
                        self.context.backup_history = [
                            entry
                            for entry in self.context.backup_history
                            if entry.backup_id != metadata.backup_id
                        ]
                        removed_count += 1
                        logger.debug(f"Removed expired backup {metadata.backup_id}")

                if removed_count > 0:
                    logger.info(f"Cleaned up {removed_count} expired backups (sequential)")

            # Cleanup metadata files (only for successfully deleted backups)
            successfully_deleted = [
                m.backup_id
                for m in expired_backups
                if m.backup_id not in self.context.backup_metadata
            ]
            if successfully_deleted:
                self.retention_service.cleanup_metadata_files(
                    Path(self.config.backup_dir), successfully_deleted
                )

            cleanup_success = True

        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")
            if self.metrics_collector:
                self.metrics_collector.record_counter("backup.retention.cleaned_failed")

        finally:
            # Record metrics
            if self.metrics_collector:
                # Record success (if no exception occurred)
                if cleanup_success:
                    self.metrics_collector.record_counter("backup.retention.cleaned_success")

                # Record removal count
                if removed_count > 0:
                    self.metrics_collector.record_histogram(
                        "backup.retention.removed_count", float(removed_count)
                    )

        return removed_count

    def get_retention_days(self, backup_type: BackupType) -> int:
        """
        Get retention period for backup type.

        Checks for generic retention override first, then delegates to retention service.

        Args:
            backup_type: Type of backup

        Returns:
            Retention period in days
        """
        generic_retention = getattr(self.config, "retention_days", None)
        if generic_retention is not None:
            return generic_retention

        return self.retention_service.get_retention_days(backup_type)
