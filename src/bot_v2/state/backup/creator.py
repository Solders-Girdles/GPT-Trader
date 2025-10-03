"""Backup creation orchestration.

Handles the core backup creation workflow:
- Data serialization and transformation
- Compression and encryption
- Storage tier management
- Metadata generation
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from bot_v2.state.backup.models import (
    BackupContext,
    BackupMetadata,
    BackupStatus,
    BackupType,
    StorageTier,
)

if TYPE_CHECKING:
    from bot_v2.state.backup.metadata import BackupMetadataManager
    from bot_v2.state.backup.models import BackupConfig
    from bot_v2.state.backup.services import (
        CompressionService,
        EncryptionService,
        TierStrategy,
        TransportService,
    )

logger = logging.getLogger(__name__)


class BackupCreator:
    """Orchestrates backup creation workflow.

    Handles serialization, compression, encryption, and storage
    while delegating data collection to the BackupManager.
    """

    def __init__(
        self,
        config: BackupConfig,
        context: BackupContext,
        metadata_manager: BackupMetadataManager,
        encryption_service: EncryptionService,
        compression_service: CompressionService,
        transport_service: TransportService,
        tier_strategy: TierStrategy,
    ):
        self.config = config
        self.context = context
        self.metadata_manager = metadata_manager
        self.encryption_service = encryption_service
        self.compression_service = compression_service
        self.transport_service = transport_service
        self.tier_strategy = tier_strategy

    async def create_backup_internal(
        self,
        backup_type: BackupType,
        backup_data: dict[str, Any],
        backup_id: str,
        start_time: datetime,
        pending_snapshot: dict[str, Any] | None,
    ) -> BackupMetadata:
        """Create backup from collected data.

        Args:
            backup_type: Type of backup to create
            backup_data: Pre-collected backup data (from BackupManager)
            backup_id: Pre-generated backup ID
            start_time: Backup start timestamp
            pending_snapshot: Normalized state snapshot (for baseline update)

        Returns:
            BackupMetadata instance

        Raises:
            Exception: If backup creation fails
        """
        logger.info(f"Creating {backup_type.value} backup {backup_id}")

        if not backup_data:
            raise Exception("No data to backup")

        serialized = self._serialize_backup_data(backup_data)

        # Calculate actual diff payload sizes for incremental/differential backups
        # backup_data["state"] already contains the diff for inc/diff backups
        state_serialized = json.dumps(backup_data.get("state", {}), default=str).encode("utf-8")
        state_original_size = len(state_serialized)
        if self.config.enable_compression:
            _, _, state_compressed_size = self.compression_service.compress(state_serialized)
        else:
            state_compressed_size = 0

        payload, _, _ = self._prepare_compressed_payload(serialized)
        payload, encryption_key_id = self._encrypt_payload(payload)
        checksum = hashlib.sha256(payload).hexdigest()
        storage_tier = self._determine_storage_tier(backup_type)
        await self._store_backup(backup_id, payload, storage_tier)

        duration_seconds = (datetime.now(timezone.utc) - start_time).total_seconds()
        metadata = self._build_backup_metadata(
            backup_id=backup_id,
            backup_type=backup_type,
            timestamp=start_time,
            original_size=state_original_size,
            compressed_size=state_compressed_size,
            checksum=checksum,
            encryption_key_id=encryption_key_id,
            storage_tier=storage_tier,
            duration_seconds=duration_seconds,
            backup_data=backup_data,
        )

        # Update baseline snapshots if we have a pending snapshot
        if pending_snapshot is not None:
            self.context.update_baseline(backup_type, pending_snapshot)

        # Verification (optional)
        if self.config.verify_after_backup:
            if await self._verify_backup(metadata, payload):
                metadata.status = BackupStatus.VERIFIED
                metadata.verification_status = "passed"
            else:
                metadata.status = BackupStatus.CORRUPTED
                metadata.verification_status = "failed"
                logger.warning(f"Backup {backup_id} verification failed")

        # Add to history
        self.metadata_manager.add_to_history(metadata, backup_type, start_time)
        if metadata.timestamp.tzinfo is not None:
            metadata.timestamp = metadata.timestamp.replace(tzinfo=None)

        logger.info(
            "Backup %s completed in %.2fs, compressed %d -> %d bytes (%.1f%%)",
            backup_id,
            metadata.backup_duration_seconds,
            state_original_size,
            state_compressed_size,
            (
                (state_compressed_size / state_original_size * 100)
                if state_original_size and state_compressed_size
                else 0.0
            ),
        )

        return metadata

    def _serialize_backup_data(self, backup_data: dict[str, Any]) -> bytes:
        """Serialize backup data to bytes."""
        return json.dumps(backup_data, default=str).encode()

    def _prepare_compressed_payload(self, serialized: bytes) -> tuple[bytes, int, int]:
        """Compress serialized backup data."""
        return self.compression_service.compress(serialized)

    def _encrypt_payload(self, payload: bytes) -> tuple[bytes, str | None]:
        """Encrypt backup payload."""
        return self.encryption_service.encrypt(payload)

    def _build_backup_metadata(
        self,
        *,
        backup_id: str,
        backup_type: BackupType,
        timestamp: datetime,
        original_size: int,
        compressed_size: int,
        checksum: str,
        encryption_key_id: str | None,
        storage_tier: StorageTier,
        duration_seconds: float,
        backup_data: dict[str, Any],
    ) -> BackupMetadata:
        """Build backup metadata from creation results."""
        state = backup_data.get("state", {})
        data_sources = list(state.keys()) if isinstance(state, dict) else []

        # Get retention days from config
        from bot_v2.state.backup.services import RetentionService

        generic_retention = getattr(self.config, "retention_days", None)
        if generic_retention is not None:
            retention_days = generic_retention
        else:
            # Use retention service logic
            retention_service = RetentionService(
                retention_incremental=self.config.retention_incremental,
                retention_differential=self.config.retention_differential,
                retention_full=self.config.retention_full,
                retention_emergency=getattr(self.config, "retention_emergency", 30),
                retention_snapshot=7,
            )
            retention_days = retention_service.get_retention_days(backup_type)

        return BackupMetadata(
            backup_id=backup_id,
            backup_type=backup_type,
            timestamp=timestamp,
            size_bytes=original_size,
            size_compressed=compressed_size,
            checksum=checksum,
            encryption_key_id=encryption_key_id,
            storage_tier=storage_tier,
            retention_days=retention_days,
            status=BackupStatus.COMPLETED,
            backup_duration_seconds=duration_seconds,
            data_sources=data_sources,
        )

    def _normalize_state_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Ensure payload is JSON serializable for persistence."""
        return json.loads(json.dumps(payload, default=str))

    def _diff_state(
        self, baseline: dict[str, Any] | None, current: dict[str, Any]
    ) -> dict[str, Any]:
        """Compute diff between baseline and current state.

        Used for incremental and differential backups.
        """
        if not baseline:
            return current

        diff: dict[str, Any] = {}

        for key, value in current.items():
            base_value = baseline.get(key)

            if isinstance(value, dict) and isinstance(base_value, dict):
                nested_diff = self._diff_state(base_value, value)
                if nested_diff:
                    diff[key] = nested_diff
            elif value != base_value:
                diff[key] = value

        return diff

    async def _store_backup(self, backup_id: str, data: bytes, storage_tier: StorageTier) -> str:
        """Store backup data to appropriate tier."""
        return await self.transport_service.store(backup_id, data, storage_tier)

    def _determine_storage_tier(self, backup_type: BackupType) -> StorageTier:
        """Determine appropriate storage tier for backup type."""
        return self.tier_strategy.determine_tier(backup_type)

    async def _verify_backup(self, metadata: BackupMetadata, data: bytes) -> bool:
        """Verify backup integrity."""
        try:
            # Verify checksum
            calculated = hashlib.sha256(data).hexdigest()

            if calculated != metadata.checksum:
                logger.error(f"Backup {metadata.backup_id} checksum verification failed")
                return False

            # Try to decrypt and decompress
            test_data = data

            if metadata.encryption_key_id:
                test_data = self.encryption_service.decrypt(test_data)

            if self.config.enable_compression:
                test_data = self.compression_service.decompress(test_data)

            # Try to parse
            json.loads(test_data.decode() if isinstance(test_data, bytes) else test_data)

            return True

        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            return False
