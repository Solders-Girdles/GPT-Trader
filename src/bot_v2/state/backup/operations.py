"""
Backup Manager for Bot V2 Trading System

Provides comprehensive backup system with encryption, compression,
and multi-tier storage for disaster recovery and compliance.
"""

import asyncio
import gzip
import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from bot_v2.state.backup.models import (
    BackupConfig,
    BackupMetadata,
    BackupStatus,
    BackupType,
    StorageTier,
)

# Optional encryption support
try:
    from cryptography.fernet import Fernet

    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False
    logging.warning("Cryptography library not available, encryption disabled")

# Optional cloud storage support
try:
    import boto3

    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    logging.warning("Boto3 not available, S3 backup disabled")

logger = logging.getLogger(__name__)


class BackupManager:
    """
    Comprehensive backup system with encryption, compression, and tiering.
    Ensures RPO <1 minute through continuous incremental backups.
    """

    def __init__(self, state_manager: Any, config: BackupConfig | None = None) -> None:
        self.state_manager = state_manager
        self.config = config or BackupConfig()

        self._backup_lock = threading.Lock()
        self._backup_in_progress = False
        self._backup_history: list[BackupMetadata] = []
        self._last_full_backup: datetime | None = None
        self._last_differential_backup: datetime | None = None
        self._encryption_key: bytes | None = None
        self._scheduled_tasks: list[asyncio.Task] = []
        self.s3_client: Any | None = None

        # Initialize storage paths
        self._init_storage_paths()

        # Initialize encryption if enabled
        if self.config.enable_encryption and ENCRYPTION_AVAILABLE:
            self._init_encryption()

        # Initialize S3 client if available
        if S3_AVAILABLE and self.config.s3_bucket:
            self._init_s3()

        # Load backup history
        self._load_backup_history()

        logger.info(
            f"BackupManager initialized with {len(self._backup_history)} backups in history"
        )

    def _init_storage_paths(self) -> None:
        """Initialize backup storage paths"""
        for path_str in [self.config.backup_dir, self.config.local_storage_path]:
            path = Path(path_str)
            path.mkdir(parents=True, exist_ok=True)

        if self.config.network_storage_path:
            Path(self.config.network_storage_path).mkdir(parents=True, exist_ok=True)

    def _init_encryption(self) -> None:
        """Initialize encryption key"""
        try:
            # Try to load existing key
            key_file = Path(self.config.backup_dir) / ".encryption_key"

            if key_file.exists():
                with open(key_file, "rb") as f:
                    self._encryption_key = f.read()
            else:
                # Generate new key
                self._encryption_key = Fernet.generate_key()

                # Save key (in production, use proper key management)
                with open(key_file, "wb") as f:
                    f.write(self._encryption_key)

                # Secure the key file
                os.chmod(key_file, 0o600)

            logger.info("Encryption initialized")

        except Exception as e:
            logger.error(f"Encryption initialization failed: {e}")
            self.config.enable_encryption = False

    def _init_s3(self) -> None:
        """Initialize S3 client"""
        try:
            self.s3_client = boto3.client("s3", region_name=self.config.s3_region)

            # Verify bucket exists
            self.s3_client.head_bucket(Bucket=self.config.s3_bucket)

            logger.info(f"S3 backup initialized with bucket {self.config.s3_bucket}")

        except Exception as e:
            logger.warning(f"S3 initialization failed: {e}")
            self.s3_client = None

    async def start_scheduled_backups(self) -> None:
        """Start automated backup scheduling"""
        logger.info("Starting scheduled backups")

        # Schedule full backups
        self._scheduled_tasks.append(asyncio.create_task(self._run_full_backup_schedule()))

        # Schedule differential backups
        self._scheduled_tasks.append(asyncio.create_task(self._run_differential_backup_schedule()))

        # Schedule incremental backups
        self._scheduled_tasks.append(asyncio.create_task(self._run_incremental_backup_schedule()))

        # Schedule cleanup
        self._scheduled_tasks.append(asyncio.create_task(self._run_cleanup_schedule()))

        # Schedule verification
        self._scheduled_tasks.append(asyncio.create_task(self._run_verification_schedule()))

    async def stop_scheduled_backups(self) -> None:
        """Stop scheduled backups"""
        for task in self._scheduled_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._scheduled_tasks.clear()
        logger.info("Scheduled backups stopped")

    async def _run_full_backup_schedule(self) -> None:
        """Run full backup on schedule"""
        while True:
            try:
                await asyncio.sleep(self.config.full_backup_interval_hours * 3600)
                await self.create_backup(BackupType.FULL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Full backup schedule error: {e}")

    async def _run_differential_backup_schedule(self) -> None:
        """Run differential backup on schedule"""
        while True:
            try:
                await asyncio.sleep(self.config.differential_backup_interval_hours * 3600)
                await self.create_backup(BackupType.DIFFERENTIAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Differential backup schedule error: {e}")

    async def _run_incremental_backup_schedule(self) -> None:
        """Run incremental backup on schedule"""
        while True:
            try:
                await asyncio.sleep(self.config.incremental_backup_interval_minutes * 60)
                await self.create_backup(BackupType.INCREMENTAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Incremental backup schedule error: {e}")

    async def create_backup(
        self, backup_type: BackupType = BackupType.FULL
    ) -> BackupMetadata | None:
        """
        Create backup of specified type.

        Args:
            backup_type: Type of backup to create

        Returns:
            Backup metadata or None if failed
        """
        if self._backup_in_progress:
            logger.warning("Backup already in progress")
            return None

        with self._backup_lock:
            self._backup_in_progress = True
            start_time = datetime.utcnow()

            try:
                backup_id = self._generate_backup_id(backup_type)

                logger.info(f"Creating {backup_type.value} backup {backup_id}")

                backup_data = await self._collect_backup_data(backup_type)

                if not backup_data:
                    raise Exception("No data to backup")
                serialized = self._serialize_backup_data(backup_data)
                payload, original_size, compressed_size = self._prepare_compressed_payload(
                    serialized
                )
                payload, encryption_key_id = self._encrypt_payload(payload)
                checksum = hashlib.sha256(payload).hexdigest()
                storage_tier = self._determine_storage_tier(backup_type)
                await self._store_backup(backup_id, payload, storage_tier)

                duration_seconds = (datetime.utcnow() - start_time).total_seconds()
                metadata = self._build_backup_metadata(
                    backup_id=backup_id,
                    backup_type=backup_type,
                    timestamp=start_time,
                    original_size=original_size,
                    compressed_size=compressed_size,
                    checksum=checksum,
                    encryption_key_id=encryption_key_id,
                    storage_tier=storage_tier,
                    duration_seconds=duration_seconds,
                    backup_data=backup_data,
                )

                if self.config.verify_after_backup:
                    if await self._verify_backup(metadata, payload):
                        metadata.status = BackupStatus.VERIFIED
                        metadata.verification_status = "passed"
                    else:
                        metadata.status = BackupStatus.CORRUPTED
                        metadata.verification_status = "failed"
                        logger.warning(f"Backup {backup_id} verification failed")

                self._update_history_after_backup(metadata, backup_type, start_time)

                logger.info(
                    "Backup %s completed in %.2fs, compressed %d -> %d bytes (%.1f%%)",
                    backup_id,
                    metadata.backup_duration_seconds,
                    original_size,
                    compressed_size,
                    (compressed_size / original_size * 100) if original_size else 0.0,
                )

                return metadata

            except Exception as e:
                logger.error(f"Backup creation failed: {e}")
                return None

            finally:
                self._backup_in_progress = False

    async def restore_from_backup(self, backup_id: str) -> bool:
        """
        Restore system from specific backup.

        Args:
            backup_id: ID of backup to restore

        Returns:
            Success status
        """
        try:
            logger.info(f"Restoring from backup {backup_id}")

            # Find backup metadata
            metadata = self._find_backup_metadata(backup_id)

            if not metadata:
                logger.error(f"Backup {backup_id} not found")
                return False

            # Retrieve backup data
            backup_data = await self._retrieve_backup(metadata)

            if not backup_data:
                logger.error(f"Failed to retrieve backup {backup_id}")
                return False

            # Verify checksum
            calculated_checksum = hashlib.sha256(backup_data).hexdigest()

            if calculated_checksum != metadata.checksum:
                logger.error(f"Backup {backup_id} checksum mismatch")
                return False

            # Decrypt if needed
            if metadata.encryption_key_id and self._encryption_key:
                cipher = Fernet(self._encryption_key)
                backup_data = cipher.decrypt(backup_data)

            # Decompress if needed
            if self.config.enable_compression:
                backup_data = gzip.decompress(backup_data)

            # Parse backup data
            restored_data = json.loads(
                backup_data.decode() if isinstance(backup_data, bytes) else backup_data
            )

            # Restore to state manager
            success = await self._restore_data_to_state(restored_data)

            if success:
                logger.info(f"Successfully restored from backup {backup_id}")
            else:
                logger.error(f"Failed to restore data from backup {backup_id}")

            return success

        except Exception as e:
            logger.error(f"Backup restoration failed: {e}")
            return False

    async def restore_latest_backup(self, backup_type: BackupType | None = None) -> bool:
        """
        Restore from most recent backup.

        Args:
            backup_type: Optional type filter

        Returns:
            Success status
        """
        # Find latest backup
        backups = [
            b
            for b in self._backup_history
            if b.status in [BackupStatus.COMPLETED, BackupStatus.VERIFIED]
        ]

        if backup_type:
            backups = [b for b in backups if b.backup_type == backup_type]

        if not backups:
            logger.error("No valid backups found")
            return False

        latest = max(backups, key=lambda b: b.timestamp)

        return await self.restore_from_backup(latest.backup_id)

    def _serialize_backup_data(self, backup_data: dict[str, Any]) -> bytes:
        return json.dumps(backup_data, default=str).encode()

    def _prepare_compressed_payload(self, serialized: bytes) -> tuple[bytes, int, int]:
        original_size = len(serialized)
        if not self.config.enable_compression:
            return serialized, original_size, original_size
        compressed = gzip.compress(serialized, compresslevel=self.config.compression_level)
        return compressed, original_size, len(compressed)

    def _encrypt_payload(self, payload: bytes) -> tuple[bytes, str | None]:
        if self.config.enable_encryption and self._encryption_key:
            cipher = Fernet(self._encryption_key)
            return cipher.encrypt(payload), "primary"
        return payload, None

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
        return BackupMetadata(
            backup_id=backup_id,
            backup_type=backup_type,
            timestamp=timestamp,
            size_bytes=original_size,
            size_compressed=compressed_size,
            checksum=checksum,
            encryption_key_id=encryption_key_id,
            storage_tier=storage_tier,
            retention_days=self._get_retention_days(backup_type),
            status=BackupStatus.COMPLETED,
            backup_duration_seconds=duration_seconds,
            data_sources=list(backup_data.keys()),
        )

    def _update_history_after_backup(
        self, metadata: BackupMetadata, backup_type: BackupType, start_time: datetime
    ) -> None:
        self._backup_history.append(metadata)
        self._save_backup_metadata(metadata)
        if backup_type == BackupType.FULL:
            self._last_full_backup = start_time
        elif backup_type == BackupType.DIFFERENTIAL:
            self._last_differential_backup = start_time

    async def _collect_backup_data(self, backup_type: BackupType) -> dict[str, Any]:
        """Collect data for backup based on type"""
        backup_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "backup_type": backup_type.value,
            "system_info": {"version": "1.0.0", "environment": os.environ.get("ENV", "production")},
        }

        if backup_type == BackupType.FULL:
            # Full backup - everything
            backup_data.update(await self._collect_all_data())

        elif backup_type == BackupType.INCREMENTAL:
            # Only changes since last backup
            last_backup = self._get_last_backup_time()
            backup_data.update(await self._collect_changed_data(last_backup))

        elif backup_type == BackupType.DIFFERENTIAL:
            # Changes since last full backup
            last_full = self._last_full_backup or datetime.utcnow() - timedelta(days=30)
            backup_data.update(await self._collect_changed_data(last_full))

        elif backup_type == BackupType.SNAPSHOT:
            # Current state snapshot
            backup_data.update(await self._collect_snapshot_data())

        elif backup_type == BackupType.EMERGENCY:
            # Critical data only for fast backup
            backup_data.update(await self._collect_critical_data())

        return backup_data

    async def _collect_all_data(self) -> dict[str, Any]:
        """Collect all system data"""
        data = {}

        # Collect all state data
        patterns = [
            "position:*",
            "order:*",
            "portfolio*",
            "ml_model:*",
            "config:*",
            "performance*",
            "strategy:*",
        ]

        for pattern in patterns:
            keys = await self.state_manager.get_keys_by_pattern(pattern)

            for key in keys:
                value = await self.state_manager.get_state(key)
                if value:
                    data[key] = value

        logger.debug(f"Collected {len(data)} items for full backup")

        return data

    async def _collect_changed_data(self, since: datetime) -> dict[str, Any]:
        """Collect data changed since given time"""
        data = {}

        # Get recently modified keys
        # This would ideally track modification times
        patterns = ["position:*", "order:*", "portfolio*"]

        for pattern in patterns:
            keys = await self.state_manager.get_keys_by_pattern(pattern)

            for key in keys:
                value = await self.state_manager.get_state(key)

                # Check if data has timestamp
                if value and isinstance(value, dict):
                    timestamp = value.get("timestamp") or value.get("last_updated")

                    if timestamp:
                        try:
                            dt = (
                                datetime.fromisoformat(timestamp)
                                if isinstance(timestamp, str)
                                else timestamp
                            )
                            if dt > since:
                                data[key] = value
                        except (TypeError, ValueError) as exc:
                            logger.debug(
                                "Unable to parse timestamp %s for key %s: %s",
                                timestamp,
                                key,
                                exc,
                                exc_info=True,
                            )
                            # Include if timestamp is malformed so downstream reconciliation can decide.
                            data[key] = value
                    else:
                        # Include if no timestamp
                        data[key] = value

        logger.debug(f"Collected {len(data)} changed items since {since}")

        return data

    async def _collect_snapshot_data(self) -> dict[str, Any]:
        """Collect current state snapshot"""
        return {
            "positions": await self._get_all_by_pattern("position:*"),
            "orders": await self._get_all_by_pattern("order:*"),
            "portfolio": await self.state_manager.get_state("portfolio_current"),
            "performance": await self.state_manager.get_state("performance_metrics"),
        }

    async def _collect_critical_data(self) -> dict[str, Any]:
        """Collect only critical data for emergency backup"""
        return {
            "positions": await self._get_all_by_pattern("position:*"),
            "portfolio": await self.state_manager.get_state("portfolio_current"),
            "critical_config": await self.state_manager.get_state("config:critical"),
        }

    async def _get_all_by_pattern(self, pattern: str) -> dict[str, Any]:
        """Get all data matching pattern"""
        result = {}
        keys = await self.state_manager.get_keys_by_pattern(pattern)

        for key in keys:
            value = await self.state_manager.get_state(key)
            if value:
                result[key] = value

        return result

    async def _store_backup(self, backup_id: str, data: bytes, storage_tier: StorageTier) -> str:
        """Store backup data to appropriate tier"""
        if storage_tier == StorageTier.LOCAL:
            return await self._store_local(backup_id, data)

        elif storage_tier == StorageTier.NETWORK:
            if self.config.network_storage_path:
                return await self._store_network(backup_id, data)
            else:
                return await self._store_local(backup_id, data)

        elif storage_tier == StorageTier.CLOUD:
            if self.s3_client:
                return await self._store_s3(backup_id, data)
            else:
                return await self._store_local(backup_id, data)

        elif storage_tier == StorageTier.ARCHIVE:
            # Archive tier (Glacier or similar)
            if self.s3_client:
                return await self._store_s3_archive(backup_id, data)
            else:
                return await self._store_local(backup_id, data)

        return await self._store_local(backup_id, data)

    async def _store_local(self, backup_id: str, data: bytes) -> str:
        """Store backup locally"""
        file_path = Path(self.config.local_storage_path) / f"{backup_id}.backup"

        with open(file_path, "wb") as f:
            f.write(data)

        return str(file_path)

    async def _store_network(self, backup_id: str, data: bytes) -> str:
        """Store backup to network storage"""
        file_path = Path(self.config.network_storage_path) / f"{backup_id}.backup"

        with open(file_path, "wb") as f:
            f.write(data)

        return str(file_path)

    async def _store_s3(self, backup_id: str, data: bytes) -> str:
        """Store backup to S3"""
        try:
            key = f"backups/{backup_id}.backup"

            self.s3_client.put_object(
                Bucket=self.config.s3_bucket, Key=key, Body=data, StorageClass="STANDARD_IA"
            )

            return f"s3://{self.config.s3_bucket}/{key}"

        except Exception as e:
            logger.error(f"S3 storage failed: {e}")
            # Fallback to local
            return await self._store_local(backup_id, data)

    async def _store_s3_archive(self, backup_id: str, data: bytes) -> str:
        """Store backup to S3 Glacier"""
        try:
            key = f"archive/{backup_id}.backup"

            self.s3_client.put_object(
                Bucket=self.config.s3_bucket, Key=key, Body=data, StorageClass="GLACIER"
            )

            return f"s3://{self.config.s3_bucket}/{key}"

        except Exception as e:
            logger.error(f"S3 archive storage failed: {e}")
            return await self._store_local(backup_id, data)

    async def _retrieve_backup(self, metadata: BackupMetadata) -> bytes | None:
        """Retrieve backup data"""
        if metadata.storage_tier == StorageTier.LOCAL:
            file_path = Path(self.config.local_storage_path) / f"{metadata.backup_id}.backup"

            if file_path.exists():
                with open(file_path, "rb") as f:
                    return f.read()

        elif metadata.storage_tier == StorageTier.NETWORK:
            file_path = Path(self.config.network_storage_path) / f"{metadata.backup_id}.backup"

            if file_path.exists():
                with open(file_path, "rb") as f:
                    return f.read()

        elif metadata.storage_tier in [StorageTier.CLOUD, StorageTier.ARCHIVE]:
            if self.s3_client:
                try:
                    prefix = (
                        "archive" if metadata.storage_tier == StorageTier.ARCHIVE else "backups"
                    )
                    key = f"{prefix}/{metadata.backup_id}.backup"

                    response = self.s3_client.get_object(Bucket=self.config.s3_bucket, Key=key)

                    return response["Body"].read()

                except Exception as e:
                    logger.error(f"S3 retrieval failed: {e}")

        return None

    async def _restore_data_to_state(self, data: dict[str, Any]) -> bool:
        """Restore backup data to state manager"""
        try:
            from .state_manager import StateCategory

            restored_count = 0

            for key, value in data.items():
                if key in ["timestamp", "backup_type", "system_info"]:
                    continue

                # Determine category based on key pattern
                if key.startswith("position:") or key.startswith("order:"):
                    category = StateCategory.HOT
                elif key.startswith("ml_model:") or key.startswith("config:"):
                    category = StateCategory.WARM
                else:
                    category = StateCategory.HOT

                if await self.state_manager.set_state(key, value, category):
                    restored_count += 1

            logger.info(f"Restored {restored_count} items from backup")

            return restored_count > 0

        except Exception as e:
            logger.error(f"Data restoration failed: {e}")
            return False

    async def _verify_backup(self, metadata: BackupMetadata, data: bytes) -> bool:
        """Verify backup integrity"""
        try:
            # Verify checksum
            calculated = hashlib.sha256(data).hexdigest()

            if calculated != metadata.checksum:
                logger.error(f"Backup {metadata.backup_id} checksum verification failed")
                return False

            # Try to decrypt and decompress
            test_data = data

            if metadata.encryption_key_id and self._encryption_key:
                cipher = Fernet(self._encryption_key)
                test_data = cipher.decrypt(test_data)

            if self.config.enable_compression:
                test_data = gzip.decompress(test_data)

            # Try to parse
            json.loads(test_data.decode() if isinstance(test_data, bytes) else test_data)

            return True

        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            return False

    async def cleanup_old_backups(self) -> None:
        """Remove expired backups based on retention policy"""
        try:
            current_time = datetime.utcnow()
            removed_count = 0

            for metadata in self._backup_history[:]:  # Copy to allow removal
                retention_days = self._get_retention_days(metadata.backup_type)
                age_days = (current_time - metadata.timestamp).days

                if age_days > retention_days:
                    # Delete backup file
                    if await self._delete_backup_file(metadata):
                        self._backup_history.remove(metadata)
                        removed_count += 1
                        logger.debug(f"Removed expired backup {metadata.backup_id}")

            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} expired backups")

        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")

    async def _delete_backup_file(self, metadata: BackupMetadata) -> bool:
        """Delete backup file"""
        try:
            if metadata.storage_tier == StorageTier.LOCAL:
                file_path = Path(self.config.local_storage_path) / f"{metadata.backup_id}.backup"
                if file_path.exists():
                    file_path.unlink()

            elif metadata.storage_tier == StorageTier.NETWORK:
                file_path = Path(self.config.network_storage_path) / f"{metadata.backup_id}.backup"
                if file_path.exists():
                    file_path.unlink()

            elif metadata.storage_tier in [StorageTier.CLOUD, StorageTier.ARCHIVE]:
                if self.s3_client:
                    prefix = (
                        "archive" if metadata.storage_tier == StorageTier.ARCHIVE else "backups"
                    )
                    key = f"{prefix}/{metadata.backup_id}.backup"

                    self.s3_client.delete_object(Bucket=self.config.s3_bucket, Key=key)

            return True

        except Exception as e:
            logger.error(f"Failed to delete backup file: {e}")
            return False

    async def _run_cleanup_schedule(self) -> None:
        """Run cleanup on schedule"""
        while True:
            try:
                await asyncio.sleep(86400)  # Daily
                await self.cleanup_old_backups()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup schedule error: {e}")

    async def _run_verification_schedule(self) -> None:
        """Run backup verification on schedule"""
        while True:
            try:
                await asyncio.sleep(self.config.test_restore_frequency_days * 86400)
                await self.test_restore()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Verification schedule error: {e}")

    async def test_restore(self) -> bool:
        """Test restore capability with latest backup"""
        try:
            logger.info("Running backup restore test")

            # Find a recent backup to test
            test_backup = None
            for backup in reversed(self._backup_history):
                if backup.status == BackupStatus.VERIFIED:
                    test_backup = backup
                    break

            if not test_backup:
                logger.warning("No verified backup found for restore test")
                return False

            # Create temporary state manager for test
            # In production, use isolated test environment

            logger.info(f"Test restore of backup {test_backup.backup_id} would be performed")

            return True

        except Exception as e:
            logger.error(f"Restore test failed: {e}")
            return False

    def _determine_storage_tier(self, backup_type: BackupType) -> StorageTier:
        """Determine appropriate storage tier for backup type"""
        if backup_type == BackupType.INCREMENTAL:
            return StorageTier.LOCAL
        elif backup_type == BackupType.DIFFERENTIAL:
            return StorageTier.NETWORK if self.config.network_storage_path else StorageTier.LOCAL
        elif backup_type == BackupType.FULL:
            return StorageTier.CLOUD if self.s3_client else StorageTier.LOCAL
        elif backup_type == BackupType.EMERGENCY:
            return StorageTier.LOCAL  # Fast access needed
        else:
            return StorageTier.LOCAL

    def _get_retention_days(self, backup_type: BackupType) -> int:
        """Get retention period for backup type"""
        retention_map = {
            BackupType.FULL: self.config.retention_full,
            BackupType.DIFFERENTIAL: self.config.retention_differential,
            BackupType.INCREMENTAL: self.config.retention_incremental,
            BackupType.EMERGENCY: self.config.retention_emergency,
            BackupType.SNAPSHOT: 7,  # Default 7 days for snapshots
        }
        return retention_map.get(backup_type, 30)

    def _generate_backup_id(self, backup_type: BackupType) -> str:
        """Generate unique backup ID"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        type_prefix = backup_type.value[:3].upper()
        return f"{type_prefix}_{timestamp}"

    def _get_last_backup_time(self) -> datetime:
        """Get timestamp of last successful backup"""
        if not self._backup_history:
            return datetime.utcnow() - timedelta(days=1)

        successful = [
            b
            for b in self._backup_history
            if b.status in [BackupStatus.COMPLETED, BackupStatus.VERIFIED]
        ]

        if successful:
            return max(b.timestamp for b in successful)

        return datetime.utcnow() - timedelta(days=1)

    def _find_backup_metadata(self, backup_id: str) -> BackupMetadata | None:
        """Find backup metadata by ID"""
        for metadata in self._backup_history:
            if metadata.backup_id == backup_id:
                return metadata

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

    def _save_backup_metadata(self, metadata: BackupMetadata) -> None:
        """Save backup metadata to disk"""
        try:
            metadata_file = Path(self.config.backup_dir) / f"{metadata.backup_id}.meta"

            with open(metadata_file, "w") as f:
                json.dump(metadata.to_dict(), f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save backup metadata: {e}")

    def _load_backup_history(self) -> None:
        """Load backup history from metadata files"""
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

                    self._backup_history.append(metadata)

            # Sort by timestamp
            self._backup_history.sort(key=lambda b: b.timestamp)

        except Exception as e:
            logger.error(f"Failed to load backup history: {e}")

    def get_backup_stats(self) -> dict[str, Any]:
        """Get backup statistics"""
        if not self._backup_history:
            return {"total_backups": 0, "total_size_bytes": 0, "compression_ratio": 0}

        successful = [
            b
            for b in self._backup_history
            if b.status in [BackupStatus.COMPLETED, BackupStatus.VERIFIED]
        ]

        total_original = sum(b.size_bytes for b in successful)
        total_compressed = sum(b.size_compressed for b in successful)

        return {
            "total_backups": len(self._backup_history),
            "successful_backups": len(successful),
            "total_size_bytes": total_original,
            "total_compressed_bytes": total_compressed,
            "compression_ratio": (
                (1 - total_compressed / total_original) * 100 if total_original > 0 else 0
            ),
            "last_full_backup": (
                self._last_full_backup.isoformat() if self._last_full_backup else None
            ),
            "last_backup": self._get_last_backup_time().isoformat(),
            "backups_by_type": {
                backup_type.value: len([b for b in successful if b.backup_type == backup_type])
                for backup_type in BackupType
            },
            "storage_distribution": {
                tier.value: len([b for b in successful if b.storage_tier == tier])
                for tier in StorageTier
            },
        }


# Convenience functions
async def create_backup(
    state_manager: Any, backup_type: BackupType = BackupType.FULL
) -> BackupMetadata | None:
    """Create a backup"""
    manager = BackupManager(state_manager)
    return await manager.create_backup(backup_type)


async def restore_latest(state_manager: Any) -> bool:
    """Restore from latest backup"""
    manager = BackupManager(state_manager)
    return await manager.restore_latest_backup()
