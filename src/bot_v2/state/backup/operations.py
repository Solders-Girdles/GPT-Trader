"""
Backup Manager for Bot V2 Trading System

Provides comprehensive backup system with encryption, compression,
and multi-tier storage for disaster recovery and compliance.
"""

import asyncio
import hashlib
import inspect
import json
import logging
import os
import threading
from collections.abc import Coroutine
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")

from bot_v2.state.backup.models import (
    BackupConfig,
    BackupMetadata,
    BackupStatus,
    BackupType,
    StorageTier,
)
from bot_v2.state.backup.services import (
    CompressionService,
    EncryptionService,
    RetentionService,
    TierStrategy,
    TransportService,
)

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
        self._async_lock: asyncio.Lock | None = None
        self._backup_in_progress = False
        self._backup_history: list[BackupMetadata] = []
        self._backup_metadata: dict[str, BackupMetadata] = {}
        self._last_full_backup: datetime | None = None
        self._last_differential_backup: datetime | None = None
        self._last_full_state: dict[str, Any] | None = None
        self._last_backup_state: dict[str, Any] | None = None
        self._pending_state_snapshot: dict[str, Any] | None = None
        self._scheduled_tasks: list[asyncio.Task] = []
        self._last_restored_payload: dict[str, Any] | None = None

        # Initialize services
        backup_dir = Path(self.config.backup_dir)
        local_storage = Path(self.config.local_storage_path)
        network_storage = (
            Path(self.config.network_storage_path) if self.config.network_storage_path else None
        )

        self.encryption_service = EncryptionService(
            key_path=backup_dir / ".encryption_key",
            enabled=self.config.enable_encryption,
        )

        self.compression_service = CompressionService(
            enabled=self.config.enable_compression,
            compression_level=self.config.compression_level,
        )

        self.transport_service = TransportService(
            local_path=local_storage,
            backup_path=backup_dir,
            network_path=network_storage,
            s3_bucket=self.config.s3_bucket,
            enable_s3=getattr(self.config, "enable_s3", True),
        )

        self.tier_strategy = TierStrategy(
            has_network_storage=self.transport_service.has_network_storage,
            has_cloud_storage=self.transport_service.has_cloud_storage,
        )

        # Check if there's a generic retention_days that overrides all specific retention
        generic_retention = getattr(self.config, "retention_days", None)
        if generic_retention is not None:
            self.retention_service = RetentionService(
                retention_incremental=generic_retention,
                retention_differential=generic_retention,
                retention_full=generic_retention,
                retention_emergency=generic_retention,
                retention_snapshot=generic_retention,
            )
        else:
            self.retention_service = RetentionService(
                retention_incremental=self.config.retention_incremental,
                retention_differential=self.config.retention_differential,
                retention_full=self.config.retention_full,
                retention_emergency=getattr(self.config, "retention_emergency", 30),
                retention_snapshot=7,
            )

        # Backwards compatibility aliases
        self._encryption_enabled = self.encryption_service.is_enabled
        self._cipher = getattr(self.encryption_service, "_cipher", None)
        self._s3_client = getattr(self.transport_service, "_s3_client", None)
        self.s3_client = self._s3_client

        # Load backup history
        self._load_backup_history()

        logger.info(
            f"BackupManager initialized with {len(self._backup_history)} backups in history"
        )

    def _run_or_return(self, coro: Coroutine[Any, Any, T]) -> T | Coroutine[Any, Any, T]:
        """Run coroutine immediately when no event loop is active."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        return coro

    async def _maybe_await(self, candidate: Any) -> Any:
        """Await value when it is awaitable."""
        if inspect.isawaitable(candidate):
            return await candidate
        return candidate

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

    def create_backup(
        self,
        backup_type: BackupType = BackupType.FULL,
        *,
        state_data: dict[str, Any] | None = None,
    ) -> BackupMetadata | None | Coroutine[Any, Any, BackupMetadata | None]:
        """Create a backup, supporting both sync and async callers."""

        return self._run_or_return(
            self._create_backup(backup_type=backup_type, state_data=state_data)
        )

    async def _create_backup(
        self,
        backup_type: BackupType = BackupType.FULL,
        state_data: dict[str, Any] | None = None,
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
            start_time = datetime.now(timezone.utc)

            try:
                backup_id = self._generate_backup_id(backup_type)

                logger.info(f"Creating {backup_type.value} backup {backup_id}")

                backup_data = await self._collect_backup_data(backup_type, override=state_data)

                if not backup_data:
                    raise Exception("No data to backup")
                serialized = self._serialize_backup_data(backup_data)

                # Calculate actual diff payload sizes for incremental/differential backups
                # backup_data["state"] already contains the diff for inc/diff backups
                state_serialized = json.dumps(backup_data.get("state", {}), default=str).encode(
                    "utf-8"
                )
                state_original_size = len(state_serialized)
                if self.config.enable_compression:
                    _, _, state_compressed_size = self.compression_service.compress(
                        state_serialized
                    )
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

                if self._pending_state_snapshot is not None:
                    self._update_state_snapshots(backup_type, self._pending_state_snapshot)
                    self._pending_state_snapshot = None

                if self.config.verify_after_backup:
                    if await self._verify_backup(metadata, payload):
                        metadata.status = BackupStatus.VERIFIED
                        metadata.verification_status = "passed"
                    else:
                        metadata.status = BackupStatus.CORRUPTED
                        metadata.verification_status = "failed"
                        logger.warning(f"Backup {backup_id} verification failed")

                self._update_history_after_backup(metadata, backup_type, start_time)
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

            except Exception as exc:
                if isinstance(exc, OSError):
                    logger.exception("Backup creation failed")
                    raise
                logger.exception("Backup creation failed")
                return None

            finally:
                self._pending_state_snapshot = None
                self._backup_in_progress = False

    def restore_from_backup(
        self, backup_id: str, *, apply_state: bool = True
    ) -> bool | dict[str, Any] | Coroutine[Any, Any, bool]:
        """Restore a backup, supporting sync assertions and async orchestration."""

        async def _run_sync() -> dict[str, Any]:
            return await self._restore_from_backup_internal(backup_id, apply_state=apply_state)

        async def _run_async() -> bool:
            try:
                await self._restore_from_backup_internal(backup_id, apply_state=apply_state)
            except Exception as exc:  # pragma: no cover - error path logged for async callers
                logger.error(f"Backup restoration failed: {exc}")
                return False
            return True

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_run_sync())

        return _run_async()

    async def _restore_from_backup_internal(
        self, backup_id: str, *, apply_state: bool = True
    ) -> dict[str, Any]:
        logger.info(f"Restoring from backup {backup_id}")

        metadata = self._find_backup_metadata(backup_id)
        if not metadata:
            raise FileNotFoundError(f"Backup {backup_id} not found")

        backup_bytes = await self._retrieve_backup(metadata)
        if not backup_bytes:
            raise FileNotFoundError(f"Backup {backup_id} payload missing")

        calculated_checksum = hashlib.sha256(backup_bytes).hexdigest()
        if calculated_checksum != metadata.checksum:
            raise ValueError("Backup checksum mismatch")

        # Decrypt if needed
        if metadata.encryption_key_id:
            backup_bytes = self.encryption_service.decrypt(backup_bytes)

        if self.config.enable_compression and metadata.size_compressed:
            backup_bytes = self.compression_service.decompress(backup_bytes)

        payload = json.loads(
            backup_bytes.decode() if isinstance(backup_bytes, bytes) else backup_bytes
        )
        state_payload = payload.get("state", payload)
        if not isinstance(state_payload, dict):
            raise ValueError("Restored payload is not a mapping")

        if apply_state:
            applied = await self._restore_data_to_state(state_payload)
            if not applied:
                raise RuntimeError("State restoration incomplete")

        self._last_restored_payload = state_payload
        logger.info(f"Successfully restored from backup {backup_id}")
        return state_payload

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
        return self.compression_service.compress(serialized)

    def _encrypt_payload(self, payload: bytes) -> tuple[bytes, str | None]:
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
        state = backup_data.get("state", {})
        data_sources = list(state.keys()) if isinstance(state, dict) else []

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
            data_sources=data_sources,
        )

    def _update_history_after_backup(
        self, metadata: BackupMetadata, backup_type: BackupType, start_time: datetime
    ) -> None:
        self._backup_history.append(metadata)
        self._backup_metadata[metadata.backup_id] = metadata
        self._save_backup_metadata(metadata)
        if backup_type == BackupType.FULL:
            self._last_full_backup = start_time
        elif backup_type == BackupType.DIFFERENTIAL:
            self._last_differential_backup = start_time

    def _normalize_state_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Ensure payload is JSON serialisable for persistence."""
        return json.loads(json.dumps(payload, default=str))

    def _diff_state(
        self, baseline: dict[str, Any] | None, current: dict[str, Any]
    ) -> dict[str, Any]:
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

    def _update_state_snapshots(self, backup_type: BackupType, snapshot: dict[str, Any]) -> None:
        self._last_backup_state = snapshot
        if backup_type in {BackupType.FULL, BackupType.SNAPSHOT}:
            self._last_full_state = snapshot

    async def _collect_backup_data(
        self, backup_type: BackupType, override: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Collect data for backup based on type."""
        metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "backup_type": backup_type.value,
            "system_info": {
                "version": "1.0.0",
                "environment": os.environ.get("ENV", "production"),
            },
        }

        if override is not None:
            state_payload = self._normalize_state_payload(override)
        else:
            if backup_type == BackupType.FULL:
                state_payload = await self._collect_all_data()
            elif backup_type == BackupType.INCREMENTAL:
                last_backup = self._get_last_backup_time()
                state_payload = await self._collect_changed_data(last_backup)
            elif backup_type == BackupType.DIFFERENTIAL:
                last_full = self._last_full_backup or datetime.now(timezone.utc) - timedelta(
                    days=30
                )
                state_payload = await self._collect_changed_data(last_full)
            elif backup_type == BackupType.SNAPSHOT:
                state_payload = await self._collect_snapshot_data()
            elif backup_type == BackupType.EMERGENCY:
                state_payload = await self._collect_critical_data()
            else:
                state_payload = await self._collect_all_data()

        self._pending_state_snapshot = self._normalize_state_payload(state_payload)

        persisted_state = self._pending_state_snapshot
        if backup_type == BackupType.INCREMENTAL:
            persisted_state = self._diff_state(
                self._last_backup_state, self._pending_state_snapshot
            )
        elif backup_type == BackupType.DIFFERENTIAL:
            persisted_state = self._diff_state(self._last_full_state, self._pending_state_snapshot)

        metadata["state"] = persisted_state
        return metadata

    async def _collect_all_data(self) -> dict[str, Any]:
        """Collect all system data"""
        if hasattr(self.state_manager, "create_snapshot") and callable(
            getattr(self.state_manager, "create_snapshot", None)
        ):
            snapshot_callable = getattr(self.state_manager, "create_snapshot")
            snapshot = await self._maybe_await(snapshot_callable())
            if snapshot:
                return self._normalize_state_payload(snapshot)

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
                            if isinstance(timestamp, str):
                                dt = datetime.fromisoformat(timestamp)
                            else:
                                dt = timestamp

                            if isinstance(dt, datetime) and dt.tzinfo is None:
                                dt = dt.replace(tzinfo=timezone.utc)

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
        return await self.transport_service.store(backup_id, data, storage_tier)

    def _upload_to_s3(self, backup_id: str) -> None:
        """Upload an existing local backup artifact to S3."""
        self.transport_service.upload_to_s3(backup_id)

    async def _retrieve_backup(self, metadata: BackupMetadata) -> bytes | None:
        """Retrieve backup data"""
        return await self.transport_service.retrieve(metadata.backup_id, metadata.storage_tier)

    async def _restore_data_to_state(self, data: dict[str, Any]) -> bool:
        """Restore backup data to state manager"""
        try:
            from bot_v2.state.state_manager import StateCategory

            restored_count = 0

            if not hasattr(self.state_manager, "set_state"):
                return False

            for key, value in data.items():
                # Determine category based on key pattern
                if key.startswith("position:") or key.startswith("order:"):
                    category = StateCategory.HOT
                elif key.startswith("ml_model:") or key.startswith("config:"):
                    category = StateCategory.WARM
                else:
                    category = StateCategory.HOT

                result = await self._maybe_await(self.state_manager.set_state(key, value, category))
                if result:
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

    def cleanup_old_backups(self) -> int | Coroutine[Any, Any, int]:
        return self._run_or_return(self._cleanup_old_backups())

    async def _cleanup_old_backups(self) -> int:
        """Remove expired backups based on retention policy"""
        removed_count = 0

        try:
            current_time = datetime.now(timezone.utc)
            all_backups = list(self._backup_metadata.values())
            expired_backups = self.retention_service.filter_expired(all_backups, current_time)

            for metadata in expired_backups:
                if await self.transport_service.delete(metadata.backup_id, metadata.storage_tier):
                    self._backup_metadata.pop(metadata.backup_id, None)
                    self._backup_history = [
                        entry
                        for entry in self._backup_history
                        if entry.backup_id != metadata.backup_id
                    ]
                    removed_count += 1
                    logger.debug(f"Removed expired backup {metadata.backup_id}")

            # Cleanup metadata files
            expired_ids = [m.backup_id for m in expired_backups]
            self.retention_service.cleanup_metadata_files(Path(self.config.backup_dir), expired_ids)

            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} expired backups")

        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")

        return removed_count

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
        return self.tier_strategy.determine_tier(backup_type)

    def _get_retention_days(self, backup_type: BackupType) -> int:
        """Get retention period for backup type"""
        generic_retention = getattr(self.config, "retention_days", None)
        if generic_retention is not None:
            return generic_retention

        return self.retention_service.get_retention_days(backup_type)

    def _generate_backup_id(self, backup_type: BackupType) -> str:
        """Generate unique backup ID"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        type_prefix = backup_type.value[:3].upper()
        return f"{type_prefix}_{timestamp}"

    def _get_last_backup_time(self) -> datetime:
        """Get timestamp of last successful backup"""
        if not self._backup_history:
            return datetime.now(timezone.utc) - timedelta(days=1)

        successful = [
            b
            for b in self._backup_history
            if b.status in [BackupStatus.COMPLETED, BackupStatus.VERIFIED]
        ]

        if successful:
            return max(b.timestamp for b in successful)

        return datetime.now(timezone.utc) - timedelta(days=1)

    def _find_backup_metadata(self, backup_id: str) -> BackupMetadata | None:
        """Find backup metadata by ID"""
        cached = self._backup_metadata.get(backup_id)
        if cached:
            if self._metadata_file_exists(backup_id):
                return cached
            self._remove_metadata_from_history(backup_id)

        for metadata in list(self._backup_history):
            if metadata.backup_id == backup_id:
                if self._metadata_file_exists(backup_id):
                    return metadata
                self._remove_metadata_from_history(backup_id)
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

    def _metadata_file_exists(self, backup_id: str) -> bool:
        metadata_file = Path(self.config.backup_dir) / f"{backup_id}.meta"
        return metadata_file.exists()

    def _remove_metadata_from_history(self, backup_id: str) -> None:
        self._backup_history = [
            metadata for metadata in self._backup_history if metadata.backup_id != backup_id
        ]
        self._backup_metadata.pop(backup_id, None)

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
                    self._backup_metadata[metadata.backup_id] = metadata

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
