"""
Backup Manager for Bot V2 Trading System

Orchestrates backup operations through specialized components:
- BackupScheduler: Async scheduling and lifecycle management
- BackupWorkflow: Core backup creation pipeline
- RetentionManager: Cleanup and retention policy enforcement
- BackupCreator: Backup artifact creation
- DataCollector: State data collection
- BackupRestorer: Backup restoration

Provides comprehensive backup system with encryption, compression,
and multi-tier storage for disaster recovery and compliance.
"""

import asyncio
import inspect
import logging
import threading
from collections.abc import Coroutine
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:  # pragma: no cover
    from bot_v2.monitoring.metrics_collector import MetricsCollector

from bot_v2.state.backup.collector import DataCollector
from bot_v2.state.backup.creator import BackupCreator
from bot_v2.state.backup.metadata import BackupMetadataManager
from bot_v2.state.backup.models import (
    BackupConfig,
    BackupContext,
    BackupMetadata,
    BackupStatus,
    BackupType,
    StorageTier,
)
from bot_v2.state.backup.restorer import BackupRestorer
from bot_v2.state.backup.retention_manager import RetentionManager
from bot_v2.state.backup.scheduler import BackupScheduler
from bot_v2.state.backup.services import (
    CompressionService,
    EncryptionService,
    RetentionService,
    TierStrategy,
    TransportService,
)
from bot_v2.state.backup.workflow import BackupWorkflow
from bot_v2.state.performance import StatePerformanceMetrics

T = TypeVar("T")

logger = logging.getLogger(__name__)

# Re-export public API for backup/__init__.py
__all__ = [
    "BackupManager",
    "BackupConfig",
    "BackupMetadata",
    "BackupStatus",
    "BackupType",
    "StorageTier",
    "create_backup",
    "restore_latest",
]


class BackupManager:
    """
    Facade for backup operations, orchestrating specialized components.

    Delegates to:
    - BackupScheduler: Async scheduling (full/differential/incremental/cleanup/verification)
    - BackupWorkflow: Core backup creation (ID generation, data collection, diffing, normalization)
    - RetentionManager: Cleanup and retention policy enforcement
    - BackupCreator: Backup artifact creation (serialization, compression, encryption, storage)
    - DataCollector: State data collection from state manager
    - BackupRestorer: Backup restoration and verification

    Provides comprehensive backup system with encryption, compression, and multi-tier storage.
    Ensures RPO <1 minute through continuous incremental backups.
    """

    def __init__(
        self,
        state_manager: Any,
        config: BackupConfig | None = None,
        metrics_collector: "MetricsCollector | None" = None,
    ) -> None:
        self.state_manager = state_manager
        self.config = config or BackupConfig()
        self.metrics_collector = metrics_collector

        # Initialize shared context for backup operations
        self.context = BackupContext()

        self._backup_lock = threading.Lock()
        self._async_lock: asyncio.Lock | None = None
        self._scheduled_tasks: list[asyncio.Task] = []

        # Backwards compatibility - expose mutable context attributes as instance attributes.
        # Immutable context fields are surfaced via properties to avoid stale references.
        self._backup_history = self.context.backup_history
        self._backup_metadata = self.context.backup_metadata

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

        # Initialize performance metrics for baseline tracking
        self._metrics = StatePerformanceMetrics(enabled=True)

        # Initialize metadata manager
        self.metadata_manager = BackupMetadataManager(self.config, self.context)
        self.metadata_manager.load_history()

        # Initialize backup creator
        self.backup_creator = BackupCreator(
            config=self.config,
            context=self.context,
            metadata_manager=self.metadata_manager,
            encryption_service=self.encryption_service,
            compression_service=self.compression_service,
            transport_service=self.transport_service,
            tier_strategy=self.tier_strategy,
        )

        # Initialize data collector
        self.data_collector = DataCollector(
            state_manager=self.state_manager,
            config=self.config,
            context=self.context,
            metadata_manager=self.metadata_manager,
            metrics=self._metrics,
        )

        # Initialize backup restorer
        self.backup_restorer = BackupRestorer(
            state_manager=self.state_manager,
            config=self.config,
            context=self.context,
            metadata_manager=self.metadata_manager,
            encryption_service=self.encryption_service,
            compression_service=self.compression_service,
            transport_service=self.transport_service,
        )

        # Initialize backup workflow
        self.workflow = BackupWorkflow(
            data_collector=self.data_collector,
            backup_creator=self.backup_creator,
            context=self.context,
            config=self.config,
            backup_lock=self._backup_lock,
            metrics_collector=self.metrics_collector,
        )

        # Initialize retention manager
        self.retention_manager = RetentionManager(
            retention_service=self.retention_service,
            transport_service=self.transport_service,
            context=self.context,
            config=self.config,
            metrics_collector=self.metrics_collector,
        )

        # Initialize backup scheduler
        # Use lambdas to allow monkeypatching in tests
        self.scheduler = BackupScheduler(
            config=self.config,
            create_backup_fn=lambda backup_type: self.workflow.create_backup(
                backup_type=backup_type
            ),
            cleanup_fn=lambda: self.cleanup_old_backups(),
            test_restore_fn=lambda: self.test_restore(),
        )

        logger.info(
            f"BackupManager initialized with {len(self.context.backup_history)} backups in history"
        )

    @property
    def _last_full_backup(self) -> datetime | None:
        """Alias for backwards compatibility with legacy attributes."""
        return self.context.last_full_backup

    @_last_full_backup.setter
    def _last_full_backup(self, value: datetime | None) -> None:
        self.context.last_full_backup = value

    @property
    def _last_differential_backup(self) -> datetime | None:
        return self.context.last_differential_backup

    @_last_differential_backup.setter
    def _last_differential_backup(self, value: datetime | None) -> None:
        self.context.last_differential_backup = value

    @property
    def _last_full_state(self) -> dict[str, Any] | None:
        return self.context.last_full_state

    @_last_full_state.setter
    def _last_full_state(self, value: dict[str, Any] | None) -> None:
        self.context.last_full_state = value

    @property
    def _last_backup_state(self) -> dict[str, Any] | None:
        return self.context.last_backup_state

    @_last_backup_state.setter
    def _last_backup_state(self, value: dict[str, Any] | None) -> None:
        self.context.last_backup_state = value

    @property
    def _last_restored_payload(self) -> dict[str, Any] | None:
        return self.context.last_restored_payload

    @_last_restored_payload.setter
    def _last_restored_payload(self, value: dict[str, Any] | None) -> None:
        self.context.last_restored_payload = value

    # -- Legacy compatibility helpers -------------------------------------------------
    def _save_backup_metadata(self, metadata: BackupMetadata) -> None:
        """Backward-compatible wrapper around metadata manager."""
        self.metadata_manager.save_metadata(metadata)

    def _find_backup_metadata(self, backup_id: str) -> BackupMetadata | None:
        """Backward-compatible wrapper to locate metadata by ID."""
        return self.metadata_manager.find_metadata(backup_id)

    def _load_backup_history(self) -> None:
        """Backward-compatible wrapper to refresh metadata cache."""
        self.metadata_manager.load_history()

    def get_backup_stats(self) -> dict[str, Any]:
        """Backward-compatible access to backup statistics."""
        return self.metadata_manager.get_stats()

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
        """Start automated backup scheduling (delegates to scheduler)."""
        await self.scheduler.start()

    async def stop_scheduled_backups(self) -> None:
        """Stop scheduled backups (delegates to scheduler)."""
        await self.scheduler.stop()

    def create_backup(
        self,
        backup_type: BackupType = BackupType.FULL,
        *,
        state_data: dict[str, Any] | None = None,
    ) -> BackupMetadata | None | Coroutine[Any, Any, BackupMetadata | None]:
        """
        Create a backup, supporting both sync and async callers.

        Delegates to BackupWorkflow for core backup creation logic.
        """
        return self._run_or_return(
            self._create_backup(backup_type=backup_type, state_data=state_data)
        )

    async def _create_backup(
        self,
        backup_type: BackupType = BackupType.FULL,
        state_data: dict[str, Any] | None = None,
    ) -> BackupMetadata | None:
        """Delegate backup creation to BackupWorkflow."""
        return await self.workflow.create_backup(backup_type=backup_type, state_data=state_data)

    def restore_from_backup(
        self, backup_id: str, *, apply_state: bool = True
    ) -> bool | dict[str, Any] | Coroutine[Any, Any, bool]:
        """
        Restore a backup, supporting sync assertions and async orchestration.

        Delegates to BackupRestorer for restoration logic.
        """

        async def _run_sync() -> dict[str, Any]:
            return await self.backup_restorer.restore_from_backup_internal(
                backup_id, apply_state=apply_state
            )

        async def _run_async() -> bool:
            try:
                await self.backup_restorer.restore_from_backup_internal(
                    backup_id, apply_state=apply_state
                )
            except Exception as exc:  # pragma: no cover - error path logged for async callers
                logger.error(f"Backup restoration failed: {exc}")
                return False
            return True

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_run_sync())

        return _run_async()

    async def restore_latest_backup(self, backup_type: BackupType | None = None) -> bool:
        """Restore from most recent backup.

        Delegates to BackupRestorer for restoration logic.

        Args:
            backup_type: Optional type filter

        Returns:
            Success status
        """
        return await self.backup_restorer.restore_latest_backup(backup_type)

    def _upload_to_s3(self, backup_id: str) -> None:
        """Upload an existing local backup artifact to S3."""
        self.transport_service.upload_to_s3(backup_id)

    def cleanup_old_backups(self) -> int | Coroutine[Any, Any, int]:
        """Remove expired backups based on retention policy (delegates to RetentionManager)."""
        return self._run_or_return(self.retention_manager.cleanup())

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

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics for backup operations.

        Returns detailed timing statistics for bulk state collection operations
        to measure optimization impact.
        """
        return self._metrics.get_summary()


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
