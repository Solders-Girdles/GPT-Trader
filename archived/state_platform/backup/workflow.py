"""
Backup Workflow - Core backup creation pipeline

Extracted from BackupManager to separate backup creation concerns.
Handles data collection, normalization, diffing, and orchestration.
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from bot_v2.state.backup.collector import DataCollector
from bot_v2.state.backup.creator import BackupCreator
from bot_v2.state.backup.models import BackupConfig, BackupContext, BackupMetadata, BackupType

if TYPE_CHECKING:  # pragma: no cover
    from bot_v2.monitoring.metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)


class BackupWorkflow:
    """
    Manages backup creation workflow.

    Responsibilities:
    - Backup ID generation
    - Data collection orchestration (delegates to DataCollector)
    - State normalization and diffing
    - Pending snapshot tracking
    - Backup creation orchestration (delegates to BackupCreator)
    - Concurrency control via lock
    """

    def __init__(
        self,
        data_collector: DataCollector,
        backup_creator: BackupCreator,
        context: BackupContext,
        config: BackupConfig,
        backup_lock: threading.Lock,
        metrics_collector: "MetricsCollector | None" = None,
    ) -> None:
        """
        Initialize backup workflow.

        Args:
            data_collector: Service for collecting backup data
            backup_creator: Service for creating backup artifacts
            context: Shared backup context
            config: Backup configuration
            backup_lock: Threading lock for concurrency control
            metrics_collector: Optional metrics collector for telemetry
        """
        self.data_collector = data_collector
        self.backup_creator = backup_creator
        self.context = context
        self.config = config
        self._backup_lock = backup_lock
        self.metrics_collector = metrics_collector

        # Workflow state
        self._backup_in_progress = False
        self._pending_state_snapshot: dict[str, Any] | None = None

    async def create_backup(
        self,
        backup_type: BackupType = BackupType.FULL,
        state_data: dict[str, Any] | None = None,
    ) -> BackupMetadata | None:
        """
        Create backup of specified type.

        Args:
            backup_type: Type of backup to create
            state_data: Optional state data override

        Returns:
            Backup metadata or None if failed
        """
        if self._backup_in_progress:
            logger.warning("Backup already in progress")
            return None

        # Record operation start
        if self.metrics_collector:
            self.metrics_collector.record_counter("backup.operations.created_total")

        # Track duration
        start_time_monotonic = time.time()

        with self._backup_lock:
            self._backup_in_progress = True
            start_time = datetime.now(timezone.utc)

            try:
                backup_id = self._generate_backup_id(backup_type)

                # Collect backup data
                backup_data = await self._collect_backup_data(backup_type, override=state_data)

                # Delegate to backup creator for orchestration
                metadata = await self.backup_creator.create_backup_internal(
                    backup_type=backup_type,
                    backup_data=backup_data,
                    backup_id=backup_id,
                    start_time=start_time,
                    pending_snapshot=self._pending_state_snapshot,
                )

                # Record success metrics
                if metadata and self.metrics_collector:
                    self.metrics_collector.record_counter("backup.operations.created_success")

                    # Record backup size if available
                    if hasattr(metadata, "size_bytes") and metadata.size_bytes is not None:
                        self.metrics_collector.record_histogram(
                            "backup.operations.size_bytes_total", float(metadata.size_bytes)
                        )

                return metadata

            except Exception as exc:
                # Record failure metric
                if self.metrics_collector:
                    self.metrics_collector.record_counter("backup.operations.created_failed")

                if isinstance(exc, OSError):
                    logger.exception("Backup creation failed")
                    raise
                logger.exception("Backup creation failed")
                return None

            finally:
                # Record duration
                if self.metrics_collector:
                    duration_seconds = time.time() - start_time_monotonic
                    self.metrics_collector.record_histogram(
                        "backup.operations.duration_seconds", duration_seconds
                    )

                self._pending_state_snapshot = None
                self._backup_in_progress = False

    def _generate_backup_id(self, backup_type: BackupType) -> str:
        """Generate unique backup ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        type_prefix = backup_type.value[:3].upper()
        return f"{type_prefix}_{timestamp}"

    async def _collect_backup_data(
        self, backup_type: BackupType, override: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Collect data for backup based on type.

        Delegates to DataCollector for actual data gathering,
        then applies normalization and diffing logic.
        """
        metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "backup_type": backup_type.value,
            "system_info": {
                "version": "1.0.0",
                "environment": os.environ.get("ENV", "production"),
            },
        }

        # Delegate data collection to DataCollector
        state_payload = await self.data_collector.collect_for_backup(backup_type, override)

        # Normalize the collected payload
        self._pending_state_snapshot = self._normalize_state_payload(state_payload)

        # Compute diff for incremental/differential backups
        persisted_state = self._pending_state_snapshot
        if backup_type == BackupType.INCREMENTAL:
            persisted_state = self._diff_state(
                self.context.last_backup_state, self._pending_state_snapshot
            )
        elif backup_type == BackupType.DIFFERENTIAL:
            persisted_state = self._diff_state(
                self.context.last_full_state, self._pending_state_snapshot
            )

        metadata["state"] = persisted_state
        return metadata

    def _normalize_state_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Ensure payload is JSON serializable for persistence."""
        return json.loads(json.dumps(payload, default=str))

    def _diff_state(
        self, baseline: dict[str, Any] | None, current: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Calculate differential state between baseline and current.

        Args:
            baseline: Baseline state (None for full backup)
            current: Current state

        Returns:
            Differential state (or full state if no baseline)
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

    @property
    def is_backup_in_progress(self) -> bool:
        """Check if backup is currently in progress."""
        return self._backup_in_progress

    @property
    def pending_snapshot(self) -> dict[str, Any] | None:
        """Get pending state snapshot (for testing/inspection)."""
        return self._pending_state_snapshot
