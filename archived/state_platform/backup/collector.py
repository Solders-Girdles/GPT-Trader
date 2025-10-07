"""Data collection for backup operations.

Handles gathering state data from various sources (state manager, repositories)
for different backup types (full, incremental, differential, snapshot, emergency).
"""

from __future__ import annotations

import inspect
import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from bot_v2.state.backup.metadata import BackupMetadataManager
    from bot_v2.state.backup.models import BackupConfig, BackupContext
    from bot_v2.state.performance import StatePerformanceMetrics

from bot_v2.state.backup.models import BackupType

logger = logging.getLogger(__name__)


class DataCollector:
    """Collects state data for backup operations.

    Provides strategy pattern for different backup types:
    - FULL: Collects all state data
    - INCREMENTAL: Collects changes since last backup
    - DIFFERENTIAL: Collects changes since last full backup
    - SNAPSHOT: Collects current positions/orders/portfolio
    - EMERGENCY: Collects only critical data
    """

    def __init__(
        self,
        state_manager: Any,
        config: BackupConfig,
        context: BackupContext,
        metadata_manager: BackupMetadataManager,
        metrics: StatePerformanceMetrics,
    ):
        """Initialize data collector.

        Args:
            state_manager: State manager with get_state, get_keys_by_pattern methods
            config: Backup configuration
            context: Shared backup context (for last_full_backup timestamp)
            metadata_manager: Metadata manager (for get_last_backup_time)
            metrics: Performance metrics tracker
        """
        self.state_manager = state_manager
        self.config = config
        self.context = context
        self.metadata_manager = metadata_manager
        self.metrics = metrics

    async def collect_for_backup(
        self, backup_type: BackupType, override: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Collect data for backup based on type.

        Args:
            backup_type: Type of backup to create
            override: Optional state data override (for testing)

        Returns:
            Collected backup data with metadata and state
        """
        state_payload: dict[str, Any]

        if override is not None:
            state_payload = override
        else:
            if backup_type == BackupType.FULL:
                state_payload = await self._collect_all_data()
            elif backup_type == BackupType.INCREMENTAL:
                last_backup = self.metadata_manager.get_last_backup_time()
                state_payload = await self._collect_changed_data(last_backup)
            elif backup_type == BackupType.DIFFERENTIAL:
                last_full = self.context.last_full_backup or datetime.now(timezone.utc) - timedelta(
                    days=30
                )
                state_payload = await self._collect_changed_data(last_full)
            elif backup_type == BackupType.SNAPSHOT:
                state_payload = await self._collect_snapshot_data()
            elif backup_type == BackupType.EMERGENCY:
                state_payload = await self._collect_critical_data()
            else:
                state_payload = await self._collect_all_data()

        return state_payload

    async def _collect_all_data(self) -> dict[str, Any]:
        """Collect all system data."""
        if hasattr(self.state_manager, "create_snapshot") and callable(
            getattr(self.state_manager, "create_snapshot", None)
        ):
            snapshot_callable = getattr(self.state_manager, "create_snapshot")
            snapshot = await self._maybe_await(snapshot_callable())
            if snapshot:
                if not isinstance(snapshot, dict):
                    raise TypeError("create_snapshot must return dict[str, Any]")
                return cast(dict[str, Any], snapshot)

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

        # Use direct repository access for batch operations (99%+ faster)
        # Fall back to StateManager if repositories unavailable or not async-compatible
        try:
            repos = self.state_manager.get_repositories()
        except (AttributeError, TypeError):
            repos = None

        with self.metrics.time_operation("backup.collect_all_data"):
            if repos is not None:
                try:
                    # Direct repository access (fast path)
                    for pattern in patterns:
                        # Try HOT tier (Redis) first
                        if repos.redis:
                            keys = await repos.redis.keys(pattern)
                            for key in keys:
                                value = await repos.redis.fetch(key)
                                if value:
                                    data[key] = value

                        # Check WARM tier (PostgreSQL) for keys not in HOT
                        if repos.postgres:
                            keys = await repos.postgres.keys(pattern)
                            for key in keys:
                                if key not in data:  # Skip if already found in HOT
                                    value = await repos.postgres.fetch(key)
                                    if value:
                                        data[key] = value

                        # Check COLD tier (S3) for keys not in HOT/WARM
                        if repos.s3:
                            keys = await repos.s3.keys(pattern)
                            for key in keys:
                                if key not in data:  # Skip if already found in HOT/WARM
                                    value = await repos.s3.fetch(key)
                                    if value:
                                        data[key] = value
                except TypeError:
                    # Repositories exist but aren't async-compatible (e.g., Mocks)
                    # Fall back to StateManager
                    data = {}
                    for pattern in patterns:
                        keys = await self.state_manager.get_keys_by_pattern(pattern)
                        for key in keys:
                            value = await self.state_manager.get_state(key)
                            if value:
                                data[key] = value
            else:
                # Fallback: StateManager access (slower but compatible)
                for pattern in patterns:
                    keys = await self.state_manager.get_keys_by_pattern(pattern)
                    for key in keys:
                        value = await self.state_manager.get_state(key)
                        if value:
                            data[key] = value

        logger.debug(f"Collected {len(data)} items for full backup")

        return data

    async def _collect_changed_data(self, since: datetime) -> dict[str, Any]:
        """Collect data changed since given time."""
        data = {}

        # Get recently modified keys
        # This would ideally track modification times
        patterns = ["position:*", "order:*", "portfolio*"]

        # Use direct repository access for batch operations (99%+ faster)
        # Fall back to StateManager if repositories unavailable or not async-compatible
        try:
            repos = self.state_manager.get_repositories()
        except (AttributeError, TypeError):
            repos = None

        def _check_timestamp(value: Any) -> bool:
            """Helper to check if value's timestamp is after 'since'."""
            if not (value and isinstance(value, dict)):
                return False

            timestamp = value.get("timestamp") or value.get("last_updated")
            if not timestamp:
                return True  # Include if no timestamp

            try:
                if isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp)
                else:
                    dt = timestamp

                if isinstance(dt, datetime) and dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)

                return dt > since
            except (TypeError, ValueError) as exc:
                logger.debug(
                    "Unable to parse timestamp %s: %s",
                    timestamp,
                    exc,
                    exc_info=True,
                )
                return True  # Include if timestamp is malformed

        with self.metrics.time_operation("backup.collect_changed_data"):
            if repos is not None:
                try:
                    # Direct repository access (fast path)
                    for pattern in patterns:
                        # Collect from HOT tier (Redis)
                        if repos.redis:
                            keys = await repos.redis.keys(pattern)
                            for key in keys:
                                value = await repos.redis.fetch(key)
                                if _check_timestamp(value):
                                    data[key] = value

                        # Check WARM tier for keys not already found
                        if repos.postgres:
                            keys = await repos.postgres.keys(pattern)
                            for key in keys:
                                if key not in data:
                                    value = await repos.postgres.fetch(key)
                                    if _check_timestamp(value):
                                        data[key] = value

                        # Check COLD tier (S3) for keys not in HOT/WARM
                        if repos.s3:
                            keys = await repos.s3.keys(pattern)
                            for key in keys:
                                if key not in data:
                                    value = await repos.s3.fetch(key)
                                    if _check_timestamp(value):
                                        data[key] = value
                except TypeError:
                    # Repositories exist but aren't async-compatible (e.g., Mocks)
                    # Fall back to StateManager
                    data = {}
                    for pattern in patterns:
                        keys = await self.state_manager.get_keys_by_pattern(pattern)
                        for key in keys:
                            value = await self.state_manager.get_state(key)
                            if _check_timestamp(value):
                                data[key] = value
            else:
                # Fallback: StateManager access (slower but compatible)
                for pattern in patterns:
                    keys = await self.state_manager.get_keys_by_pattern(pattern)
                    for key in keys:
                        value = await self.state_manager.get_state(key)
                        if _check_timestamp(value):
                            data[key] = value

        logger.debug(f"Collected {len(data)} changed items since {since}")

        return data

    async def _collect_snapshot_data(self) -> dict[str, Any]:
        """Collect current state snapshot."""
        return {
            "positions": await self._get_all_by_pattern("position:*"),
            "orders": await self._get_all_by_pattern("order:*"),
            "portfolio": await self.state_manager.get_state("portfolio_current"),
            "performance": await self.state_manager.get_state("performance_metrics"),
        }

    async def _collect_critical_data(self) -> dict[str, Any]:
        """Collect only critical data for emergency backup."""
        return {
            "positions": await self._get_all_by_pattern("position:*"),
            "portfolio": await self.state_manager.get_state("portfolio_current"),
            "critical_config": await self.state_manager.get_state("config:critical"),
        }

    async def _get_all_by_pattern(self, pattern: str) -> dict[str, Any]:
        """Get all data matching pattern."""
        result = {}

        # Use direct repository access for batch operations (99%+ faster)
        # Fall back to StateManager if repositories unavailable or not async-compatible
        try:
            repos = self.state_manager.get_repositories()
        except (AttributeError, TypeError):
            repos = None

        with self.metrics.time_operation("backup.get_all_by_pattern"):
            if repos is not None:
                try:
                    # Direct repository access (fast path)
                    # Try HOT tier (Redis) first
                    if repos.redis:
                        keys = await repos.redis.keys(pattern)
                        for key in keys:
                            value = await repos.redis.fetch(key)
                            if value:
                                result[key] = value

                    # Check WARM tier (PostgreSQL) for keys not in HOT
                    if repos.postgres:
                        keys = await repos.postgres.keys(pattern)
                        for key in keys:
                            if key not in result:  # Skip if already found in HOT
                                value = await repos.postgres.fetch(key)
                                if value:
                                    result[key] = value

                    # Check COLD tier (S3) for keys not in HOT/WARM
                    if repos.s3:
                        keys = await repos.s3.keys(pattern)
                        for key in keys:
                            if key not in result:  # Skip if already found in HOT/WARM
                                value = await repos.s3.fetch(key)
                                if value:
                                    result[key] = value
                except TypeError:
                    # Repositories exist but aren't async-compatible (e.g., Mocks)
                    # Fall back to StateManager
                    result = {}
                    keys = await self.state_manager.get_keys_by_pattern(pattern)
                    for key in keys:
                        value = await self.state_manager.get_state(key)
                        if value:
                            result[key] = value
            else:
                # Fallback: StateManager access (slower but compatible)
                keys = await self.state_manager.get_keys_by_pattern(pattern)
                for key in keys:
                    value = await self.state_manager.get_state(key)
                    if value:
                        result[key] = value

        return result

    async def _maybe_await(self, candidate: Any) -> Any:
        """Await value when it is awaitable."""
        if inspect.isawaitable(candidate):
            return await candidate
        return candidate
