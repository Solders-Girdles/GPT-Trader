"""Backup data collection strategies"""

import logging
import os
from datetime import datetime, timedelta
from typing import Any

from bot_v2.state.backup.models import BackupType

logger = logging.getLogger(__name__)


class BackupCollector:
    """Collects data for backups based on backup type"""

    def __init__(self, state_manager: Any) -> None:
        self.state_manager = state_manager

    async def collect_backup_data(
        self, backup_type: BackupType, last_full_backup: datetime | None, last_backup_time: datetime
    ) -> dict[str, Any]:
        """Collect data for backup based on type"""
        backup_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "backup_type": backup_type.value,
            "system_info": {"version": "1.0.0", "environment": os.environ.get("ENV", "production")},
        }

        if backup_type == BackupType.FULL:
            # Full backup - everything
            backup_data.update(await self.collect_all_data())

        elif backup_type == BackupType.INCREMENTAL:
            # Only changes since last backup
            backup_data.update(await self.collect_changed_data(last_backup_time))

        elif backup_type == BackupType.DIFFERENTIAL:
            # Changes since last full backup
            since = last_full_backup or datetime.utcnow() - timedelta(days=30)
            backup_data.update(await self.collect_changed_data(since))

        elif backup_type == BackupType.SNAPSHOT:
            # Current state snapshot
            backup_data.update(await self.collect_snapshot_data())

        elif backup_type == BackupType.EMERGENCY:
            # Critical data only for fast backup
            backup_data.update(await self.collect_critical_data())

        return backup_data

    async def collect_all_data(self) -> dict[str, Any]:
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

    async def collect_changed_data(self, since: datetime) -> dict[str, Any]:
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

    async def collect_snapshot_data(self) -> dict[str, Any]:
        """Collect current state snapshot"""
        return {
            "positions": await self.get_all_by_pattern("position:*"),
            "orders": await self.get_all_by_pattern("order:*"),
            "portfolio": await self.state_manager.get_state("portfolio_current"),
            "performance": await self.state_manager.get_state("performance_metrics"),
        }

    async def collect_critical_data(self) -> dict[str, Any]:
        """Collect only critical data for emergency backup"""
        return {
            "positions": await self.get_all_by_pattern("position:*"),
            "portfolio": await self.state_manager.get_state("portfolio_current"),
            "critical_config": await self.state_manager.get_state("config:critical"),
        }

    async def get_all_by_pattern(self, pattern: str) -> dict[str, Any]:
        """Get all data matching pattern"""
        result = {}
        keys = await self.state_manager.get_keys_by_pattern(pattern)

        for key in keys:
            value = await self.state_manager.get_state(key)
            if value:
                result[key] = value

        return result
