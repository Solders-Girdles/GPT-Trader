"""
Backup Scheduler - Manages automated backup scheduling

Extracted from BackupManager to separate scheduling concerns.
Handles periodic execution of backups, cleanup, and verification.
"""

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from bot_v2.state.backup.models import BackupConfig, BackupType

logger = logging.getLogger(__name__)


class BackupScheduler:
    """
    Manages automated backup scheduling and lifecycle.

    Responsibilities:
    - Start/stop async scheduling loops
    - Execute full/differential/incremental backups on intervals
    - Run cleanup and verification on schedule
    - Handle task cancellation and error recovery
    """

    def __init__(
        self,
        config: BackupConfig,
        create_backup_fn: Callable[[BackupType], Coroutine[Any, Any, Any]],
        cleanup_fn: Callable[[], Coroutine[Any, Any, None]],
        test_restore_fn: Callable[[], Coroutine[Any, Any, bool]],
    ) -> None:
        """
        Initialize backup scheduler.

        Args:
            config: Backup configuration with scheduling intervals
            create_backup_fn: Async function to create backup (accepts BackupType)
            cleanup_fn: Async function to cleanup old backups
            test_restore_fn: Async function to test restore capability
        """
        self.config = config
        self.create_backup = create_backup_fn
        self.cleanup_old_backups = cleanup_fn
        self.test_restore = test_restore_fn

        self._scheduled_tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        """Start automated backup scheduling."""
        if self.is_running():
            logger.warning("Scheduler already running")
            return

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

    async def stop(self) -> None:
        """Stop scheduled backups and cancel all running tasks."""
        if not self._scheduled_tasks:
            return

        for task in self._scheduled_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._scheduled_tasks.clear()
        logger.info("Scheduled backups stopped")

    def is_running(self) -> bool:
        """Check if scheduler is currently running."""
        return bool(self._scheduled_tasks) and any(
            not task.done() for task in self._scheduled_tasks
        )

    async def _run_full_backup_schedule(self) -> None:
        """Run full backup on configured interval."""
        while True:
            try:
                await asyncio.sleep(self.config.full_backup_interval_hours * 3600)
                await self.create_backup(BackupType.FULL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Full backup schedule error: {e}")

    async def _run_differential_backup_schedule(self) -> None:
        """Run differential backup on configured interval."""
        while True:
            try:
                await asyncio.sleep(self.config.differential_backup_interval_hours * 3600)
                await self.create_backup(BackupType.DIFFERENTIAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Differential backup schedule error: {e}")

    async def _run_incremental_backup_schedule(self) -> None:
        """Run incremental backup on configured interval."""
        while True:
            try:
                await asyncio.sleep(self.config.incremental_backup_interval_minutes * 60)
                await self.create_backup(BackupType.INCREMENTAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Incremental backup schedule error: {e}")

    async def _run_cleanup_schedule(self) -> None:
        """Run cleanup on daily schedule."""
        while True:
            try:
                await asyncio.sleep(86400)  # Daily
                await self.cleanup_old_backups()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup schedule error: {e}")

    async def _run_verification_schedule(self) -> None:
        """Run backup verification on configured interval."""
        while True:
            try:
                await asyncio.sleep(self.config.test_restore_frequency_days * 86400)
                await self.test_restore()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Verification schedule error: {e}")
