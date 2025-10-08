"""Disk Full recovery strategy."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from bot_v2.state.recovery.base import RecoveryStrategy

if TYPE_CHECKING:
    from bot_v2.state.recovery_handler import RecoveryOperation

logger = logging.getLogger(__name__)


class DiskRecoveryStrategy(RecoveryStrategy):
    """Recovers from disk full."""

    @property
    def failure_type_name(self) -> str:
        return "Disk Full"

    async def recover(self, operation: RecoveryOperation) -> bool:
        """Recover from disk full condition"""
        try:
            logger.info("Recovering from disk full")
            operation.actions_taken.append("Starting disk space recovery")

            # Clean old checkpoints
            if hasattr(self.checkpoint_handler, "_cleanup_old_checkpoints"):
                self.checkpoint_handler._cleanup_old_checkpoints()
                operation.actions_taken.append("Cleaned old checkpoints")

            # Clean old backups
            if self.backup_manager:
                await self.backup_manager.cleanup_old_backups()
                operation.actions_taken.append("Cleaned old backups")

            # Clear temporary files
            import shutil
            import tempfile

            temp_dir = tempfile.gettempdir()
            bot_temp = f"{temp_dir}/bot_v2"

            if os.path.exists(bot_temp):
                shutil.rmtree(bot_temp, ignore_errors=True)
                operation.actions_taken.append("Cleared temporary files")

            return True

        except Exception as e:
            logger.error(f"Disk recovery failed: {e}")
            return False
