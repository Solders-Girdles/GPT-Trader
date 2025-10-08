"""PostgreSQL Down recovery strategy."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from bot_v2.state.recovery.base import RecoveryStrategy

if TYPE_CHECKING:
    from bot_v2.state.recovery_handler import RecoveryOperation

logger = logging.getLogger(__name__)


class PostgresRecoveryStrategy(RecoveryStrategy):
    """Recovers from postgresql down."""

    @property
    def failure_type_name(self) -> str:
        return "PostgreSQL Down"

    async def recover(self, operation: RecoveryOperation) -> bool:
        """Recover from PostgreSQL failure"""
        try:
            logger.info("Recovering PostgreSQL from checkpoint")
            operation.actions_taken.append("Starting PostgreSQL recovery from checkpoint")

            # Get latest checkpoint
            latest_checkpoint = self.checkpoint_handler.get_latest_checkpoint()

            if not latest_checkpoint:
                # Try backup recovery
                if self.backup_manager:
                    logger.info("No checkpoint found, attempting backup recovery")
                    return await self.backup_manager.restore_latest_backup()
                return False

            # Restore from checkpoint
            success = await self.checkpoint_handler.restore_from_checkpoint(latest_checkpoint)

            operation.actions_taken.append(
                f"Restored from checkpoint {latest_checkpoint.checkpoint_id}"
            )

            # Calculate data loss
            time_since_checkpoint = datetime.utcnow() - latest_checkpoint.timestamp
            operation.data_loss_estimate = (
                f"Up to {time_since_checkpoint.total_seconds():.0f} seconds"
            )

            return success

        except Exception as e:
            logger.error(f"PostgreSQL recovery failed: {e}")
            operation.actions_taken.append(f"PostgreSQL recovery error: {str(e)}")
            return False
