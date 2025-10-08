"""Data Corruption recovery strategy."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from bot_v2.state.recovery.base import RecoveryStrategy

if TYPE_CHECKING:
    from bot_v2.state.recovery_handler import RecoveryOperation

logger = logging.getLogger(__name__)


class CorruptionRecoveryStrategy(RecoveryStrategy):
    """Recovers from data corruption."""

    @property
    def failure_type_name(self) -> str:
        return "Data Corruption"

    async def recover(self, operation: RecoveryOperation) -> bool:
        """Recover from data corruption"""
        try:
            logger.info("Recovering from data corruption")
            operation.actions_taken.append("Starting corruption recovery")

            # Find last valid checkpoint
            valid_checkpoint = await self.checkpoint_handler.find_valid_checkpoint()

            if not valid_checkpoint:
                logger.error("No valid checkpoint found for corruption recovery")
                return False

            # Restore from checkpoint
            success = await self.checkpoint_handler.restore_from_checkpoint(valid_checkpoint)

            if success:
                operation.actions_taken.append(
                    f"Restored from valid checkpoint {valid_checkpoint.checkpoint_id}"
                )

                # Replay transactions if available
                if await self._replay_transactions_from(valid_checkpoint.timestamp):
                    operation.actions_taken.append("Replayed transactions from checkpoint")

            return success

        except Exception as e:
            logger.error(f"Corruption recovery failed: {e}")
            operation.actions_taken.append(f"Corruption recovery error: {str(e)}")
            return False
