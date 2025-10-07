"""Checkpoint verification operations"""

import logging
from typing import Any

from bot_v2.state.checkpoint.models import Checkpoint

logger = logging.getLogger(__name__)


class CheckpointVerification:
    """Handles checkpoint verification and validation"""

    def __init__(self, state_manager: Any, storage: Any) -> None:
        self.state_manager = state_manager
        self.storage = storage

    async def verify_checkpoint(self, checkpoint: Checkpoint) -> bool:
        """Verify checkpoint validity"""
        try:
            # Recalculate hash
            calculated_hash = self.storage.calculate_consistency_hash(checkpoint.state_snapshot)

            if calculated_hash != checkpoint.consistency_hash:
                logger.error(f"Checkpoint {checkpoint.checkpoint_id} hash mismatch")
                return False

            # Verify critical data presence
            if not checkpoint.state_snapshot.get("timestamp"):
                logger.error(f"Checkpoint {checkpoint.checkpoint_id} missing timestamp")
                return False

            return True

        except Exception as e:
            logger.error(f"Checkpoint verification failed: {e}")
            return False

    async def verify_restoration(self, checkpoint: Checkpoint) -> bool:
        """Verify state was restored correctly"""
        try:
            # Verify critical data was restored
            portfolio = await self.state_manager.get_state("portfolio_current")
            if not portfolio:
                logger.warning("Portfolio data not restored")
                return False

            # Verify positions were restored
            position_keys = await self.state_manager.get_keys_by_pattern("position:*")
            expected_positions = len(checkpoint.state_snapshot.get("positions", {}))

            if len(position_keys) != expected_positions:
                logger.warning(
                    f"Position count mismatch: expected {expected_positions}, got {len(position_keys)}"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Restoration verification failed: {e}")
            return False
