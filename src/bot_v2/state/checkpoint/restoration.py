"""Checkpoint restoration and rollback operations"""

import logging
from typing import Any

from bot_v2.state.checkpoint.models import Checkpoint

logger = logging.getLogger(__name__)


class CheckpointRestoration:
    """Handles checkpoint restoration and rollback"""

    def __init__(self, state_manager: Any, storage: Any, verification: Any) -> None:
        self.state_manager = state_manager
        self.storage = storage
        self.verification = verification

    async def restore_from_checkpoint(self, checkpoint: Checkpoint) -> bool:
        """
        Restore system state from checkpoint.

        Args:
            checkpoint: Checkpoint to restore from

        Returns:
            Success status
        """
        try:
            logger.info(f"Restoring from checkpoint {checkpoint.checkpoint_id}")

            # Verify checkpoint integrity
            if not await self.storage.verify_checkpoint_integrity(checkpoint):
                logger.error("Checkpoint integrity check failed")
                return False

            # Pause system operations
            await self._pause_trading_operations()

            # Clear current state
            await self._clear_current_state()

            # Restore state from snapshot
            success = await self._restore_state_from_snapshot(checkpoint.state_snapshot)

            if success:
                # Verify restoration
                if await self.verification.verify_restoration(checkpoint):
                    logger.info(f"Successfully restored from checkpoint {checkpoint.checkpoint_id}")
                else:
                    logger.warning("Restoration verification failed but state was restored")
                    success = False

            # Resume operations
            await self._resume_trading_operations()

            return success

        except Exception as e:
            logger.error(f"Checkpoint restoration failed: {e}")
            await self._resume_trading_operations()
            return False

    async def _clear_current_state(self) -> None:
        """Clear current state before restoration"""
        try:
            # Clear trading positions
            position_keys = await self.state_manager.get_keys_by_pattern("position:*")
            for key in position_keys:
                await self.state_manager.delete_state(key)

            # Clear open orders
            order_keys = await self.state_manager.get_keys_by_pattern("order:*")
            for key in order_keys:
                await self.state_manager.delete_state(key)

            logger.debug("Cleared current state successfully")

        except Exception as e:
            logger.error(f"Failed to clear current state: {e}")

    async def _restore_state_from_snapshot(self, state_snapshot: dict[str, Any]) -> bool:
        """Restore state from snapshot"""
        try:
            from bot_v2.state.state_manager import StateCategory

            restored_count = 0

            # Restore positions
            for key, value in state_snapshot.get("positions", {}).items():
                if await self.state_manager.set_state(key, value, StateCategory.HOT):
                    restored_count += 1

            # Restore orders
            for key, value in state_snapshot.get("orders", {}).items():
                if await self.state_manager.set_state(key, value, StateCategory.HOT):
                    restored_count += 1

            # Restore portfolio
            portfolio = state_snapshot.get("portfolio")
            if portfolio:
                await self.state_manager.set_state(
                    "portfolio_current", portfolio, StateCategory.HOT
                )
                restored_count += 1

            # Restore ML models
            for key, value in state_snapshot.get("ml_models", {}).items():
                if await self.state_manager.set_state(key, value, StateCategory.WARM):
                    restored_count += 1

            # Restore configuration
            for key, value in state_snapshot.get("configuration", {}).items():
                if await self.state_manager.set_state(key, value, StateCategory.WARM):
                    restored_count += 1

            # Restore performance metrics
            metrics = state_snapshot.get("performance_metrics")
            if metrics:
                await self.state_manager.set_state(
                    "performance_metrics", metrics, StateCategory.WARM
                )
                restored_count += 1

            logger.info(f"Restored {restored_count} state entries")
            return restored_count > 0

        except Exception as e:
            logger.error(f"State restoration failed: {e}")
            return False

    async def _pause_trading_operations(self) -> None:
        """Pause trading operations during restoration"""
        try:
            await self.state_manager.set_state("system:trading_paused", True)
            logger.debug("Trading operations paused")
        except Exception as e:
            logger.error(f"Failed to pause trading operations: {e}")

    async def _resume_trading_operations(self) -> None:
        """Resume trading operations after restoration"""
        await self.state_manager.set_state("system:trading_paused", False)
        logger.debug("Trading operations resumed")
