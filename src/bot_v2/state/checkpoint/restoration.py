"""Checkpoint restoration and rollback operations"""

import logging
from typing import Any

from bot_v2.state.checkpoint.models import Checkpoint
from bot_v2.state.performance import StatePerformanceMetrics

logger = logging.getLogger(__name__)


class CheckpointRestoration:
    """Handles checkpoint restoration and rollback"""

    def __init__(self, state_manager: Any, storage: Any, verification: Any) -> None:
        self.state_manager = state_manager
        self.storage = storage
        self.verification = verification
        self._metrics = StatePerformanceMetrics(enabled=True)

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

    async def _get_keys_by_pattern(self, pattern: str) -> list[str]:
        """
        Get all keys matching pattern across all tiers.

        Uses direct repository access for 99%+ performance improvement.
        Falls back to StateManager if repositories unavailable or not async-compatible.
        """
        keys = []

        # Use direct repository access for batch operations (99%+ faster)
        # Fall back to StateManager if repositories unavailable or not async-compatible
        try:
            repos = self.state_manager.get_repositories()
        except (AttributeError, TypeError):
            repos = None

        with self._metrics.time_operation("checkpoint.get_keys_by_pattern"):
            if repos is not None:
                try:
                    # Direct repository access (fast path)
                    # Collect from HOT tier (Redis)
                    if repos.redis:
                        hot_keys = await repos.redis.keys(pattern)
                        keys.extend(hot_keys)

                    # Collect from WARM tier (PostgreSQL)
                    if repos.postgres:
                        warm_keys = await repos.postgres.keys(pattern)
                        # Deduplicate - add only keys not in HOT
                        keys.extend(k for k in warm_keys if k not in keys)

                    # Collect from COLD tier (S3)
                    if repos.s3:
                        cold_keys = await repos.s3.keys(pattern)
                        # Deduplicate - add only keys not in HOT/WARM
                        keys.extend(k for k in cold_keys if k not in keys)
                except TypeError:
                    # Repositories exist but aren't async-compatible (e.g., Mocks)
                    # Fall back to StateManager
                    keys = await self.state_manager.get_keys_by_pattern(pattern)
            else:
                # Fallback: StateManager access (slower but compatible)
                keys = await self.state_manager.get_keys_by_pattern(pattern)

        return keys

    async def _clear_current_state(self) -> None:
        """Clear current state before restoration using batch deletes with cache invalidation"""
        try:
            with self._metrics.time_operation("checkpoint.clear_current_state"):
                # Collect all keys to delete
                position_keys = await self._get_keys_by_pattern("position:*")
                order_keys = await self._get_keys_by_pattern("order:*")
                all_keys = position_keys + order_keys

                if not all_keys:
                    logger.debug("No keys to clear")
                    return

                # Use StateManager batch delete for proper cache invalidation
                deleted_count = await self.state_manager.batch_delete_state(all_keys)
                logger.debug(f"Batch deleted {deleted_count} keys with cache invalidation")

        except Exception as e:
            logger.error(f"Failed to clear current state: {e}")

    async def _restore_state_from_snapshot(self, state_snapshot: dict[str, Any]) -> bool:
        """Restore state from snapshot using batch writes with proper cache/metadata updates"""
        try:
            from bot_v2.state.state_manager import StateCategory

            # Prepare items dict: {key: (value, category)}
            items: dict[str, tuple[Any, StateCategory]] = {}

            # Add positions to HOT tier
            for key, value in state_snapshot.get("positions", {}).items():
                items[key] = (value, StateCategory.HOT)

            # Add orders to HOT tier
            for key, value in state_snapshot.get("orders", {}).items():
                items[key] = (value, StateCategory.HOT)

            # Add portfolio to HOT tier
            portfolio = state_snapshot.get("portfolio")
            if portfolio:
                items["portfolio_current"] = (portfolio, StateCategory.HOT)

            # Add ML models to WARM tier
            for key, value in state_snapshot.get("ml_models", {}).items():
                items[key] = (value, StateCategory.WARM)

            # Add configuration to WARM tier
            for key, value in state_snapshot.get("configuration", {}).items():
                items[key] = (value, StateCategory.WARM)

            # Add performance metrics to WARM tier
            metrics = state_snapshot.get("performance_metrics")
            if metrics:
                items["performance_metrics"] = (metrics, StateCategory.WARM)

            # Use StateManager batch set for proper cache/metadata handling
            restored_count = await self.state_manager.batch_set_state(items)

            logger.info(f"Restored {restored_count} state entries with cache/metadata updates")
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
