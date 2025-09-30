"""System-related recovery handlers (memory, disk, network, API)"""

import asyncio
import logging
import os
import shutil
import tempfile

from bot_v2.state.recovery.models import RecoveryOperation

logger = logging.getLogger(__name__)


class SystemRecoveryHandlers:
    """Handles recovery for system resource and infrastructure failures"""

    def __init__(self, state_manager, checkpoint_handler) -> None:
        self.state_manager = state_manager
        self.checkpoint_handler = checkpoint_handler

    async def recover_from_memory_overflow(self, operation: RecoveryOperation) -> bool:
        """Recover from memory overflow"""
        try:
            logger.info("Recovering from memory overflow")
            operation.actions_taken.append("Starting memory recovery")

            # Clear local caches
            if hasattr(self.state_manager, "_local_cache"):
                self.state_manager._local_cache.clear()
                operation.actions_taken.append("Cleared local cache")

            # Demote data to cold storage
            hot_keys = await self.state_manager.get_keys_by_pattern("*")
            demoted_count = 0

            for key in hot_keys[:100]:  # Demote oldest 100 keys
                if await self.state_manager.demote_to_cold(key):
                    demoted_count += 1

            operation.actions_taken.append(f"Demoted {demoted_count} keys to cold storage")

            # Trigger garbage collection
            import gc

            gc.collect()
            operation.actions_taken.append("Triggered garbage collection")

            return True

        except Exception as e:
            logger.error(f"Memory recovery failed: {e}")
            return False

    async def recover_from_disk_full(self, operation: RecoveryOperation) -> bool:
        """Recover from disk full condition"""
        try:
            logger.info("Recovering from disk full")
            operation.actions_taken.append("Starting disk space recovery")

            # Clean old checkpoints
            if hasattr(self.checkpoint_handler, "_cleanup_old_checkpoints"):
                self.checkpoint_handler._cleanup_old_checkpoints()
                operation.actions_taken.append("Cleaned old checkpoints")

            # Clear temporary files
            temp_dir = tempfile.gettempdir()
            bot_temp = f"{temp_dir}/bot_v2"

            if os.path.exists(bot_temp):
                shutil.rmtree(bot_temp, ignore_errors=True)
                operation.actions_taken.append("Cleared temporary files")

            return True

        except Exception as e:
            logger.error(f"Disk recovery failed: {e}")
            return False

    async def recover_from_network_partition(self, operation: RecoveryOperation) -> bool:
        """Recover from network partition"""
        try:
            logger.info("Recovering from network partition")
            operation.actions_taken.append("Handling network partition")

            # Wait for network to stabilize
            await asyncio.sleep(5)

            # Re-establish connections
            if hasattr(self.state_manager, "_init_redis"):
                self.state_manager._init_redis()
                operation.actions_taken.append("Re-established Redis connection")

            if hasattr(self.state_manager, "_init_postgres"):
                self.state_manager._init_postgres()
                operation.actions_taken.append("Re-established PostgreSQL connection")

            # Synchronize state
            await self._synchronize_state()
            operation.actions_taken.append("Synchronized distributed state")

            return True

        except Exception as e:
            logger.error(f"Network recovery failed: {e}")
            return False

    async def recover_api_gateway(self, operation: RecoveryOperation) -> bool:
        """Recover from API gateway failure"""
        try:
            logger.info("Recovering API gateway")
            operation.actions_taken.append("Starting API gateway recovery")

            # Signal API gateway restart
            await self.state_manager.set_state("system:api_gateway_status", "restarting")

            # Clear API rate limit counters
            rate_limit_keys = await self.state_manager.get_keys_by_pattern("rate_limit:*")
            for key in rate_limit_keys:
                await self.state_manager.delete_state(key)

            operation.actions_taken.append("Cleared rate limit counters")

            # Update gateway status
            await self.state_manager.set_state("system:api_gateway_status", "recovered")
            operation.actions_taken.append("API gateway recovery completed")

            return True

        except Exception as e:
            logger.error(f"API gateway recovery failed: {e}")
            return False

    async def _synchronize_state(self) -> None:
        """Synchronize distributed state after network recovery"""
        try:
            from bot_v2.state.state_manager import StateCategory

            # Re-sync critical state across tiers
            hot_keys = await self.state_manager.get_keys_by_pattern("position:*")

            for key in hot_keys:
                value = await self.state_manager.get_state(key)
                if value:
                    # Force write to ensure consistency
                    await self.state_manager.set_state(key, value, StateCategory.HOT)

            logger.info("State synchronization completed")

        except Exception as e:
            logger.error(f"State synchronization failed: {e}")
