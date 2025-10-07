"""Memory Overflow recovery strategy."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from bot_v2.state.recovery.base import RecoveryStrategy

if TYPE_CHECKING:
    from bot_v2.state.recovery_handler import RecoveryOperation

logger = logging.getLogger(__name__)


class MemoryRecoveryStrategy(RecoveryStrategy):
    """Recovers from memory overflow."""

    @property
    def failure_type_name(self) -> str:
        return "Memory Overflow"

async def recover(self, operation: RecoveryOperation) -> bool:
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

