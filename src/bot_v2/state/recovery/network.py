"""Network Partition recovery strategy."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from bot_v2.state.recovery.base import RecoveryStrategy

if TYPE_CHECKING:
    from bot_v2.state.recovery_handler import RecoveryOperation

logger = logging.getLogger(__name__)


class NetworkRecoveryStrategy(RecoveryStrategy):
    """Recovers from network partition."""

    @property
    def failure_type_name(self) -> str:
        return "Network Partition"

async def recover(self, operation: RecoveryOperation) -> bool:
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

