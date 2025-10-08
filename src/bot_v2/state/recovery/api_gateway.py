"""API Gateway Down recovery strategy."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from bot_v2.state.recovery.base import RecoveryStrategy

if TYPE_CHECKING:
    from bot_v2.state.recovery_handler import RecoveryOperation

logger = logging.getLogger(__name__)


class APIGatewayRecoveryStrategy(RecoveryStrategy):
    """Recovers from api gateway down."""

    @property
    def failure_type_name(self) -> str:
        return "API Gateway Down"

    async def recover(self, operation: RecoveryOperation) -> bool:
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
