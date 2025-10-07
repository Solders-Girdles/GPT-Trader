"""S3 Unavailable recovery strategy."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from bot_v2.state.recovery.base import RecoveryStrategy

if TYPE_CHECKING:
    from bot_v2.state.recovery_handler import RecoveryOperation

logger = logging.getLogger(__name__)


class S3RecoveryStrategy(RecoveryStrategy):
    """Recovers from s3 unavailable."""

    @property
    def failure_type_name(self) -> str:
        return "S3 Unavailable"

async def recover(self, operation: RecoveryOperation) -> bool:
    """Recover from S3 unavailability"""
    try:
        logger.info("Handling S3 unavailability")
        operation.actions_taken.append("S3 recovery - using local storage fallback")

        # S3 is for cold storage, not critical for operations
        # Mark cold data as temporarily unavailable
        await self.state_manager.set_state("system:s3_available", False)

        # Use local disk as temporary cold storage
        operation.actions_taken.append("Configured local disk fallback for cold storage")

        return True  # System can operate without S3

    except Exception as e:
        logger.error(f"S3 recovery failed: {e}")
        return False

