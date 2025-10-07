"""Trading Engine Crash recovery strategy."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from bot_v2.state.recovery.base import RecoveryStrategy

if TYPE_CHECKING:
    from bot_v2.state.recovery_handler import RecoveryOperation

logger = logging.getLogger(__name__)


class TradingEngineRecoveryStrategy(RecoveryStrategy):
    """Recovers from trading engine crash."""

    @property
    def failure_type_name(self) -> str:
        return "Trading Engine Crash"

async def recover(self, operation: RecoveryOperation) -> bool:
    """Recover from trading engine crash"""
    try:
        logger.info("Recovering trading engine")
        operation.actions_taken.append("Starting trading engine recovery")

        # Cancel all pending orders
        order_keys = await self.state_manager.get_keys_by_pattern("order:*")
        cancelled_count = 0

        for key in order_keys:
            order_data = await self.state_manager.get_state(key)
            if order_data and order_data.get("status") == "pending":
                order_data["status"] = "cancelled"
                order_data["cancel_reason"] = "Trading engine recovery"
                await self.state_manager.set_state(key, order_data)
                cancelled_count += 1

        operation.actions_taken.append(f"Cancelled {cancelled_count} pending orders")

        # Restore positions from last known state
        portfolio_data = await self.state_manager.get_state("portfolio_current")

        if portfolio_data:
            operation.actions_taken.append("Restored portfolio state")

            # Verify position consistency
            position_keys = await self.state_manager.get_keys_by_pattern("position:*")
            for key in position_keys:
                position = await self.state_manager.get_state(key)
                if position:
                    # Validate position data
                    if not self._validate_position(position):
                        logger.warning(f"Invalid position found: {key}")
                        await self.state_manager.delete_state(key)

        # Signal trading engine restart
        await self.state_manager.set_state("system:trading_engine_status", "recovered")
        operation.actions_taken.append("Trading engine recovery completed")

        return True

    except Exception as e:
        logger.error(f"Trading engine recovery failed: {e}")
        operation.actions_taken.append(f"Trading engine recovery error: {str(e)}")
        return False

