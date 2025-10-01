"""Trading-related recovery handlers (trading engine, ML models)"""

import logging
from typing import Any

from bot_v2.state.recovery.models import RecoveryOperation

logger = logging.getLogger(__name__)


class TradingRecoveryHandlers:
    """Handles recovery for trading system failures"""

    def __init__(self, state_manager) -> None:
        self.state_manager = state_manager

    async def recover_trading_engine(self, operation: RecoveryOperation) -> bool:
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

    async def recover_ml_models(self, operation: RecoveryOperation) -> bool:
        """Recover from ML model failure"""
        try:
            logger.info("Recovering ML models")
            operation.actions_taken.append("Starting ML model recovery")

            # Load last known good model states
            ml_keys = await self.state_manager.get_keys_by_pattern("ml_model:*")
            if not ml_keys:
                operation.actions_taken.append(
                    "No ML models found, using baseline strategies"
                )
                await self.state_manager.set_state("system:ml_models_available", False)
                return True

            recovered_models = 0

            for key in ml_keys:
                model_state = await self.state_manager.get_state(key)
                if model_state:
                    # Reset model to last stable version
                    if "last_stable_version" in model_state:
                        model_state["current_version"] = model_state["last_stable_version"]
                        await self.state_manager.set_state(key, model_state)
                        recovered_models += 1

            operation.actions_taken.append(f"Recovered {recovered_models} ML models")

            # Fall back to baseline models if needed
            if recovered_models == 0:
                operation.actions_taken.append(
                    "No ML models recovered, using baseline strategies"
                )
                await self.state_manager.set_state("system:ml_models_available", False)

            return True  # System can operate without ML models

        except Exception as e:
            logger.error(f"ML model recovery failed: {e}")
            return False

    def _validate_position(self, position: dict[str, Any]) -> bool:
        """Validate position data structure"""
        required_fields = ["symbol", "quantity", "entry_price"]
        return all(field in position for field in required_fields)
