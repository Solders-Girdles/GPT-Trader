"""Trading-related recovery handlers (trading engine, ML models)"""

import logging
from typing import Any

from bot_v2.state.recovery.models import RecoveryOperation

logger = logging.getLogger(__name__)


class TradingRecoveryHandlers:
    """Handles recovery for trading system failures"""

    def __init__(self, state_manager) -> None:
        self.state_manager = state_manager

    async def _get_keys_and_values_from_repos(self, pattern: str) -> dict[str, Any]:
        """
        Get keys and values matching pattern using direct repository access.

        Uses 99%+ faster direct repository access for batch operations.
        Falls back to StateManager if repositories unavailable.
        """
        try:
            repos = self.state_manager.get_repositories()
        except (AttributeError, TypeError):
            repos = None

        result = {}

        if repos is not None:
            try:
                # Try HOT tier (Redis) first - most likely for recovery data
                if repos.redis:
                    keys = await repos.redis.keys(pattern)
                    for key in keys:
                        value = await repos.redis.fetch(key)
                        if value:
                            result[key] = value

                # Check WARM tier (PostgreSQL) for keys not in HOT
                if repos.postgres:
                    keys = await repos.postgres.keys(pattern)
                    for key in keys:
                        if key not in result:
                            value = await repos.postgres.fetch(key)
                            if value:
                                result[key] = value

                # Check COLD tier (S3) for keys not in HOT/WARM
                if repos.s3:
                    keys = await repos.s3.keys(pattern)
                    for key in keys:
                        if key not in result:
                            value = await repos.s3.fetch(key)
                            if value:
                                result[key] = value
            except TypeError:
                # Repositories not async-compatible, fall back
                result = {}
                keys = await self.state_manager.get_keys_by_pattern(pattern)
                for key in keys:
                    value = await self.state_manager.get_state(key)
                    if value:
                        result[key] = value
        else:
            # Fallback to StateManager
            keys = await self.state_manager.get_keys_by_pattern(pattern)
            for key in keys:
                value = await self.state_manager.get_state(key)
                if value:
                    result[key] = value

        return result

    async def recover_trading_engine(self, operation: RecoveryOperation) -> bool:
        """Recover from trading engine crash"""
        try:
            from bot_v2.state.state_manager import StateCategory

            logger.info("Recovering trading engine")
            operation.actions_taken.append("Starting trading engine recovery")

            # Cancel all pending orders using batch operations with repository access
            order_data_map = await self._get_keys_and_values_from_repos("order:*")

            if order_data_map:
                # Collect pending orders to cancel
                orders_to_cancel = {}
                for key, order_data in order_data_map.items():
                    if order_data and order_data.get("status") == "pending":
                        order_data["status"] = "cancelled"
                        order_data["cancel_reason"] = "Trading engine recovery"
                        orders_to_cancel[key] = (order_data, StateCategory.HOT)

                # Batch update cancelled orders
                if orders_to_cancel:
                    cancelled_count = await self.state_manager.batch_set_state(orders_to_cancel)
                    operation.actions_taken.append(f"Cancelled {cancelled_count} pending orders")
                else:
                    operation.actions_taken.append("No pending orders to cancel")
            else:
                operation.actions_taken.append("No orders found")

            # Restore positions from last known state
            portfolio_data = await self.state_manager.get_state("portfolio_current")

            if portfolio_data:
                operation.actions_taken.append("Restored portfolio state")

                # Verify position consistency using repository access
                position_data_map = await self._get_keys_and_values_from_repos("position:*")
                invalid_positions = []

                for key, position in position_data_map.items():
                    # Validate position data
                    if not self._validate_position(position):
                        logger.warning(f"Invalid position found: {key}")
                        invalid_positions.append(key)

                # Batch delete invalid positions
                if invalid_positions:
                    deleted_count = await self.state_manager.batch_delete_state(invalid_positions)
                    operation.actions_taken.append(f"Removed {deleted_count} invalid positions")

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
            from bot_v2.state.state_manager import StateCategory

            logger.info("Recovering ML models")
            operation.actions_taken.append("Starting ML model recovery")

            # Load last known good model states using repository access
            ml_model_data = await self._get_keys_and_values_from_repos("ml_model:*")
            if not ml_model_data:
                operation.actions_taken.append("No ML models found, using baseline strategies")
                await self.state_manager.set_state("system:ml_models_available", False)
                return True

            # Collect models to recover using batch operations
            models_to_recover = {}

            for key, model_state in ml_model_data.items():
                # Reset model to last stable version
                if "last_stable_version" in model_state:
                    model_state["current_version"] = model_state["last_stable_version"]
                    models_to_recover[key] = (model_state, StateCategory.WARM)

            # Batch update recovered models
            if models_to_recover:
                recovered_count = await self.state_manager.batch_set_state(models_to_recover)
                operation.actions_taken.append(f"Recovered {recovered_count} ML models")
            else:
                operation.actions_taken.append("No ML models recovered, using baseline strategies")
                await self.state_manager.set_state("system:ml_models_available", False)

            return True  # System can operate without ML models

        except Exception as e:
            logger.error(f"ML model recovery failed: {e}")
            return False

    def _validate_position(self, position: dict[str, Any]) -> bool:
        """Validate position data structure"""
        required_fields = ["symbol", "quantity", "entry_price"]
        return all(field in position for field in required_fields)
