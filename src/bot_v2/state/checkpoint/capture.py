"""State capture for checkpoints"""

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class StateCapture:
    """Captures system state for checkpoints"""

    def __init__(self, state_manager: Any) -> None:
        self.state_manager = state_manager

    async def capture_system_state(self) -> dict[str, Any]:
        """Capture complete system state"""
        state = {
            "timestamp": datetime.utcnow().isoformat(),
            "positions": {},
            "orders": {},
            "portfolio": {},
            "ml_models": {},
            "configuration": {},
            "performance_metrics": {},
            "market_data_cache": {},
        }

        try:
            # Capture trading positions
            position_keys = await self.state_manager.get_keys_by_pattern("position:*")
            for key in position_keys:
                value = await self.state_manager.get_state(key)
                if value:
                    state["positions"][key] = value

            # Capture open orders
            order_keys = await self.state_manager.get_keys_by_pattern("order:*")
            for key in order_keys:
                value = await self.state_manager.get_state(key)
                if value and value.get("status") != "filled":
                    state["orders"][key] = value

            # Capture portfolio state
            portfolio_data = await self.state_manager.get_state("portfolio_current")
            if portfolio_data:
                state["portfolio"] = portfolio_data

            # Capture ML model states
            ml_keys = await self.state_manager.get_keys_by_pattern("ml_model:*")
            for key in ml_keys:
                value = await self.state_manager.get_state(key)
                if value:
                    state["ml_models"][key] = value

            # Capture configuration
            config_keys = await self.state_manager.get_keys_by_pattern("config:*")
            for key in config_keys:
                value = await self.state_manager.get_state(key)
                if value:
                    state["configuration"][key] = value

            # Capture performance metrics
            metrics_data = await self.state_manager.get_state("performance_metrics")
            if metrics_data:
                state["performance_metrics"] = metrics_data

            logger.debug(
                f"Captured state with {len(state['positions'])} positions, "
                f"{len(state['orders'])} orders"
            )

            return state

        except Exception as e:
            logger.error(f"Failed to capture system state: {e}")
            return {}
