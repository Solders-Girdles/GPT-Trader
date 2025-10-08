"""ML Model Failure recovery strategy."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from bot_v2.state.recovery.base import RecoveryStrategy

if TYPE_CHECKING:
    from bot_v2.state.recovery_handler import RecoveryOperation

logger = logging.getLogger(__name__)


class MLModelsRecoveryStrategy(RecoveryStrategy):
    """Recovers from ml model failure."""

    @property
    def failure_type_name(self) -> str:
        return "ML Model Failure"

    async def recover(self, operation: RecoveryOperation) -> bool:
        """Recover from ML model failure"""
        try:
            logger.info("Recovering ML models")
            operation.actions_taken.append("Starting ML model recovery")

            # Load last known good model states
            ml_keys = await self.state_manager.get_keys_by_pattern("ml_model:*")
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
                operation.actions_taken.append("No ML models recovered, using baseline strategies")
                await self.state_manager.set_state("system:ml_models_available", False)

            return True  # System can operate without ML models

        except Exception as e:
            logger.error(f"ML model recovery failed: {e}")
            return False
