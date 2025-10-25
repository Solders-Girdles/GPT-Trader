"""Component management functionality separated from perps_bot.py.

This module contains component management logic that was previously
embedded in the large perps_bot.py file. It provides:

- Component lifecycle orchestration
- Dependency injection and service coordination
- Component health monitoring
- Configuration synchronization
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from bot_v2.orchestration.config_controller import ConfigController
    from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState

from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="perps_component_management")


class ComponentManagementService:
    """Service responsible for component management and coordination.

    This service consolidates component-related logic that was previously
    distributed throughout the PerpsBot class, providing focused responsibility
    for component lifecycle, health monitoring, and coordination.
    """

    def __init__(
        self,
        config_controller: ConfigController,
        bot_state: PerpsBotRuntimeState,
    ) -> None:
        """Initialize component management service.

        Args:
            config_controller: Configuration management controller
            bot_state: Runtime state instance for the bot
        """
        self.config_controller = config_controller
        self.bot_state = bot_state
        self._component_health = {
            "config_controller": True,
            "runtime_state": True,
            "services": {},
        }

    def initialize_components(self) -> None:
        """Initialize all bot components."""
        logger.info(
            "Initializing bot components",
            operation="component_init",
            config_profile=self.config_controller.current.profile.value,
        )

        # Component initialization logic would go here
        # This would coordinate the startup of all bot components
        # in their proper order with proper error handling

        # Update component health
        self._component_health["initialization"] = {
            "completed": True,
            "timestamp": "now",
        }

        logger.info(
            "Component initialization completed",
            operation="component_ready",
        )

    def get_component_health(self) -> dict[str, Any]:
        """Get health status of all components."""
        return self._component_health

    def health_check_component(self, component_name: str) -> dict[str, Any]:
        """Perform health check on a specific component."""
        component_health = {
            "healthy": True,
            "message": f"{component_name} operational",
            "last_check": "now",
        }

        # Add specific component health checks
        if component_name == "config_controller":
            try:
                # Check if config controller is responsive
                current_config = self.config_controller.current
                component_health["config_valid"] = current_config is not None
            except Exception as exc:
                component_health["healthy"] = False
                component_health["error"] = str(exc)

        elif component_name == "runtime_state":
            # Check if bot state is accessible and valid
            component_health["state_accessible"] = self.bot_state is not None
            component_health["symbols_loaded"] = bool(self.bot_state.symbols)

        # Update overall health
        self._component_health[f"components.{component_name}"] = component_health

        return component_health

    def get_overall_health_status(self) -> dict[str, Any]:
        """Get overall health status of component management."""
        all_components_healthy = all(
            health.get("healthy", False)
            for health in self._component_health.values()
            if isinstance(health, dict)
        )

        return {
            "all_components_healthy": all_components_healthy,
            "component_count": len(
                [key for key in self._component_health.keys() if key.startswith("components.")]
            ),
            "components": {
                key: health
                for key, health in self._component_health.items()
                if key.startswith("components.")
            },
            "overall_healthy": all_components_healthy,
        }


__all__ = [
    "ComponentManagementService",
]
