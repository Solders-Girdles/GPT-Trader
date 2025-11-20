"""Risk management functionality separated from runtime coordinator.

This module contains risk manager initialization and configuration logic that was previously
embedded in the large runtime.py file. It provides:

- Risk manager bootstrap and configuration
- Risk configuration validation
- Integration with unified state management
- Runtime risk coordination
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from bot_v2.orchestration.engines.base import CoordinatorContext

from bot_v2.features.live_trade.risk import LiveRiskManager
from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="risk_management_service")


class RiskManagementService:
    """Service responsible for risk manager initialization and coordination.

    This service consolidates risk-related logic that was previously
    distributed across runtime coordinator, providing focused responsibility
    for risk management operations.
    """

    def __init__(self, context: CoordinatorContext) -> None:
        """Initialize risk management service.

        Args:
            context: Coordinator context with runtime configuration
        """
        self.context = context

    def create_risk_manager(self) -> LiveRiskManager:
        """Create and configure a risk manager instance.

        This method handles the complex risk manager initialization that was
        previously embedded in runtime coordinator, providing clean
        error handling and logging.

        Returns:
            Configured LiveRiskManager instance
        """
        config = self.context.config

        try:
            # Create risk manager with configuration
            risk_manager = LiveRiskManager(config=config)

            # Apply state management integration if available
            if hasattr(self.context, "reduce_only_state_manager"):
                risk_manager._centralized_state_manager = self.context.reduce_only_state_manager

            logger.info(
                "Risk manager created successfully",
                operation="risk_manager_creation",
                config_profile=config.profile.value,
            )

            return risk_manager

        except Exception as exc:
            logger.error(
                "Risk manager creation failed",
                operation="risk_manager_creation",
                config_profile=getattr(config, "profile", "unknown"),
                error=str(exc),
                exc_info=True,
            )
            raise

    def validate_risk_configuration(self, config: Any) -> dict[str, Any]:
        """Validate risk configuration settings.

        Args:
            config: Configuration to validate

        Returns:
            Validation results with status and any issues found
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }

        try:
            # Perform basic risk configuration validation
            if hasattr(config, "max_leverage"):
                if config.max_leverage <= 0:
                    validation_results["valid"] = False
                    validation_results["errors"].append("max_leverage must be positive")
                elif config.max_leverage > 20:
                    validation_results["valid"] = False
                    validation_results["errors"].append("max_leverage cannot exceed 20")
                    validation_results["warnings"].append("High leverage detected")

            if hasattr(config, "daily_loss_limit"):
                if config.daily_loss_limit < 0:
                    validation_results["valid"] = False
                    validation_results["errors"].append("daily_loss_limit cannot be negative")

            # Additional validation rules can be added here
            logger.debug(
                "Risk configuration validation completed",
                operation="risk_config_validation",
                validation_results=validation_results,
            )

        except Exception as exc:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Validation failed: {exc}")
            logger.error(
                "Risk configuration validation error",
                operation="risk_config_validation",
                error=str(exc),
                exc_info=True,
            )

        return validation_results

    def get_risk_health_status(self, risk_manager: LiveRiskManager) -> dict[str, Any]:
        """Get health status of the risk manager."""
        health_status = {
            "risk_manager_healthy": risk_manager is not None,
        }

        # Add risk manager specific health checks
        if risk_manager and hasattr(risk_manager, "is_healthy"):
            try:
                health_status["risk_manager_specific_health"] = risk_manager.is_healthy()
            except Exception as exc:
                health_status["risk_manager_specific_health"] = False
                health_status["risk_manager_health_error"] = str(exc)

        return health_status


__all__ = [
    "RiskManagementService",
]
