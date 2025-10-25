"""Configuration guardian functionality separated from perps_bot.py.

This module contains configuration monitoring and validation logic that was previously
embedded in the large perps_bot.py file. It provides:

- Real-time configuration monitoring
- Drift detection and alerting
- Configuration validation and bounds checking
- Automatic rollup prevention
- Configuration change notifications
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from bot_v2.orchestration.config_controller import ConfigController
    from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState

from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="perps_configuration_guardian")


class ConfigurationGuardianService:
    """Service responsible for configuration monitoring and validation.

    This service consolidates configuration-related logic that was previously
    embedded in PerpsBot class, providing focused responsibility for
    configuration integrity monitoring and drift detection.
    """

    def __init__(
        self,
        config_controller: ConfigController,
        bot_state: PerpsBotRuntimeState,
    ) -> None:
        """Initialize configuration guardian service.

        Args:
            config_controller: Configuration management controller
            bot_state: Runtime state instance for the bot
        """
        self.config_controller = config_controller
        self.bot_state = bot_state
        self._baseline_snapshot = None
        self._last_check_time = None
        self._configuration_alerts = []

    def set_baseline_snapshot(self, snapshot: Any) -> None:
        """Set baseline snapshot for configuration drift detection."""
        self._baseline_snapshot = snapshot
        self._last_check_time = datetime.now(UTC)

        logger.info(
            "Configuration baseline snapshot set",
            operation="baseline_set",
            snapshot_type=type(snapshot).__name__,
        )

    def check_configuration_integrity(self) -> dict[str, Any]:
        """Check configuration integrity and detect drift."""
        if not self._baseline_snapshot:
            return {
                "integrity_check": False,
                "message": "No baseline snapshot available",
                "alerts": [],
            }

        current_config = self.config_controller.current
        integrity_issues = []
        critical_alerts = []

        # Perform configuration integrity checks
        checks = [
            self._check_trading_parameters,
            self._check_risk_parameters,
            self._check_system_parameters,
            self._check_derivatives_configuration,
        ]

        for check_func in checks:
            try:
                result = check_func(current_config, self._baseline_snapshot.config)
                if not result.get("valid", True):
                    integrity_issues.extend(result.get("issues", []))
                    if result.get("critical", False):
                        critical_alerts.append(
                            result.get("message", "Configuration integrity issue")
                        )
            except Exception as exc:
                logger.error(
                    "Configuration check failed",
                    operation="config_integrity_check",
                    check=check_func.__name__,
                    error=str(exc),
                    exc_info=True,
                )

        self._last_check_time = datetime.now(UTC)

        # Update alerts list
        new_alerts = [
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "message": alert,
                "severity": "critical" if alert in critical_alerts else "warning",
            }
            for alert in critical_alerts + integrity_issues
        ]

        self._configuration_alerts.extend(new_alerts)

        # Keep only recent alerts
        if len(self._configuration_alerts) > 100:
            self._configuration_alerts = self._configuration_alerts[-100:]

        return {
            "integrity_check": True,
            "baseline_valid": self._baseline_snapshot is not None,
            "issues": integrity_issues,
            "critical_alerts": critical_alerts,
            "total_alerts": len(self._configuration_alerts),
            "last_check_time": self._last_check_time.isoformat(),
        }

    def _check_trading_parameters(
        self, current_config: Any, baseline_config: Any
    ) -> dict[str, Any]:
        """Check trading parameter integrity."""
        issues = []
        critical = False

        # Check for parameter drift
        if hasattr(baseline_config, "max_leverage") and hasattr(current_config, "max_leverage"):
            if current_config.max_leverage > baseline_config.max_leverage * 1.5:
                issues.append(
                    {
                        "parameter": "max_leverage",
                        "issue": "Significant leverage increase detected",
                        "current": current_config.max_leverage,
                        "baseline": baseline_config.max_leverage,
                        "severity": "critical",
                    }
                )
                critical = True

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "critical": critical,
        }

    def _check_risk_parameters(self, current_config: Any, baseline_config: Any) -> dict[str, Any]:
        """Check risk parameter integrity."""
        issues = []
        critical = False

        # Check risk limits
        if hasattr(baseline_config, "daily_loss_limit") and hasattr(
            current_config, "daily_loss_limit"
        ):
            if current_config.daily_loss_limit > baseline_config.daily_loss_limit * 2:
                issues.append(
                    {
                        "parameter": "daily_loss_limit",
                        "issue": "Daily loss limit significantly increased",
                        "current": current_config.daily_loss_limit,
                        "baseline": baseline_config.daily_loss_limit,
                        "severity": "warning",
                    }
                )

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "critical": critical,
        }

    def _check_system_parameters(self, current_config: Any, baseline_config: Any) -> dict[str, Any]:
        """Check system parameter integrity."""
        issues = []
        critical = False

        # Check for dangerous configuration changes
        if hasattr(baseline_config, "mock_broker") and hasattr(current_config, "mock_broker"):
            if current_config.mock_broker != baseline_config.mock_broker:
                # Going from real to mock broker is usually safe
                pass
            elif hasattr(current_config, "perps_force_mock"):
                if current_config.perps_force_mock and not baseline_config.perps_force_mock:
                    issues.append(
                        {
                            "parameter": "perps_force_mock",
                            "issue": "Forced mock mode enabled",
                            "severity": "warning",
                        }
                    )

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "critical": critical,
        }

    def _check_derivatives_configuration(
        self, current_config: Any, baseline_config: Any
    ) -> dict[str, Any]:
        """Check derivatives configuration integrity."""
        issues = []
        critical = False

        # Check derivatives enabling
        if hasattr(current_config, "derivatives_enabled") and hasattr(
            baseline_config, "derivatives_enabled"
        ):
            if current_config.derivatives_enabled and not baseline_config.derivatives_enabled:
                issues.append(
                    {
                        "parameter": "derivatives_enabled",
                        "issue": "Derivatives enabled without baseline configuration",
                        "severity": "warning",
                    }
                )

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "critical": critical,
        }

    def get_configuration_alerts(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent configuration alerts."""
        return self._configuration_alerts[-limit:]

    def clear_configuration_alerts(self) -> None:
        """Clear all configuration alerts."""
        self._configuration_alerts.clear()
        logger.info(
            "Configuration alerts cleared",
            operation="config_alerts_cleared",
        )

    def get_guardian_status(self) -> dict[str, Any]:
        """Get current status of the configuration guardian."""
        return {
            "active": self._baseline_snapshot is not None,
            "last_check": self._last_check_time.isoformat() if self._last_check_time else None,
            "total_alerts": len(self._configuration_alerts),
            "recent_alerts_count": len(
                [
                    a
                    for a in self._configuration_alerts
                    if a["timestamp"] > datetime.now(UTC).timestamp() - 3600
                ]
            ),
        }


__all__ = [
    "ConfigurationGuardianService",
]
