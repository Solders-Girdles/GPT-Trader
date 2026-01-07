"""Observability sub-container for ApplicationContainer.

This container manages observability-related dependencies:
- Notification service (alerts, webhooks)
- Health state (liveness/readiness probes)
- Secrets manager (credential storage)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gpt_trader.app.config import BotConfig
from gpt_trader.app.health_server import HealthState

if TYPE_CHECKING:
    from gpt_trader.monitoring.notifications.service import NotificationService
    from gpt_trader.security.secrets_manager import SecretsManager


class ObservabilityContainer:
    """Container for observability-related dependencies.

    This container lazily initializes notification service, health state,
    and secrets manager.

    Args:
        config: Bot configuration.
    """

    def __init__(self, config: BotConfig):
        self._config = config

        self._notification_service: NotificationService | None = None
        self._health_state: HealthState | None = None
        self._secrets_manager: SecretsManager | None = None

    @property
    def notification_service(self) -> NotificationService:
        """Get or create the notification service.

        The notification service handles alerts and webhook notifications.
        """
        if self._notification_service is None:
            from gpt_trader.monitoring.notifications import create_notification_service

            self._notification_service = create_notification_service(
                webhook_url=self._config.webhook_url,
                console_enabled=True,
            )
        return self._notification_service

    @property
    def health_state(self) -> HealthState:
        """Get or create the health state.

        The health state tracks application liveness and readiness for
        Kubernetes/Docker health probes.
        """
        if self._health_state is None:
            self._health_state = HealthState()
        return self._health_state

    @property
    def secrets_manager(self) -> SecretsManager:
        """Get or create the secrets manager.

        The secrets manager provides secure storage and retrieval of
        sensitive data including API keys and credentials. Supports
        HashiCorp Vault integration with encrypted file fallback.
        """
        if self._secrets_manager is None:
            from gpt_trader.security.secrets_manager import SecretsManager as SM

            self._secrets_manager = SM(config=self._config)
        return self._secrets_manager
