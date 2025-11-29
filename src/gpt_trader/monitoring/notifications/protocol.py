"""Protocol definitions for notification backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from gpt_trader.monitoring.alert_types import Alert


@runtime_checkable
class NotificationBackend(Protocol):
    """
    Protocol for notification delivery backends.

    Implementations handle the actual dispatch of alerts to external
    systems (console, webhooks, email, SMS, etc.).
    """

    @property
    def name(self) -> str:
        """Unique identifier for this backend."""
        ...

    @property
    def is_enabled(self) -> bool:
        """Whether this backend is currently enabled."""
        ...

    async def send(self, alert: Alert) -> bool:
        """
        Send an alert through this backend.

        Args:
            alert: The alert to send

        Returns:
            True if the alert was successfully sent, False otherwise
        """
        ...

    async def test_connection(self) -> bool:
        """
        Test if this backend is properly configured and reachable.

        Returns:
            True if the backend is ready to send notifications
        """
        ...
