"""Notification service for dispatching alerts to multiple backends."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from gpt_trader.monitoring.alert_types import Alert, AlertSeverity
from gpt_trader.monitoring.notifications.protocol import NotificationBackend
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="notifications")


@dataclass
class NotificationService:
    """
    Central service for dispatching alerts to multiple notification backends.

    Features:
    - Multiple backend support (console, webhook, file, etc.)
    - Rate limiting to prevent alert storms
    - Alert deduplication within a time window
    - Severity filtering
    - Async dispatch for non-blocking operation
    """

    backends: list[NotificationBackend] = field(default_factory=list)
    min_severity: AlertSeverity = AlertSeverity.WARNING
    rate_limit_per_minute: int = 30
    dedup_window_seconds: int = 300  # 5 minutes
    enabled: bool = True

    # Internal state
    _recent_alerts: dict[str, datetime] = field(default_factory=dict, repr=False)
    _sent_count_this_minute: int = field(default=0, repr=False)
    _minute_reset_time: datetime = field(default_factory=datetime.utcnow, repr=False)

    def add_backend(self, backend: NotificationBackend) -> None:
        """Add a notification backend."""
        self.backends.append(backend)
        logger.info(f"Added notification backend: {backend.name}")

    def remove_backend(self, name: str) -> bool:
        """Remove a backend by name."""
        for i, backend in enumerate(self.backends):
            if backend.name == name:
                self.backends.pop(i)
                logger.info(f"Removed notification backend: {name}")
                return True
        return False

    async def notify(
        self,
        title: str,
        message: str,
        severity: AlertSeverity | str = AlertSeverity.WARNING,
        source: str | None = None,
        category: str | None = None,
        context: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        force: bool = False,
    ) -> bool:
        """
        Send a notification to all enabled backends.

        Args:
            title: Alert title (short description)
            message: Alert message (detailed description)
            severity: Alert severity level
            source: Component that generated the alert
            category: Alert category for grouping
            context: Additional context data
            metadata: Additional metadata
            force: Bypass rate limiting and deduplication

        Returns:
            True if at least one backend succeeded
        """
        if not self.enabled:
            return False

        # Coerce severity
        if isinstance(severity, str):
            severity = AlertSeverity.coerce(severity)

        # Severity filter
        if severity.numeric_level < self.min_severity.numeric_level:
            return True  # Filtered, not failed

        # Rate limiting
        if not force and not self._check_rate_limit():
            logger.warning("Rate limit exceeded, notification dropped")
            return False

        # Deduplication
        dedup_key = f"{title}:{source}:{category}"
        if not force and self._is_duplicate(dedup_key):
            logger.debug(f"Duplicate alert suppressed: {title}")
            return True  # Deduplicated, not failed

        # Create alert
        alert = Alert(
            severity=severity,
            title=title,
            message=message,
            source=source,
            category=category,
            context=context or {},
            metadata=metadata or {},
        )

        # Dispatch to all backends
        success = await self._dispatch(alert)

        # Track for deduplication
        self._recent_alerts[dedup_key] = datetime.utcnow()
        self._cleanup_old_alerts()

        return success

    async def notify_alert(self, alert: Alert, force: bool = False) -> bool:
        """
        Send a pre-constructed Alert to all enabled backends.

        Args:
            alert: The alert to send
            force: Bypass rate limiting and deduplication

        Returns:
            True if at least one backend succeeded
        """
        if not self.enabled:
            return False

        if alert.severity.numeric_level < self.min_severity.numeric_level:
            return True

        if not force and not self._check_rate_limit():
            logger.warning("Rate limit exceeded, notification dropped")
            return False

        dedup_key = f"{alert.title}:{alert.source}:{alert.category}"
        if not force and self._is_duplicate(dedup_key):
            logger.debug(f"Duplicate alert suppressed: {alert.title}")
            return True

        success = await self._dispatch(alert)

        self._recent_alerts[dedup_key] = datetime.utcnow()
        self._cleanup_old_alerts()

        return success

    async def _dispatch(self, alert: Alert) -> bool:
        """Dispatch alert to all enabled backends."""
        if not self.backends:
            logger.warning("No notification backends configured")
            return False

        tasks = []
        enabled_backends = [b for b in self.backends if b.is_enabled]

        if not enabled_backends:
            logger.warning("No enabled notification backends")
            return False

        for backend in enabled_backends:
            tasks.append(self._send_to_backend(backend, alert))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successes
        successes = sum(1 for r in results if r is True)
        failures = len(results) - successes

        if failures > 0:
            logger.warning(f"Notification dispatched: {successes} succeeded, {failures} failed")

        return successes > 0

    async def _send_to_backend(self, backend: NotificationBackend, alert: Alert) -> bool:
        """Send alert to a single backend with error handling."""
        try:
            return await backend.send(alert)
        except Exception as e:
            logger.error(f"Backend {backend.name} failed: {e}")
            return False

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = datetime.utcnow()

        # Reset counter each minute
        if now - self._minute_reset_time > timedelta(minutes=1):
            self._sent_count_this_minute = 0
            self._minute_reset_time = now

        if self._sent_count_this_minute >= self.rate_limit_per_minute:
            return False

        self._sent_count_this_minute += 1
        return True

    def _is_duplicate(self, key: str) -> bool:
        """Check if this alert was recently sent."""
        if key not in self._recent_alerts:
            return False

        last_sent = self._recent_alerts[key]
        return datetime.utcnow() - last_sent < timedelta(seconds=self.dedup_window_seconds)

    def _cleanup_old_alerts(self) -> None:
        """Remove expired entries from dedup cache."""
        cutoff = datetime.utcnow() - timedelta(seconds=self.dedup_window_seconds * 2)
        expired = [k for k, v in self._recent_alerts.items() if v < cutoff]
        for k in expired:
            del self._recent_alerts[k]

    async def test_backends(self) -> dict[str, bool]:
        """Test connectivity to all backends."""
        results = {}
        for backend in self.backends:
            try:
                results[backend.name] = await backend.test_connection()
            except Exception as e:
                logger.error(f"Backend {backend.name} test failed: {e}")
                results[backend.name] = False
        return results

    def get_status(self) -> dict[str, Any]:
        """Get notification service status."""
        return {
            "enabled": self.enabled,
            "min_severity": self.min_severity.value,
            "rate_limit": self.rate_limit_per_minute,
            "dedup_window_seconds": self.dedup_window_seconds,
            "backends": [{"name": b.name, "enabled": b.is_enabled} for b in self.backends],
            "recent_alerts_count": len(self._recent_alerts),
            "sent_this_minute": self._sent_count_this_minute,
        }


# Factory function for common configurations
def create_notification_service(
    webhook_url: str | None = None,
    console_enabled: bool = True,
    file_path: str | None = None,
    min_severity: AlertSeverity = AlertSeverity.WARNING,
) -> NotificationService:
    """
    Create a NotificationService with common backend configuration.

    Args:
        webhook_url: Slack/Discord webhook URL (optional)
        console_enabled: Enable console output
        file_path: Path to alert log file (optional)
        min_severity: Minimum severity to notify

    Returns:
        Configured NotificationService
    """
    from gpt_trader.monitoring.notifications.backends import (
        ConsoleNotificationBackend,
        FileNotificationBackend,
        WebhookNotificationBackend,
    )

    service = NotificationService(min_severity=min_severity)

    if console_enabled:
        service.add_backend(ConsoleNotificationBackend(min_severity=min_severity))

    if webhook_url:
        service.add_backend(
            WebhookNotificationBackend(
                webhook_url=webhook_url,
                min_severity=min_severity,
            )
        )

    if file_path:
        service.add_backend(
            FileNotificationBackend(
                file_path=file_path,
                min_severity=min_severity,
            )
        )

    return service
