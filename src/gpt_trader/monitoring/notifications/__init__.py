"""Notification service for alerting on critical events.

This module provides a pluggable notification system that can dispatch
alerts to multiple backends (console, webhook, email, etc.).
"""

from gpt_trader.monitoring.notifications.backends import (
    ConsoleNotificationBackend,
    FileNotificationBackend,
    WebhookNotificationBackend,
)
from gpt_trader.monitoring.notifications.protocol import NotificationBackend
from gpt_trader.monitoring.notifications.service import (
    NotificationService,
    create_notification_service,
)

__all__ = [
    "NotificationService",
    "NotificationBackend",
    "ConsoleNotificationBackend",
    "WebhookNotificationBackend",
    "FileNotificationBackend",
    "create_notification_service",
]
