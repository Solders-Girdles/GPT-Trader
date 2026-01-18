"""Notification backend implementations."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from gpt_trader.monitoring.alert_types import Alert, AlertSeverity
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="notifications")


# ANSI color codes for console output
COLORS = {
    AlertSeverity.DEBUG: "\033[90m",  # Gray
    AlertSeverity.INFO: "\033[94m",  # Blue
    AlertSeverity.WARNING: "\033[93m",  # Yellow
    AlertSeverity.ERROR: "\033[91m",  # Red
    AlertSeverity.CRITICAL: "\033[95m",  # Magenta (bold)
}
RESET = "\033[0m"
BOLD = "\033[1m"


@dataclass
class ConsoleNotificationBackend:
    """
    Logs alerts to the console with color-coded severity.

    This is the default backend that's always available.
    Useful for development, debugging, and as a fallback.
    """

    enabled: bool = True
    use_colors: bool = True
    min_severity: AlertSeverity = AlertSeverity.WARNING

    @property
    def name(self) -> str:
        return "console"

    @property
    def is_enabled(self) -> bool:
        return self.enabled

    async def send(self, alert: Alert) -> bool:
        """Print alert to console with formatting."""
        if not self.enabled:
            return False

        if alert.severity.numeric_level < self.min_severity.numeric_level:
            return True  # Filtered but not failed

        try:
            formatted = self._format_alert(alert)
            print(formatted)
            return True
        except Exception as e:
            logger.error(f"Console notification failed: {e}")
            return False

    async def test_connection(self) -> bool:
        """Console is always available."""
        return True

    def _format_alert(self, alert: Alert) -> str:
        """Format alert for console display."""
        severity_str = alert.severity.value.upper()

        if self.use_colors:
            color = COLORS.get(alert.severity, "")
            severity_display = f"{color}{BOLD}[{severity_str}]{RESET}"
        else:
            severity_display = f"[{severity_str}]"

        timestamp = alert.created_at.strftime("%Y-%m-%d %H:%M:%S")
        source = f" ({alert.source})" if alert.source else ""

        lines = [
            f"{severity_display} {timestamp}{source}",
            f"  {alert.title}",
            f"  {alert.message}",
        ]

        if alert.context:
            context_str = ", ".join(f"{k}={v}" for k, v in alert.context.items())
            lines.append(f"  Context: {context_str}")

        return "\n".join(lines)


@dataclass
class WebhookNotificationBackend:
    """
    Sends alerts to a webhook endpoint (Slack, Discord, custom).

    Supports Slack-compatible webhook format by default.
    """

    webhook_url: str
    enabled: bool = True
    min_severity: AlertSeverity = AlertSeverity.WARNING
    timeout_seconds: float = 10.0
    bot_name: str = "GPT-Trader"
    include_metadata: bool = True
    custom_headers: dict[str, str] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return "webhook"

    @property
    def is_enabled(self) -> bool:
        return self.enabled and bool(self.webhook_url)

    async def send(self, alert: Alert) -> bool:
        """Send alert to webhook endpoint."""
        if not self.is_enabled:
            return False

        if alert.severity.numeric_level < self.min_severity.numeric_level:
            return True  # Filtered but not failed

        try:
            payload = self._build_payload(alert)
            success = await self._post_webhook(payload)
            return success
        except Exception as e:
            logger.error(f"Webhook notification failed: {e}")
            return False

    async def test_connection(self) -> bool:
        """Test webhook connectivity with a test payload."""
        if not self.webhook_url:
            return False

        try:
            test_alert = Alert(
                severity=AlertSeverity.INFO,
                title="Connection Test",
                message="GPT-Trader notification system test",
                source="notification_service",
            )
            payload = self._build_payload(test_alert)
            return await self._post_webhook(payload)
        except Exception as e:
            logger.error(f"Webhook connection test failed: {e}")
            return False

    def _build_payload(self, alert: Alert) -> dict[str, Any]:
        """Build Slack-compatible webhook payload."""
        severity_emoji = {
            AlertSeverity.DEBUG: ":bug:",
            AlertSeverity.INFO: ":information_source:",
            AlertSeverity.WARNING: ":warning:",
            AlertSeverity.ERROR: ":x:",
            AlertSeverity.CRITICAL: ":rotating_light:",
        }

        emoji = severity_emoji.get(alert.severity, ":bell:")
        timestamp = alert.created_at.strftime("%Y-%m-%d %H:%M:%S UTC")

        # Build Slack blocks for rich formatting
        blocks: list[dict[str, Any]] = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} {alert.title}",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": alert.message,
                },
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Severity:* {alert.severity.value.upper()} | *Time:* {timestamp}",
                    }
                ],
            },
        ]

        # Add source if available
        if alert.source:
            blocks[2]["elements"].append({"type": "mrkdwn", "text": f"*Source:* {alert.source}"})

        # Add context/metadata fields
        if self.include_metadata and (alert.context or alert.metadata):
            fields = []
            all_data = {**alert.context, **alert.metadata}
            for key, value in list(all_data.items())[:10]:  # Limit fields
                fields.append({"type": "mrkdwn", "text": f"*{key}:* {value}"})

            if fields:
                blocks.append({"type": "section", "fields": fields[:10]})  # Slack limit

        return {
            "username": self.bot_name,
            "icon_emoji": ":robot_face:",
            "blocks": blocks,
            # Fallback text for notifications
            "text": f"[{alert.severity.value.upper()}] {alert.title}: {alert.message}",
        }

    async def _post_webhook(self, payload: dict[str, Any]) -> bool:
        """Post payload to webhook URL."""
        import aiohttp

        headers = {
            "Content-Type": "application/json",
            **self.custom_headers,
        }

        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.webhook_url,
                    data=json.dumps(payload),
                    headers=headers,
                ) as response:
                    if response.status == 200:
                        return True
                    else:
                        body = await response.text()
                        logger.warning(f"Webhook returned {response.status}: {body[:200]}")
                        return False
        except TimeoutError:
            logger.error(f"Webhook request timed out after {self.timeout_seconds}s")
            return False
        except Exception as e:
            logger.error(f"Webhook request failed: {e}")
            return False


@dataclass
class FileNotificationBackend:
    """
    Appends alerts to a JSON Lines file for offline review.

    Useful for audit trails and debugging when other backends fail.
    """

    file_path: str
    enabled: bool = True
    min_severity: AlertSeverity = AlertSeverity.INFO

    @property
    def name(self) -> str:
        return "file"

    @property
    def is_enabled(self) -> bool:
        return self.enabled and bool(self.file_path)

    async def send(self, alert: Alert) -> bool:
        """Append alert to file as JSON line."""
        if not self.is_enabled:
            return False

        if alert.severity.numeric_level < self.min_severity.numeric_level:
            return True

        try:
            line = json.dumps(alert.to_dict()) + "\n"
            await asyncio.to_thread(self._append_to_file, line)
            return True
        except Exception as e:
            logger.error(f"File notification failed: {e}")
            return False

    def _append_to_file(self, line: str) -> None:
        with open(self.file_path, "a") as f:
            f.write(line)

    async def test_connection(self) -> bool:
        """Test if file is writable."""
        if not self.file_path:
            return False

        try:
            with open(self.file_path, "a"):
                pass  # Just test if we can open for append
            return True
        except Exception:
            return False
