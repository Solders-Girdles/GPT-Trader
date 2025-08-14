"""
Enhanced Alerting System for Phase 5 Production Integration.
Provides comprehensive alerting capabilities for performance monitoring, risk management, and strategy selection.
"""

from __future__ import annotations

import json
import logging
import smtplib
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts."""

    PERFORMANCE = "performance"
    RISK = "risk"
    STRATEGY = "strategy"
    SYSTEM = "system"
    TRADE = "trade"


@dataclass
class Alert:
    """Alert message."""

    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    data: dict[str, Any]
    timestamp: datetime
    acknowledged: bool = False
    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = None


@dataclass
class AlertConfig:
    """Configuration for alerting system."""

    # Email settings
    email_enabled: bool = False
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_recipients: list[str] = None

    # Slack settings
    slack_enabled: bool = False
    slack_webhook_url: str = ""
    slack_channel: str = "#alerts"

    # Discord settings
    discord_enabled: bool = False
    discord_webhook_url: str = ""

    # Webhook settings
    webhook_enabled: bool = False
    webhook_url: str = ""

    # Alert thresholds
    min_severity_for_email: AlertSeverity = AlertSeverity.WARNING
    min_severity_for_slack: AlertSeverity = AlertSeverity.WARNING
    min_severity_for_discord: AlertSeverity = AlertSeverity.ERROR
    min_severity_for_webhook: AlertSeverity = AlertSeverity.ERROR

    # Rate limiting
    alert_cooldown_minutes: int = 30
    max_alerts_per_hour: int = 10

    def __post_init__(self):
        if self.email_recipients is None:
            self.email_recipients = []


class AlertManager:
    """Comprehensive alerting system."""

    def __init__(self, config: AlertConfig) -> None:
        self.config = config
        self.alerts: list[Alert] = []
        self.alert_history: list[Alert] = []
        self.rate_limit_tracker: dict[str, list[datetime]] = {}

        logger.info("Alert manager initialized")

    async def send_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        title: str,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> str:
        """Send an alert through configured channels."""

        # Create alert
        alert_id = (
            f"{alert_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.alerts)}"
        )
        alert = Alert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            data=data or {},
            timestamp=datetime.now(),
        )

        # Check rate limiting
        if not self._check_rate_limit(alert_type, severity):
            logger.warning(f"Rate limit exceeded for {alert_type.value} alerts")
            return alert_id

        # Add to alerts
        self.alerts.append(alert)
        self.alert_history.append(alert)

        # Send through configured channels
        await self._send_alert_channels(alert)

        logger.info(f"Alert sent: {alert_id} - {title}")
        return alert_id

    async def send_performance_alert(
        self,
        strategy_id: str,
        metric: str,
        current_value: float,
        threshold_value: float,
        severity: AlertSeverity = AlertSeverity.WARNING,
    ) -> str:
        """Send a performance-related alert."""

        title = f"Performance Alert: {strategy_id}"
        message = f"Strategy {strategy_id} {metric} ({current_value:.3f}) exceeded threshold ({threshold_value:.3f})"

        data = {
            "strategy_id": strategy_id,
            "metric": metric,
            "current_value": current_value,
            "threshold_value": threshold_value,
        }

        return await self.send_alert(AlertType.PERFORMANCE, severity, title, message, data)

    async def send_risk_alert(
        self,
        risk_type: str,
        current_value: float,
        limit_value: float,
        severity: AlertSeverity = AlertSeverity.WARNING,
    ) -> str:
        """Send a risk-related alert."""

        title = f"Risk Alert: {risk_type}"
        message = (
            f"Risk metric {risk_type} ({current_value:.3f}) exceeded limit ({limit_value:.3f})"
        )

        data = {"risk_type": risk_type, "current_value": current_value, "limit_value": limit_value}

        return await self.send_alert(AlertType.RISK, severity, title, message, data)

    async def send_strategy_alert(
        self,
        strategy_id: str,
        event: str,
        details: str,
        severity: AlertSeverity = AlertSeverity.INFO,
    ) -> str:
        """Send a strategy-related alert."""

        title = f"Strategy Alert: {strategy_id}"
        message = f"Strategy {strategy_id}: {event} - {details}"

        data = {"strategy_id": strategy_id, "event": event, "details": details}

        return await self.send_alert(AlertType.STRATEGY, severity, title, message, data)

    async def send_system_alert(
        self,
        component: str,
        event: str,
        details: str,
        severity: AlertSeverity = AlertSeverity.ERROR,
    ) -> str:
        """Send a system-related alert."""

        title = f"System Alert: {component}"
        message = f"System component {component}: {event} - {details}"

        data = {"component": component, "event": event, "details": details}

        return await self.send_alert(AlertType.SYSTEM, severity, title, message, data)

    async def send_trade_alert(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        severity: AlertSeverity = AlertSeverity.INFO,
    ) -> str:
        """Send a trade-related alert."""

        title = f"Trade Alert: {symbol}"
        message = f"Trade executed: {action} {quantity} {symbol} @ ${price:.2f}"

        data = {"symbol": symbol, "action": action, "quantity": quantity, "price": price}

        return await self.send_alert(AlertType.TRADE, severity, title, message, data)

    async def _send_alert_channels(self, alert: Alert) -> None:
        """Send alert through all configured channels."""

        # Email
        if (
            self.config.email_enabled
            and alert.severity.value >= self.config.min_severity_for_email.value
        ):
            await self._send_email_alert(alert)

        # Slack
        if (
            self.config.slack_enabled
            and alert.severity.value >= self.config.min_severity_for_slack.value
        ):
            await self._send_slack_alert(alert)

        # Discord
        if (
            self.config.discord_enabled
            and alert.severity.value >= self.config.min_severity_for_discord.value
        ):
            await self._send_discord_alert(alert)

        # Webhook
        if (
            self.config.webhook_enabled
            and alert.severity.value >= self.config.min_severity_for_webhook.value
        ):
            await self._send_webhook_alert(alert)

    async def _send_email_alert(self, alert: Alert) -> None:
        """Send alert via email."""
        try:
            if not self.config.email_username or not self.config.email_password:
                logger.warning("Email credentials not configured")
                return

            # Create message
            msg = MIMEMultipart()
            msg["From"] = self.config.email_username
            msg["To"] = ", ".join(self.config.email_recipients)
            msg["Subject"] = f"[{alert.severity.value.upper()}] {alert.title}"

            # Create HTML body
            html_body = self._create_email_html(alert)
            msg.attach(MIMEText(html_body, "html"))

            # Send email
            with smtplib.SMTP(self.config.email_smtp_server, self.config.email_smtp_port) as server:
                server.starttls()
                server.login(self.config.email_username, self.config.email_password)
                server.send_message(msg)

            logger.info(f"Email alert sent: {alert.alert_id}")

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    async def _send_slack_alert(self, alert: Alert) -> None:
        """Send alert via Slack webhook."""
        try:
            if not self.config.slack_webhook_url:
                logger.warning("Slack webhook URL not configured")
                return

            # Create Slack message
            slack_message = self._create_slack_message(alert)

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.slack_webhook_url, json=slack_message
                ) as response:
                    if response.status == 200:
                        logger.info(f"Slack alert sent: {alert.alert_id}")
                    else:
                        logger.error(f"Slack alert failed: {response.status}")

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

    async def _send_discord_alert(self, alert: Alert) -> None:
        """Send alert via Discord webhook."""
        try:
            if not self.config.discord_webhook_url:
                logger.warning("Discord webhook URL not configured")
                return

            # Create Discord message
            discord_message = self._create_discord_message(alert)

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.discord_webhook_url, json=discord_message
                ) as response:
                    if response.status == 204:  # Discord returns 204 on success
                        logger.info(f"Discord alert sent: {alert.alert_id}")
                    else:
                        logger.error(f"Discord alert failed: {response.status}")

        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")

    async def _send_webhook_alert(self, alert: Alert) -> None:
        """Send alert via generic webhook."""
        try:
            if not self.config.webhook_url:
                logger.warning("Webhook URL not configured")
                return

            # Create webhook payload
            webhook_payload = self._create_webhook_payload(alert)

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.webhook_url,
                    json=webhook_payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status in [200, 201, 202]:
                        logger.info(f"Webhook alert sent: {alert.alert_id}")
                    else:
                        logger.error(f"Webhook alert failed: {response.status}")

        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")

    def _create_email_html(self, alert: Alert) -> str:
        """Create HTML email body."""
        severity_colors = {
            AlertSeverity.INFO: "#17a2b8",
            AlertSeverity.WARNING: "#ffc107",
            AlertSeverity.ERROR: "#dc3545",
            AlertSeverity.CRITICAL: "#721c24",
        }

        color = severity_colors.get(alert.severity, "#6c757d")

        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .alert {{ border-left: 4px solid {color}; padding: 15px; margin: 10px 0; background-color: #f8f9fa; }}
                .header {{ font-size: 18px; font-weight: bold; margin-bottom: 10px; }}
                .message {{ margin-bottom: 15px; }}
                .data {{ background-color: #e9ecef; padding: 10px; border-radius: 4px; }}
                .timestamp {{ color: #6c757d; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="alert">
                <div class="header">{alert.title}</div>
                <div class="message">{alert.message}</div>
                <div class="data">
                    <strong>Alert Type:</strong> {alert.alert_type.value}<br>
                    <strong>Severity:</strong> {alert.severity.value}<br>
                    <strong>Data:</strong> {json.dumps(alert.data, indent=2)}
                </div>
                <div class="timestamp">Sent at: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</div>
            </div>
        </body>
        </html>
        """

        return html

    def _create_slack_message(self, alert: Alert) -> dict[str, Any]:
        """Create Slack message format."""
        severity_colors = {
            AlertSeverity.INFO: "#17a2b8",
            AlertSeverity.WARNING: "#ffc107",
            AlertSeverity.ERROR: "#dc3545",
            AlertSeverity.CRITICAL: "#721c24",
        }

        color = severity_colors.get(alert.severity, "#6c757d")

        return {
            "channel": self.config.slack_channel,
            "attachments": [
                {
                    "color": color,
                    "title": alert.title,
                    "text": alert.message,
                    "fields": [
                        {"title": "Alert Type", "value": alert.alert_type.value, "short": True},
                        {"title": "Severity", "value": alert.severity.value, "short": True},
                        {
                            "title": "Data",
                            "value": json.dumps(alert.data, indent=2),
                            "short": False,
                        },
                    ],
                    "footer": f"Sent at {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                }
            ],
        }

    def _create_discord_message(self, alert: Alert) -> dict[str, Any]:
        """Create Discord message format."""
        severity_colors = {
            AlertSeverity.INFO: 0x17A2B8,
            AlertSeverity.WARNING: 0xFFC107,
            AlertSeverity.ERROR: 0xDC3545,
            AlertSeverity.CRITICAL: 0x721C24,
        }

        color = severity_colors.get(alert.severity, 0x6C757D)

        return {
            "embeds": [
                {
                    "title": alert.title,
                    "description": alert.message,
                    "color": color,
                    "fields": [
                        {"name": "Alert Type", "value": alert.alert_type.value, "inline": True},
                        {"name": "Severity", "value": alert.severity.value, "inline": True},
                        {
                            "name": "Data",
                            "value": f"```json\n{json.dumps(alert.data, indent=2)}\n```",
                            "inline": False,
                        },
                    ],
                    "footer": {"text": f"Sent at {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"},
                }
            ]
        }

    def _create_webhook_payload(self, alert: Alert) -> dict[str, Any]:
        """Create generic webhook payload."""
        return {
            "alert_id": alert.alert_id,
            "alert_type": alert.alert_type.value,
            "severity": alert.severity.value,
            "title": alert.title,
            "message": alert.message,
            "data": alert.data,
            "timestamp": alert.timestamp.isoformat(),
        }

    def _check_rate_limit(self, alert_type: AlertType, severity: AlertSeverity) -> bool:
        """Check if alert should be rate limited."""
        key = f"{alert_type.value}_{severity.value}"
        now = datetime.now()

        # Clean old entries
        if key in self.rate_limit_tracker:
            self.rate_limit_tracker[key] = [
                ts for ts in self.rate_limit_tracker[key] if now - ts < timedelta(hours=1)
            ]
        else:
            self.rate_limit_tracker[key] = []

        # Check hourly limit
        if len(self.rate_limit_tracker[key]) >= self.config.max_alerts_per_hour:
            return False

        # Check cooldown
        if self.rate_limit_tracker[key]:
            last_alert = max(self.rate_limit_tracker[key])
            if now - last_alert < timedelta(minutes=self.config.alert_cooldown_minutes):
                return False

        # Add current alert
        self.rate_limit_tracker[key].append(now)
        return True

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now()
                return True
        return False

    def get_active_alerts(self, alert_type: AlertType | None = None) -> list[Alert]:
        """Get active (unacknowledged) alerts."""
        alerts = [alert for alert in self.alerts if not alert.acknowledged]
        if alert_type:
            alerts = [alert for alert in alerts if alert.alert_type == alert_type]
        return alerts

    def get_alert_history(
        self,
        alert_type: AlertType | None = None,
        severity: AlertSeverity | None = None,
        days: int = 7,
    ) -> list[Alert]:
        """Get alert history with filters."""
        cutoff_date = datetime.now() - timedelta(days=days)
        alerts = [alert for alert in self.alert_history if alert.timestamp >= cutoff_date]

        if alert_type:
            alerts = [alert for alert in alerts if alert.alert_type == alert_type]

        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]

        return alerts

    def get_alert_summary(self) -> dict[str, Any]:
        """Get a summary of alert activity."""
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        last_7d = now - timedelta(days=7)

        recent_alerts = [alert for alert in self.alert_history if alert.timestamp >= last_24h]

        return {
            "total_alerts": len(self.alert_history),
            "active_alerts": len(self.get_active_alerts()),
            "alerts_last_24h": len(recent_alerts),
            "alerts_last_7d": len(
                [alert for alert in self.alert_history if alert.timestamp >= last_7d]
            ),
            "alerts_by_type": self._count_alerts_by_type(),
            "alerts_by_severity": self._count_alerts_by_severity(),
        }

    def _count_alerts_by_type(self) -> dict[str, int]:
        """Count alerts by type."""
        counts = {}
        for alert in self.alert_history:
            alert_type = alert.alert_type.value
            counts[alert_type] = counts.get(alert_type, 0) + 1
        return counts

    def _count_alerts_by_severity(self) -> dict[str, int]:
        """Count alerts by severity."""
        counts = {}
        for alert in self.alert_history:
            severity = alert.severity.value
            counts[severity] = counts.get(severity, 0) + 1
        return counts
