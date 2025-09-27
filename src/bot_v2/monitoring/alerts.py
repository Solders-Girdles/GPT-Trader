"""
Alert Dispatcher Module

Routes alerts to multiple channels (log, Slack, PagerDuty, email) based on 
severity levels and configuration.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Callable, Any

# Optional imports
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    
try:
    import aiosmtplib
    HAS_AIOSMTPLIB = True
except ImportError:
    HAS_AIOSMTPLIB = False

logger = logging.getLogger(__name__)


class AlertChannelType(Enum):
    """Available alert channel types."""
    LOG = "log"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    EMAIL = "email"
    WEBHOOK = "webhook"


class AlertSeverity(Enum):
    """Alert severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    
    @property
    def numeric_level(self) -> int:
        """Get numeric severity level for comparison."""
        levels = {
            AlertSeverity.DEBUG: 10,
            AlertSeverity.INFO: 20,
            AlertSeverity.WARNING: 30,
            AlertSeverity.ERROR: 40,
            AlertSeverity.CRITICAL: 50
        }
        return levels[self]


@dataclass
class Alert:
    """Alert data structure."""
    timestamp: datetime
    source: str
    severity: AlertSeverity
    title: str
    message: str
    context: Dict[str, Any] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'severity': self.severity.value,
            'title': self.title,
            'message': self.message,
            'context': self.context or {}
        }


class AlertChannel:
    """Base class for alert channels."""
    
    def __init__(self, min_severity: AlertSeverity = AlertSeverity.INFO):
        self.min_severity = min_severity
        
    async def send(self, alert: Alert) -> bool:
        """
        Send alert through this channel.
        
        Returns:
            True if sent successfully
        """
        if alert.severity.numeric_level < self.min_severity.numeric_level:
            return False  # Below threshold
        
        try:
            return await self._send_impl(alert)
        except Exception as e:
            logger.error(f"Failed to send alert via {self.__class__.__name__}: {e}")
            return False
    
    async def _send_impl(self, alert: Alert) -> bool:
        """Implementation-specific send logic. Override in subclasses."""
        raise NotImplementedError


class LogChannel(AlertChannel):
    """Log-based alert channel."""
    
    async def _send_impl(self, alert: Alert) -> bool:
        """Log the alert."""
        log_method = {
            AlertSeverity.DEBUG: logger.debug,
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.ERROR: logger.error,
            AlertSeverity.CRITICAL: logger.critical
        }.get(alert.severity, logger.info)
        
        log_method(
            f"[ALERT] {alert.title}: {alert.message} "
            f"(source: {alert.source}, context: {alert.context})"
        )
        return True


class SlackChannel(AlertChannel):
    """Slack webhook alert channel."""
    
    def __init__(self, webhook_url: str, min_severity: AlertSeverity = AlertSeverity.WARNING):
        super().__init__(min_severity)
        self.webhook_url = webhook_url
        
    async def _send_impl(self, alert: Alert) -> bool:
        """Send alert to Slack."""
        if not HAS_AIOHTTP:
            logger.warning("aiohttp not installed, cannot send Slack alerts")
            return False
            
        # Color mapping
        colors = {
            AlertSeverity.DEBUG: "#808080",
            AlertSeverity.INFO: "#0000FF",
            AlertSeverity.WARNING: "#FFA500",
            AlertSeverity.ERROR: "#FF0000",
            AlertSeverity.CRITICAL: "#8B0000"
        }
        
        # Build Slack message
        payload = {
            "attachments": [{
                "color": colors.get(alert.severity, "#808080"),
                "title": f"🚨 {alert.title}",
                "text": alert.message,
                "fields": [
                    {
                        "title": "Severity",
                        "value": alert.severity.value.upper(),
                        "short": True
                    },
                    {
                        "title": "Source",
                        "value": alert.source,
                        "short": True
                    },
                    {
                        "title": "Time",
                        "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                        "short": True
                    }
                ],
                "footer": "Trading Bot Alert System",
                "ts": int(alert.timestamp.timestamp())
            }]
        }
        
        # Add context fields if present
        if alert.context:
            for key, value in alert.context.items():
                if key in ['symbol', 'pnl', 'position_size', 'error_count']:
                    payload["attachments"][0]["fields"].append({
                        "title": key.replace('_', ' ').title(),
                        "value": str(value),
                        "short": True
                    })
        
        # Send to Slack
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.webhook_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                return response.status == 200


class PagerDutyChannel(AlertChannel):
    """PagerDuty alert channel for critical alerts."""
    
    def __init__(self, api_key: str, min_severity: AlertSeverity = AlertSeverity.ERROR):
        super().__init__(min_severity)
        self.api_key = api_key
        self.api_url = "https://events.pagerduty.com/v2/enqueue"
        
    async def _send_impl(self, alert: Alert) -> bool:
        """Send alert to PagerDuty."""
        if not HAS_AIOHTTP:
            logger.warning("aiohttp not installed, cannot send PagerDuty alerts")
            return False
            
        # Map severity to PagerDuty severity
        pd_severity = {
            AlertSeverity.WARNING: "warning",
            AlertSeverity.ERROR: "error",
            AlertSeverity.CRITICAL: "critical"
        }.get(alert.severity, "info")
        
        # Build PagerDuty event
        payload = {
            "routing_key": self.api_key,
            "event_action": "trigger",
            "payload": {
                "summary": f"{alert.title}: {alert.message}",
                "source": alert.source,
                "severity": pd_severity,
                "timestamp": alert.timestamp.isoformat(),
                "custom_details": alert.context or {}
            }
        }
        
        # Send to PagerDuty
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                return response.status == 202


class EmailChannel(AlertChannel):
    """Email alert channel."""
    
    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        from_email: str,
        to_emails: List[str],
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_tls: bool = True,
        min_severity: AlertSeverity = AlertSeverity.ERROR
    ):
        super().__init__(min_severity)
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.from_email = from_email
        self.to_emails = to_emails
        self.username = username
        self.password = password
        self.use_tls = use_tls
        
    async def _send_impl(self, alert: Alert) -> bool:
        """Send alert via email."""
        if not HAS_AIOSMTPLIB:
            logger.warning("aiosmtplib not installed, cannot send email alerts")
            return False
            
        import aiosmtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        # Build email
        msg = MIMEMultipart()
        msg['From'] = self.from_email
        msg['To'] = ', '.join(self.to_emails)
        msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
        
        # Build body
        body = f"""
Trading Bot Alert

Severity: {alert.severity.value.upper()}
Source: {alert.source}
Time: {alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")}

{alert.message}

"""
        
        if alert.context:
            body += "Context:\n"
            for key, value in alert.context.items():
                body += f"  {key}: {value}\n"
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        try:
            await aiosmtplib.send(
                msg,
                hostname=self.smtp_host,
                port=self.smtp_port,
                username=self.username,
                password=self.password,
                use_tls=self.use_tls
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False


class WebhookChannel(AlertChannel):
    """Generic webhook alert channel."""
    
    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        min_severity: AlertSeverity = AlertSeverity.WARNING
    ):
        super().__init__(min_severity)
        self.url = url
        self.headers = headers or {}
        
    async def _send_impl(self, alert: Alert) -> bool:
        """Send alert to webhook."""
        if not HAS_AIOHTTP:
            logger.warning("aiohttp not installed, cannot send webhook alerts")
            return False
            
        payload = alert.to_dict()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.url,
                json=payload,
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                return response.status in [200, 201, 202, 204]


class AlertDispatcher:
    """Central alert dispatcher managing multiple channels."""
    
    def __init__(self):
        self.channels: Dict[str, AlertChannel] = {}
        self.alert_history: List[Alert] = []
        self.max_history = 1000
        
        # Always add log channel
        self.add_channel('log', LogChannel())
        
    def add_channel(self, name: str, channel: AlertChannel):
        """Add an alert channel."""
        self.channels[name] = channel
        logger.info(f"Added alert channel: {name}")
        
    def remove_channel(self, name: str):
        """Remove an alert channel."""
        if name in self.channels:
            del self.channels[name]
            logger.info(f"Removed alert channel: {name}")
            
    async def dispatch(self, alert: Alert):
        """
        Dispatch alert to all configured channels.
        
        Returns:
            Dict mapping channel names to send results
        """
        # Store in history
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]
        
        # Send to all channels
        results = {}
        tasks = []
        
        for name, channel in self.channels.items():
            # Skip channels below severity threshold without invoking send
            if alert.severity.numeric_level < channel.min_severity.numeric_level:
                continue
            task = asyncio.create_task(channel.send(alert))
            tasks.append((name, task))
        
        # Wait for all sends to complete
        for name, task in tasks:
            try:
                results[name] = await task
            except Exception as e:
                logger.error(f"Error dispatching to {name}: {e}")
                results[name] = False
        
        # Log dispatch results
        successful = [name for name, success in results.items() if success]
        failed = [name for name, success in results.items() if not success]
        
        if successful:
            logger.debug(f"Alert dispatched to: {', '.join(successful)}")
        if failed:
            logger.warning(f"Alert dispatch failed for: {', '.join(failed)}")
        
        return results
    
    def get_recent_alerts(
        self,
        count: int = 10,
        severity: Optional[AlertSeverity] = None,
        source: Optional[str] = None
    ) -> List[Alert]:
        """Get recent alerts with optional filtering."""
        alerts = self.alert_history
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if source:
            alerts = [a for a in alerts if a.source == source]
        
        return alerts[-count:]
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AlertDispatcher':
        """Create dispatcher from configuration dictionary."""
        dispatcher = cls()
        
        # Configure Slack
        slack_webhook = config.get('slack_webhook_url') or os.getenv('SLACK_WEBHOOK_URL')
        if slack_webhook:
            dispatcher.add_channel(
                'slack',
                SlackChannel(
                    webhook_url=slack_webhook,
                    min_severity=AlertSeverity[config.get('slack_min_severity', 'WARNING').upper()]
                )
            )
        
        # Configure PagerDuty
        pagerduty_key = config.get('pagerduty_api_key') or os.getenv('PAGERDUTY_API_KEY')
        if pagerduty_key:
            dispatcher.add_channel(
                'pagerduty',
                PagerDutyChannel(
                    api_key=pagerduty_key,
                    min_severity=AlertSeverity[config.get('pagerduty_min_severity', 'ERROR').upper()]
                )
            )
        
        # Configure Email
        email_config = config.get('email')
        if email_config and email_config.get('enabled'):
            dispatcher.add_channel(
                'email',
                EmailChannel(
                    smtp_host=email_config['smtp_host'],
                    smtp_port=email_config['smtp_port'],
                    from_email=email_config['from_email'],
                    to_emails=email_config['to_emails'],
                    username=email_config.get('username'),
                    password=email_config.get('password'),
                    use_tls=email_config.get('use_tls', True),
                    min_severity=AlertSeverity[email_config.get('min_severity', 'ERROR').upper()]
                )
            )
        
        # Configure generic webhooks
        webhooks = config.get('webhooks', [])
        for i, webhook_config in enumerate(webhooks):
            dispatcher.add_channel(
                f'webhook_{i}',
                WebhookChannel(
                    url=webhook_config['url'],
                    headers=webhook_config.get('headers'),
                    min_severity=AlertSeverity[webhook_config.get('min_severity', 'WARNING').upper()]
                )
            )
        
        return dispatcher


# Convenience functions for quick alert creation

def create_risk_alert(
    title: str,
    message: str,
    severity: AlertSeverity = AlertSeverity.WARNING,
    **context
) -> Alert:
    """Create a risk-related alert."""
    return Alert(
        timestamp=datetime.now(),
        source="risk_manager",
        severity=severity,
        title=title,
        message=message,
        context=context
    )


def create_execution_alert(
    title: str,
    message: str,
    severity: AlertSeverity = AlertSeverity.INFO,
    **context
) -> Alert:
    """Create an execution-related alert."""
    return Alert(
        timestamp=datetime.now(),
        source="execution_engine",
        severity=severity,
        title=title,
        message=message,
        context=context
    )


def create_system_alert(
    title: str,
    message: str,
    severity: AlertSeverity = AlertSeverity.ERROR,
    **context
) -> Alert:
    """Create a system-level alert."""
    return Alert(
        timestamp=datetime.now(),
        source="system",
        severity=severity,
        title=title,
        message=message,
        context=context
    )


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def test_alerts():
        # Create dispatcher
        dispatcher = AlertDispatcher()
        
        # Add test Slack channel (would need real webhook URL)
        if os.getenv('SLACK_WEBHOOK_URL'):
            dispatcher.add_channel(
                'slack',
                SlackChannel(os.getenv('SLACK_WEBHOOK_URL'))
            )
        
        # Create test alerts
        alerts = [
            create_risk_alert(
                "Daily Loss Warning",
                "Daily loss approaching limit: $8.50 of $10.00",
                AlertSeverity.WARNING,
                current_loss=8.50,
                limit=10.00
            ),
            create_execution_alert(
                "Order Filled",
                "BTC-PERP buy order filled at $45,123.50",
                AlertSeverity.INFO,
                symbol="BTC-PERP",
                side="buy",
                price=45123.50,
                size=0.01
            ),
            create_system_alert(
                "Connection Lost",
                "WebSocket connection to Coinbase lost",
                AlertSeverity.ERROR,
                exchange="coinbase",
                retry_count=3
            )
        ]
        
        # Dispatch alerts
        for alert in alerts:
            results = await dispatcher.dispatch(alert)
            print(f"Alert '{alert.title}' dispatched: {results}")
        
        # Get recent alerts
        recent = dispatcher.get_recent_alerts(severity=AlertSeverity.WARNING)
        print(f"\nRecent warnings: {len(recent)}")
        for alert in recent:
            print(f"  - {alert.title}: {alert.message}")
    
    asyncio.run(test_alerts())
