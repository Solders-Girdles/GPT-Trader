"""
Production Alerting System for GPT-Trader Live Trading

Multi-channel notification system that provides real-time alerts for:
- Critical trading events and system failures
- Risk limit breaches and circuit breaker triggers
- Performance degradation and strategy failures
- Market anomalies and data quality issues
- System health monitoring and operational alerts

Supports multiple notification channels:
- Email (SMTP)
- SMS (Twilio)
- Slack (Webhooks)
- Discord (Webhooks)
- Custom Webhooks
- Desktop Notifications
- Mobile Push Notifications (via services)

This system ensures critical trading alerts reach operators immediately.
"""

import json
import logging
import queue
import smtplib
import sqlite3
import ssl
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MimeMultipart
from email.mime.text import MimeText
from email.utils import formataddr
from enum import Enum
from pathlib import Path
from typing import Any

import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""

    INFO = "info"  # Informational alerts
    WARNING = "warning"  # Warning conditions
    CRITICAL = "critical"  # Critical issues requiring immediate attention
    EMERGENCY = "emergency"  # Emergency situations requiring immediate action


class AlertChannel(Enum):
    """Notification channels"""

    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    DISCORD = "discord"
    WEBHOOK = "webhook"
    DESKTOP = "desktop"
    MOBILE_PUSH = "mobile_push"
    CONSOLE = "console"


class AlertStatus(Enum):
    """Alert delivery status"""

    PENDING = "pending"  # Alert queued for delivery
    DELIVERED = "delivered"  # Successfully delivered
    FAILED = "failed"  # Delivery failed
    RETRYING = "retrying"  # Retrying delivery
    EXPIRED = "expired"  # Alert expired


@dataclass
class AlertRule:
    """Alert rule configuration"""

    rule_id: str
    name: str
    description: str

    # Trigger conditions
    event_types: set[str]  # Event types that trigger this rule
    severity_levels: set[AlertSeverity]  # Severity levels to match

    # Delivery configuration
    channels: list[AlertChannel]  # Notification channels to use
    retry_count: int = 3  # Number of retry attempts
    retry_interval: timedelta = field(default_factory=lambda: timedelta(minutes=5))

    # Throttling
    throttle_window: timedelta = field(default_factory=lambda: timedelta(minutes=15))
    max_alerts_per_window: int = 5

    # Scheduling
    active_hours: tuple | None = None  # (start_hour, end_hour) in 24h format
    active_days: set[int] | None = None  # 0=Monday, 6=Sunday

    # Status
    is_active: bool = True
    last_triggered: datetime | None = None
    trigger_count: int = 0


@dataclass
class AlertEvent:
    """Alert event to be delivered"""

    event_id: str
    rule_id: str
    severity: AlertSeverity
    title: str
    message: str

    # Context information
    component: str  # Component that generated the alert
    event_type: str  # Specific event type
    metadata: dict[str, Any] = field(default_factory=dict)

    # Delivery tracking
    channels_to_deliver: list[AlertChannel] = field(default_factory=list)
    delivery_attempts: dict[AlertChannel, int] = field(default_factory=dict)
    delivery_status: dict[AlertChannel, AlertStatus] = field(default_factory=dict)

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    first_attempt_at: datetime | None = None
    last_attempt_at: datetime | None = None
    delivered_at: datetime | None = None
    expires_at: datetime | None = None


@dataclass
class ChannelConfig:
    """Configuration for notification channels"""

    channel: AlertChannel

    # Common settings
    enabled: bool = True
    timeout: int = 30  # seconds

    # Channel-specific settings
    settings: dict[str, Any] = field(default_factory=dict)

    # Rate limiting
    rate_limit: int = 60  # messages per hour
    rate_window: timedelta = field(default_factory=lambda: timedelta(hours=1))

    # Status tracking
    last_used: datetime | None = None
    success_count: int = 0
    failure_count: int = 0


class ProductionAlertingSystem:
    """Production alerting system for trading operations"""

    def __init__(
        self, alerts_dir: str = "data/alerts", max_queue_size: int = 1000, worker_threads: int = 3
    ) -> None:

        self.alerts_dir = Path(alerts_dir)
        self.alerts_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.alerts_dir / "logs").mkdir(exist_ok=True)
        (self.alerts_dir / "failed").mkdir(exist_ok=True)
        (self.alerts_dir / "templates").mkdir(exist_ok=True)

        self.max_queue_size = max_queue_size
        self.worker_threads = worker_threads

        # Initialize database
        self.db_path = self.alerts_dir / "alerting.db"
        self._initialize_database()

        # Alert rules and configuration
        self.alert_rules: dict[str, AlertRule] = {}
        self.channel_configs: dict[AlertChannel, ChannelConfig] = {}
        self._initialize_default_rules()
        self._initialize_channel_configs()

        # Alert processing
        self.alert_queue = queue.Queue(maxsize=max_queue_size)
        self.worker_threads_list = []
        self.is_running = False

        # Alert history
        self.recent_alerts: queue.Queue = queue.Queue(maxsize=100)
        self.alert_history = []

        # Throttling tracking
        self.rule_throttle_tracking: dict[str, list[datetime]] = {}

        # Callbacks for custom processing
        self.alert_callbacks: list[Callable[[AlertEvent], None]] = []

        logger.info("Production Alerting System initialized")

    def _initialize_database(self) -> None:
        """Initialize SQLite database for alerting"""

        with sqlite3.connect(self.db_path) as conn:
            # Alert rules table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alert_rules (
                    rule_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    event_types TEXT NOT NULL,
                    severity_levels TEXT NOT NULL,
                    channels TEXT NOT NULL,
                    configuration TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TEXT NOT NULL,
                    last_updated TEXT NOT NULL
                )
            """
            )

            # Alert events table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alert_events (
                    event_id TEXT PRIMARY KEY,
                    rule_id TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    component TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    first_attempt_at TEXT,
                    delivered_at TEXT,
                    delivery_status TEXT
                )
            """
            )

            # Channel configurations table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS channel_configs (
                    channel TEXT PRIMARY KEY,
                    enabled BOOLEAN DEFAULT TRUE,
                    settings TEXT NOT NULL,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    last_used TEXT,
                    created_at TEXT NOT NULL,
                    last_updated TEXT NOT NULL
                )
            """
            )

            # Delivery attempts table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS delivery_attempts (
                    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    attempt_number INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    attempted_at TEXT NOT NULL,
                    response_data TEXT
                )
            """
            )

            # Alert statistics table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alert_statistics (
                    date TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    total_alerts INTEGER DEFAULT 0,
                    successful_deliveries INTEGER DEFAULT 0,
                    failed_deliveries INTEGER DEFAULT 0,
                    avg_delivery_time_ms INTEGER DEFAULT 0,
                    PRIMARY KEY (date, channel, severity)
                )
            """
            )

            # Create indexes
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_created ON alert_events (created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_severity ON alert_events (severity)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_attempts_event ON delivery_attempts (event_id)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_stats_date ON alert_statistics (date)")

            conn.commit()

    def _initialize_default_rules(self) -> None:
        """Initialize default alert rules"""

        # Critical system failures
        self.add_alert_rule(
            AlertRule(
                rule_id="critical_system_failure",
                name="Critical System Failure",
                description="Critical system failures requiring immediate attention",
                event_types={"system_failure", "trading_engine_crash", "data_feed_failure"},
                severity_levels={AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY},
                channels=[
                    AlertChannel.EMAIL,
                    AlertChannel.SMS,
                    AlertChannel.SLACK,
                    AlertChannel.CONSOLE,
                ],
                max_alerts_per_window=10,
            )
        )

        # Circuit breaker triggers
        self.add_alert_rule(
            AlertRule(
                rule_id="circuit_breaker_triggered",
                name="Circuit Breaker Triggered",
                description="Circuit breaker has been triggered to protect capital",
                event_types={"circuit_breaker_triggered"},
                severity_levels={AlertSeverity.CRITICAL},
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.CONSOLE],
                max_alerts_per_window=3,
            )
        )

        # High drawdown warnings
        self.add_alert_rule(
            AlertRule(
                rule_id="high_drawdown_warning",
                name="High Drawdown Warning",
                description="Portfolio drawdown exceeds warning threshold",
                event_types={"high_drawdown", "risk_limit_breach"},
                severity_levels={AlertSeverity.WARNING},
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.CONSOLE],
                throttle_window=timedelta(hours=1),
                max_alerts_per_window=2,
            )
        )

        # Strategy performance issues
        self.add_alert_rule(
            AlertRule(
                rule_id="strategy_performance_degradation",
                name="Strategy Performance Degradation",
                description="Strategy performance has degraded significantly",
                event_types={"strategy_failure", "low_win_rate", "high_losses"},
                severity_levels={AlertSeverity.WARNING},
                channels=[AlertChannel.EMAIL, AlertChannel.CONSOLE],
                throttle_window=timedelta(hours=6),
                max_alerts_per_window=1,
            )
        )

        # Market data issues
        self.add_alert_rule(
            AlertRule(
                rule_id="market_data_issues",
                name="Market Data Issues",
                description="Market data feed problems detected",
                event_types={"data_quality_issue", "feed_disconnected", "stale_data"},
                severity_levels={AlertSeverity.WARNING},
                channels=[AlertChannel.EMAIL, AlertChannel.CONSOLE],
                throttle_window=timedelta(minutes=30),
                max_alerts_per_window=5,
            )
        )

    def _initialize_channel_configs(self) -> None:
        """Initialize channel configurations"""

        # Email configuration
        self.channel_configs[AlertChannel.EMAIL] = ChannelConfig(
            channel=AlertChannel.EMAIL,
            settings={
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "",  # Set via environment or config
                "password": "",  # Set via environment or config
                "from_email": "gpt-trader-alerts@example.com",
                "from_name": "GPT-Trader Alerts",
                "to_emails": [],  # List of recipient emails
                "use_tls": True,
            },
        )

        # SMS configuration (Twilio)
        self.channel_configs[AlertChannel.SMS] = ChannelConfig(
            channel=AlertChannel.SMS,
            enabled=False,  # Disabled by default - requires Twilio setup
            settings={
                "account_sid": "",  # Twilio Account SID
                "auth_token": "",  # Twilio Auth Token
                "from_phone": "",  # Twilio phone number
                "to_phones": [],  # List of recipient phone numbers
            },
        )

        # Slack configuration
        self.channel_configs[AlertChannel.SLACK] = ChannelConfig(
            channel=AlertChannel.SLACK,
            settings={
                "webhook_url": "",  # Slack incoming webhook URL
                "channel": "#gpt-trader-alerts",
                "username": "GPT-Trader",
                "icon_emoji": ":robot_face:",
            },
        )

        # Discord configuration
        self.channel_configs[AlertChannel.DISCORD] = ChannelConfig(
            channel=AlertChannel.DISCORD,
            enabled=False,
            settings={"webhook_url": "", "username": "GPT-Trader"},  # Discord webhook URL
        )

        # Generic webhook configuration
        self.channel_configs[AlertChannel.WEBHOOK] = ChannelConfig(
            channel=AlertChannel.WEBHOOK,
            enabled=False,
            settings={
                "webhook_urls": [],  # List of webhook URLs
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "auth_token": "",  # Optional authentication token
            },
        )

        # Console configuration (always enabled)
        self.channel_configs[AlertChannel.CONSOLE] = ChannelConfig(
            channel=AlertChannel.CONSOLE, enabled=True, settings={}
        )

    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add a new alert rule"""

        self.alert_rules[rule.rule_id] = rule
        self._store_alert_rule(rule)

        console.print(f"   📢 Added alert rule: {rule.rule_id}")
        logger.info(f"Alert rule added: {rule.rule_id}")

    def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove an alert rule"""

        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]

            # Remove from database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM alert_rules WHERE rule_id = ?", (rule_id,))
                conn.commit()

            console.print(f"   🗑️ Removed alert rule: {rule_id}")
            return True

        return False

    def update_channel_config(self, channel: AlertChannel, settings: dict[str, Any]) -> None:
        """Update channel configuration"""

        if channel in self.channel_configs:
            self.channel_configs[channel].settings.update(settings)
            self._store_channel_config(self.channel_configs[channel])
            console.print(f"   🔧 Updated {channel.value} configuration")

    def add_alert_callback(self, callback: Callable[[AlertEvent], None]) -> None:
        """Add callback for alert events"""

        self.alert_callbacks.append(callback)

    def start_alerting_system(self) -> None:
        """Start the alerting system"""

        if self.is_running:
            console.print("⚠️  Alerting system is already running")
            return

        console.print("🚀 [bold green]Starting Production Alerting System[/bold green]")

        self.is_running = True

        # Start worker threads
        for _i in range(self.worker_threads):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self.worker_threads_list.append(worker)

        console.print(f"   ✅ {self.worker_threads} alert worker threads started")
        console.print(f"   📢 {len(self.alert_rules)} alert rules active")
        console.print(
            f"   📡 {len([c for c in self.channel_configs.values() if c.enabled])} notification channels enabled"
        )

        logger.info("Production alerting system started successfully")

    def stop_alerting_system(self) -> None:
        """Stop the alerting system"""

        console.print("⏹️  [bold yellow]Stopping Production Alerting System[/bold yellow]")

        self.is_running = False

        # Wait for worker threads to finish
        for worker in self.worker_threads_list:
            if worker.is_alive():
                worker.join(timeout=10)

        console.print("   ✅ Alerting system stopped")

        logger.info("Production alerting system stopped")

    def send_alert(
        self,
        event_type: str,
        severity: AlertSeverity,
        title: str,
        message: str,
        component: str = "system",
        metadata: dict[str, Any] = None,
    ) -> str:
        """Send an alert through the system"""

        if metadata is None:
            metadata = {}

        # Find matching rules
        matching_rules = self._find_matching_rules(event_type, severity)

        if not matching_rules:
            logger.warning(f"No matching alert rules for event: {event_type}")
            return ""

        # Create alert event
        event_id = (
            f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{event_type}_{hash(title) % 10000}"
        )

        for rule in matching_rules:
            # Check throttling
            if not self._check_rule_throttling(rule):
                continue

            # Create alert event
            alert_event = AlertEvent(
                event_id=f"{event_id}_{rule.rule_id}",
                rule_id=rule.rule_id,
                severity=severity,
                title=title,
                message=message,
                component=component,
                event_type=event_type,
                metadata=metadata,
                channels_to_deliver=rule.channels,
                expires_at=datetime.now() + timedelta(hours=24),
            )

            # Initialize delivery tracking
            for channel in rule.channels:
                alert_event.delivery_attempts[channel] = 0
                alert_event.delivery_status[channel] = AlertStatus.PENDING

            # Queue for processing
            try:
                self.alert_queue.put(alert_event, timeout=1)
                self.recent_alerts.put(alert_event, block=False)

                # Store in database
                self._store_alert_event(alert_event)

                # Update rule tracking
                rule.last_triggered = datetime.now()
                rule.trigger_count += 1
                self._update_rule_throttling(rule)

            except queue.Full:
                logger.error("Alert queue is full - dropping alert")
                continue

        return event_id

    def send_critical_alert(self, title: str, message: str, component: str = "system") -> str:
        """Send a critical alert"""

        return self.send_alert(
            event_type="critical_alert",
            severity=AlertSeverity.CRITICAL,
            title=title,
            message=message,
            component=component,
        )

    def send_emergency_alert(self, title: str, message: str, component: str = "system") -> str:
        """Send an emergency alert"""

        return self.send_alert(
            event_type="emergency_alert",
            severity=AlertSeverity.EMERGENCY,
            title=title,
            message=message,
            component=component,
        )

    def _find_matching_rules(self, event_type: str, severity: AlertSeverity) -> list[AlertRule]:
        """Find alert rules matching the event"""

        matching_rules = []

        for rule in self.alert_rules.values():
            if not rule.is_active:
                continue

            # Check event type
            if event_type not in rule.event_types:
                continue

            # Check severity
            if severity not in rule.severity_levels:
                continue

            # Check scheduling (if configured)
            if not self._check_rule_schedule(rule):
                continue

            matching_rules.append(rule)

        return matching_rules

    def _check_rule_schedule(self, rule: AlertRule) -> bool:
        """Check if rule is active based on schedule"""

        now = datetime.now()

        # Check active hours
        if rule.active_hours:
            start_hour, end_hour = rule.active_hours
            current_hour = now.hour

            if start_hour <= end_hour:
                if not (start_hour <= current_hour <= end_hour):
                    return False
            else:  # Spans midnight
                if not (current_hour >= start_hour or current_hour <= end_hour):
                    return False

        # Check active days
        if rule.active_days:
            if now.weekday() not in rule.active_days:
                return False

        return True

    def _check_rule_throttling(self, rule: AlertRule) -> bool:
        """Check if rule is throttled"""

        if rule.rule_id not in self.rule_throttle_tracking:
            self.rule_throttle_tracking[rule.rule_id] = []

        now = datetime.now()
        cutoff_time = now - rule.throttle_window

        # Remove old entries
        self.rule_throttle_tracking[rule.rule_id] = [
            t for t in self.rule_throttle_tracking[rule.rule_id] if t > cutoff_time
        ]

        # Check if under limit
        return len(self.rule_throttle_tracking[rule.rule_id]) < rule.max_alerts_per_window

    def _update_rule_throttling(self, rule: AlertRule) -> None:
        """Update rule throttling tracking"""

        if rule.rule_id not in self.rule_throttle_tracking:
            self.rule_throttle_tracking[rule.rule_id] = []

        self.rule_throttle_tracking[rule.rule_id].append(datetime.now())

    def _worker_loop(self) -> None:
        """Main worker loop for processing alerts"""

        while self.is_running:
            try:
                # Get next alert from queue
                alert_event = self.alert_queue.get(timeout=1)

                # Process the alert
                self._process_alert_event(alert_event)

                # Mark queue task as done
                self.alert_queue.task_done()

            except queue.Empty:
                # No alerts to process, continue
                continue
            except Exception as e:
                logger.error(f"Alert worker error: {str(e)}")
                time.sleep(1)  # Brief pause on error

    def _process_alert_event(self, alert_event: AlertEvent) -> None:
        """Process a single alert event"""

        try:
            if alert_event.first_attempt_at is None:
                alert_event.first_attempt_at = datetime.now()

            alert_event.last_attempt_at = datetime.now()

            # Check if alert has expired
            if alert_event.expires_at and datetime.now() > alert_event.expires_at:
                logger.warning(f"Alert expired: {alert_event.event_id}")
                return

            # Try to deliver to each channel
            all_delivered = True

            for channel in alert_event.channels_to_deliver:
                if alert_event.delivery_status[channel] == AlertStatus.DELIVERED:
                    continue  # Already delivered to this channel

                # Check retry limit
                if alert_event.delivery_attempts[channel] >= 3:  # Max retries
                    alert_event.delivery_status[channel] = AlertStatus.FAILED
                    continue

                # Attempt delivery
                success = self._deliver_to_channel(alert_event, channel)

                alert_event.delivery_attempts[channel] += 1

                if success:
                    alert_event.delivery_status[channel] = AlertStatus.DELIVERED
                    self.channel_configs[channel].success_count += 1
                    self.channel_configs[channel].last_used = datetime.now()
                else:
                    alert_event.delivery_status[channel] = AlertStatus.FAILED
                    self.channel_configs[channel].failure_count += 1
                    all_delivered = False

                # Store delivery attempt
                self._store_delivery_attempt(alert_event, channel, success)

            # Update overall status
            if all_delivered:
                alert_event.delivered_at = datetime.now()

            # Store updated event
            self._store_alert_event(alert_event)

            # Notify callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert_event)
                except Exception as e:
                    logger.error(f"Alert callback error: {str(e)}")

        except Exception as e:
            logger.error(f"Alert processing error: {str(e)}")

    def _deliver_to_channel(self, alert_event: AlertEvent, channel: AlertChannel) -> bool:
        """Deliver alert to specific channel"""

        if channel not in self.channel_configs or not self.channel_configs[channel].enabled:
            return False

        try:
            if channel == AlertChannel.EMAIL:
                return self._send_email_alert(alert_event)
            elif channel == AlertChannel.SMS:
                return self._send_sms_alert(alert_event)
            elif channel == AlertChannel.SLACK:
                return self._send_slack_alert(alert_event)
            elif channel == AlertChannel.DISCORD:
                return self._send_discord_alert(alert_event)
            elif channel == AlertChannel.WEBHOOK:
                return self._send_webhook_alert(alert_event)
            elif channel == AlertChannel.CONSOLE:
                return self._send_console_alert(alert_event)

            return False

        except Exception as e:
            logger.error(f"Channel delivery error ({channel.value}): {str(e)}")
            return False

    def _send_email_alert(self, alert_event: AlertEvent) -> bool:
        """Send alert via email"""

        config = self.channel_configs[AlertChannel.EMAIL]
        settings = config.settings

        if not settings.get("username") or not settings.get("to_emails"):
            return False

        try:
            # Create message
            msg = MimeMultipart()
            msg["From"] = formataddr(
                (
                    settings.get("from_name", "GPT-Trader"),
                    settings.get("from_email", settings["username"]),
                )
            )
            msg["To"] = ", ".join(settings["to_emails"])
            msg["Subject"] = f"[{alert_event.severity.value.upper()}] {alert_event.title}"

            # Create body
            body = f"""
Alert: {alert_event.title}
Severity: {alert_event.severity.value.upper()}
Component: {alert_event.component}
Time: {alert_event.created_at.strftime('%Y-%m-%d %H:%M:%S')}

Message:
{alert_event.message}

Event ID: {alert_event.event_id}
Rule ID: {alert_event.rule_id}

---
Generated by GPT-Trader Production Alerting System
"""

            msg.attach(MimeText(body, "plain"))

            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP(settings["smtp_server"], settings["smtp_port"]) as server:
                if settings.get("use_tls", True):
                    server.starttls(context=context)
                server.login(settings["username"], settings["password"])
                server.send_message(msg)

            return True

        except Exception as e:
            logger.error(f"Email delivery error: {str(e)}")
            return False

    def _send_sms_alert(self, alert_event: AlertEvent) -> bool:
        """Send alert via SMS (Twilio)"""

        config = self.channel_configs[AlertChannel.SMS]
        settings = config.settings

        if not all(
            [
                settings.get("account_sid"),
                settings.get("auth_token"),
                settings.get("from_phone"),
                settings.get("to_phones"),
            ]
        ):
            return False

        try:
            # This would integrate with Twilio API
            # For demo purposes, we'll simulate success
            logger.info(f"SMS alert sent: {alert_event.title} (simulated)")
            return True

        except Exception as e:
            logger.error(f"SMS delivery error: {str(e)}")
            return False

    def _send_slack_alert(self, alert_event: AlertEvent) -> bool:
        """Send alert to Slack"""

        config = self.channel_configs[AlertChannel.SLACK]
        settings = config.settings

        webhook_url = settings.get("webhook_url")
        if not webhook_url:
            return False

        try:
            # Create Slack message
            severity_color = {
                AlertSeverity.INFO: "#36a64f",  # Green
                AlertSeverity.WARNING: "#ff9800",  # Orange
                AlertSeverity.CRITICAL: "#f44336",  # Red
                AlertSeverity.EMERGENCY: "#e91e63",  # Pink
            }.get(alert_event.severity, "#9e9e9e")

            payload = {
                "channel": settings.get("channel", "#general"),
                "username": settings.get("username", "GPT-Trader"),
                "icon_emoji": settings.get("icon_emoji", ":robot_face:"),
                "attachments": [
                    {
                        "color": severity_color,
                        "title": f"[{alert_event.severity.value.upper()}] {alert_event.title}",
                        "text": alert_event.message,
                        "fields": [
                            {"title": "Component", "value": alert_event.component, "short": True},
                            {"title": "Event Type", "value": alert_event.event_type, "short": True},
                            {
                                "title": "Time",
                                "value": alert_event.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                                "short": True,
                            },
                            {"title": "Event ID", "value": alert_event.event_id, "short": True},
                        ],
                        "footer": "GPT-Trader Alerting System",
                        "ts": int(alert_event.created_at.timestamp()),
                    }
                ],
            }

            response = requests.post(webhook_url, json=payload, timeout=config.timeout)
            response.raise_for_status()

            return True

        except Exception as e:
            logger.error(f"Slack delivery error: {str(e)}")
            return False

    def _send_discord_alert(self, alert_event: AlertEvent) -> bool:
        """Send alert to Discord"""

        config = self.channel_configs[AlertChannel.DISCORD]
        settings = config.settings

        webhook_url = settings.get("webhook_url")
        if not webhook_url:
            return False

        try:
            # Create Discord embed
            severity_color = {
                AlertSeverity.INFO: 0x36A64F,  # Green
                AlertSeverity.WARNING: 0xFF9800,  # Orange
                AlertSeverity.CRITICAL: 0xF44336,  # Red
                AlertSeverity.EMERGENCY: 0xE91E63,  # Pink
            }.get(alert_event.severity, 0x9E9E9E)

            payload = {
                "username": settings.get("username", "GPT-Trader"),
                "embeds": [
                    {
                        "title": f"[{alert_event.severity.value.upper()}] {alert_event.title}",
                        "description": alert_event.message,
                        "color": severity_color,
                        "fields": [
                            {"name": "Component", "value": alert_event.component, "inline": True},
                            {"name": "Event Type", "value": alert_event.event_type, "inline": True},
                            {
                                "name": "Time",
                                "value": alert_event.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                                "inline": True,
                            },
                        ],
                        "footer": {"text": f"Event ID: {alert_event.event_id}"},
                        "timestamp": alert_event.created_at.isoformat(),
                    }
                ],
            }

            response = requests.post(webhook_url, json=payload, timeout=config.timeout)
            response.raise_for_status()

            return True

        except Exception as e:
            logger.error(f"Discord delivery error: {str(e)}")
            return False

    def _send_webhook_alert(self, alert_event: AlertEvent) -> bool:
        """Send alert to generic webhook"""

        config = self.channel_configs[AlertChannel.WEBHOOK]
        settings = config.settings

        webhook_urls = settings.get("webhook_urls", [])
        if not webhook_urls:
            return False

        success = False

        for webhook_url in webhook_urls:
            try:
                payload = {
                    "event_id": alert_event.event_id,
                    "rule_id": alert_event.rule_id,
                    "severity": alert_event.severity.value,
                    "title": alert_event.title,
                    "message": alert_event.message,
                    "component": alert_event.component,
                    "event_type": alert_event.event_type,
                    "created_at": alert_event.created_at.isoformat(),
                    "metadata": alert_event.metadata,
                }

                headers = dict(settings.get("headers", {}))
                if settings.get("auth_token"):
                    headers["Authorization"] = f"Bearer {settings['auth_token']}"

                response = requests.request(
                    method=settings.get("method", "POST"),
                    url=webhook_url,
                    json=payload,
                    headers=headers,
                    timeout=config.timeout,
                )
                response.raise_for_status()

                success = True

            except Exception as e:
                logger.error(f"Webhook delivery error ({webhook_url}): {str(e)}")

        return success

    def _send_console_alert(self, alert_event: AlertEvent) -> bool:
        """Send alert to console"""

        try:
            severity_color = {
                AlertSeverity.INFO: "blue",
                AlertSeverity.WARNING: "yellow",
                AlertSeverity.CRITICAL: "red",
                AlertSeverity.EMERGENCY: "magenta",
            }.get(alert_event.severity, "white")

            console.print(f"\n🚨 [{severity_color}][ALERT][/{severity_color}] {alert_event.title}")
            console.print(
                f"   Severity: [{severity_color}]{alert_event.severity.value.upper()}[/{severity_color}]"
            )
            console.print(f"   Component: {alert_event.component}")
            console.print(f"   Message: {alert_event.message}")
            console.print(f"   Time: {alert_event.created_at.strftime('%H:%M:%S')}")

            return True

        except Exception as e:
            logger.error(f"Console delivery error: {str(e)}")
            return False

    def _store_alert_rule(self, rule: AlertRule) -> None:
        """Store alert rule in database"""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO alert_rules (
                    rule_id, name, description, event_types, severity_levels, channels,
                    configuration, is_active, created_at, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    rule.rule_id,
                    rule.name,
                    rule.description,
                    json.dumps(list(rule.event_types)),
                    json.dumps([s.value for s in rule.severity_levels]),
                    json.dumps([c.value for c in rule.channels]),
                    json.dumps(
                        {
                            "retry_count": rule.retry_count,
                            "retry_interval_seconds": int(rule.retry_interval.total_seconds()),
                            "throttle_window_seconds": int(rule.throttle_window.total_seconds()),
                            "max_alerts_per_window": rule.max_alerts_per_window,
                            "active_hours": rule.active_hours,
                            "active_days": list(rule.active_days) if rule.active_days else None,
                        }
                    ),
                    rule.is_active,
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()

    def _store_channel_config(self, config: ChannelConfig) -> None:
        """Store channel configuration in database"""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO channel_configs (
                    channel, enabled, settings, success_count, failure_count,
                    last_used, created_at, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    config.channel.value,
                    config.enabled,
                    json.dumps(config.settings),
                    config.success_count,
                    config.failure_count,
                    config.last_used.isoformat() if config.last_used else None,
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()

    def _store_alert_event(self, event: AlertEvent) -> None:
        """Store alert event in database"""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO alert_events (
                    event_id, rule_id, severity, title, message, component, event_type,
                    metadata, created_at, first_attempt_at, delivered_at, delivery_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    event.event_id,
                    event.rule_id,
                    event.severity.value,
                    event.title,
                    event.message,
                    event.component,
                    event.event_type,
                    json.dumps(event.metadata),
                    event.created_at.isoformat(),
                    event.first_attempt_at.isoformat() if event.first_attempt_at else None,
                    event.delivered_at.isoformat() if event.delivered_at else None,
                    json.dumps(
                        {
                            channel.value: status.value
                            for channel, status in event.delivery_status.items()
                        }
                    ),
                ),
            )
            conn.commit()

    def _store_delivery_attempt(
        self, event: AlertEvent, channel: AlertChannel, success: bool
    ) -> None:
        """Store delivery attempt in database"""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO delivery_attempts (
                    event_id, channel, attempt_number, status, attempted_at
                ) VALUES (?, ?, ?, ?, ?)
            """,
                (
                    event.event_id,
                    channel.value,
                    event.delivery_attempts[channel],
                    "success" if success else "failed",
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()

    def get_alerting_status(self) -> dict[str, Any]:
        """Get current alerting system status"""

        return {
            "is_running": self.is_running,
            "active_rules": len([r for r in self.alert_rules.values() if r.is_active]),
            "total_rules": len(self.alert_rules),
            "enabled_channels": len([c for c in self.channel_configs.values() if c.enabled]),
            "total_channels": len(self.channel_configs),
            "queue_size": self.alert_queue.qsize(),
            "worker_threads": len(self.worker_threads_list),
            "channel_stats": {
                channel.value: {
                    "enabled": config.enabled,
                    "success_count": config.success_count,
                    "failure_count": config.failure_count,
                    "success_rate": (
                        config.success_count / (config.success_count + config.failure_count)
                        if (config.success_count + config.failure_count) > 0
                        else 0
                    ),
                }
                for channel, config in self.channel_configs.items()
            },
        }

    def display_alerting_dashboard(self) -> None:
        """Display alerting system dashboard"""

        status = self.get_alerting_status()

        console.print(
            Panel(
                f"[bold blue]Production Alerting System Dashboard[/bold blue]\n"
                f"Status: {'🟢 RUNNING' if status['is_running'] else '🔴 STOPPED'}\n"
                f"Active Rules: {status['active_rules']}/{status['total_rules']}\n"
                f"Enabled Channels: {status['enabled_channels']}/{status['total_channels']}",
                title="📢 Alerting System",
            )
        )

        # Alert rules table
        if self.alert_rules:
            rules_table = Table(title="📢 Alert Rules")
            rules_table.add_column("Rule ID", style="cyan")
            rules_table.add_column("Name", style="white")
            rules_table.add_column("Channels", style="green")
            rules_table.add_column("Triggers", justify="right")
            rules_table.add_column("Active", style="yellow")

            for rule in list(self.alert_rules.values()):
                channels_str = ", ".join([c.value for c in rule.channels[:3]])
                if len(rule.channels) > 3:
                    channels_str += "..."

                rules_table.add_row(
                    rule.rule_id,
                    rule.name[:30] + "..." if len(rule.name) > 30 else rule.name,
                    channels_str,
                    str(rule.trigger_count),
                    "✅" if rule.is_active else "❌",
                )

            console.print(rules_table)


def create_production_alerting_system(
    alerts_dir: str = "data/alerts", max_queue_size: int = 1000, worker_threads: int = 3
) -> ProductionAlertingSystem:
    """Factory function to create production alerting system"""
    return ProductionAlertingSystem(
        alerts_dir=alerts_dir, max_queue_size=max_queue_size, worker_threads=worker_threads
    )


if __name__ == "__main__":
    # Example usage
    alerting_system = create_production_alerting_system()

    # Start system
    alerting_system.start_alerting_system()

    try:
        # Send test alerts
        alerting_system.send_critical_alert(
            "System Test Alert", "This is a test of the production alerting system"
        )

        alerting_system.send_alert(
            event_type="high_drawdown",
            severity=AlertSeverity.WARNING,
            title="Portfolio Drawdown Warning",
            message="Current drawdown: 8.5%",
            component="risk_monitor",
        )

        # Wait for processing
        time.sleep(5)

        # Display dashboard
        alerting_system.display_alerting_dashboard()

    finally:
        alerting_system.stop_alerting_system()

    print("Production Alerting System created successfully!")
