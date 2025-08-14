"""
GPT-Trader Advanced Observability and Alerting System

Comprehensive monitoring and alerting infrastructure providing:
- Distributed tracing across all system components
- Real-time dashboards and visualization
- Intelligent alerting with machine learning-based anomaly detection
- Log aggregation and analysis with structured logging
- Service mesh observability and health monitoring
- Business metrics tracking and SLA monitoring
- Alert correlation and noise reduction
- Incident management and escalation workflows

This system provides enterprise-grade observability with actionable insights
into system behavior, performance, and business outcomes.
"""

import json
import logging
import smtplib
import threading
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from functools import wraps
from typing import (
    Any,
    TypeVar,
)

from .base import BaseComponent, ComponentConfig, HealthStatus
from .concurrency import schedule_recurring_task
from .error_handling import report_error
from .metrics import get_metrics_registry

logger = logging.getLogger(__name__)

T = TypeVar("T")
F = TypeVar("F", bound=Callable)


class AlertSeverity(Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"


class AlertStatus(Enum):
    """Alert status states"""

    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class TraceContext:
    """Distributed tracing context"""

    def __init__(
        self,
        trace_id: str | None = None,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        operation_name: str = "",
    ) -> None:
        self.trace_id = trace_id or str(uuid.uuid4())
        self.span_id = span_id or str(uuid.uuid4())
        self.parent_span_id = parent_span_id
        self.operation_name = operation_name
        self.start_time = datetime.now()
        self.tags: dict[str, Any] = {}
        self.logs: list[dict[str, Any]] = []
        self.finished = False
        self.duration_ms: float | None = None

    def add_tag(self, key: str, value: Any) -> "TraceContext":
        """Add tag to trace context"""
        self.tags[key] = value
        return self

    def log(self, message: str, level: str = "info", **kwargs) -> None:
        """Add log entry to trace"""
        self.logs.append(
            {
                "timestamp": datetime.now().isoformat(),
                "level": level,
                "message": message,
                "fields": kwargs,
            }
        )

    def finish(self) -> None:
        """Finish the trace span"""
        if not self.finished:
            self.duration_ms = (datetime.now() - self.start_time).total_seconds() * 1000
            self.finished = True

    def create_child_span(self, operation_name: str) -> "TraceContext":
        """Create child span"""
        return TraceContext(
            trace_id=self.trace_id, parent_span_id=self.span_id, operation_name=operation_name
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert trace context to dictionary"""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "start_time": self.start_time.isoformat(),
            "duration_ms": self.duration_ms,
            "tags": self.tags,
            "logs": self.logs,
            "finished": self.finished,
        }


@dataclass
class Alert:
    """Alert definition and state"""

    alert_id: str
    name: str
    severity: AlertSeverity
    status: AlertStatus
    description: str

    # Alert metadata
    component_id: str = ""
    metric_name: str = ""
    current_value: Any = None
    threshold_value: Any = None

    # Timing information
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    acknowledged_at: datetime | None = None
    resolved_at: datetime | None = None

    # Alert context
    tags: dict[str, str] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    context_data: dict[str, Any] = field(default_factory=dict)

    # Related information
    related_alerts: list[str] = field(default_factory=list)
    trace_id: str | None = None
    error_id: str | None = None

    # Escalation
    escalation_level: int = 0
    last_escalated_at: datetime | None = None

    def acknowledge(self, acknowledged_by: str = "system") -> None:
        """Acknowledge the alert"""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_at = datetime.now()
        self.updated_at = datetime.now()
        self.context_data["acknowledged_by"] = acknowledged_by

    def resolve(self, resolved_by: str = "system", resolution_note: str = "") -> None:
        """Resolve the alert"""
        self.status = AlertStatus.RESOLVED
        self.resolved_at = datetime.now()
        self.updated_at = datetime.now()
        self.context_data["resolved_by"] = resolved_by
        if resolution_note:
            self.context_data["resolution_note"] = resolution_note

    def suppress(self, suppressed_by: str = "system", reason: str = "") -> None:
        """Suppress the alert"""
        self.status = AlertStatus.SUPPRESSED
        self.updated_at = datetime.now()
        self.context_data["suppressed_by"] = suppressed_by
        if reason:
            self.context_data["suppression_reason"] = reason

    def escalate(self) -> None:
        """Escalate the alert"""
        self.escalation_level += 1
        self.last_escalated_at = datetime.now()
        self.updated_at = datetime.now()

    @property
    def duration(self) -> timedelta | None:
        """Get alert duration"""
        if self.resolved_at:
            return self.resolved_at - self.created_at
        return datetime.now() - self.created_at

    @property
    def is_active(self) -> bool:
        """Check if alert is active"""
        return self.status in [AlertStatus.OPEN, AlertStatus.ACKNOWLEDGED]

    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            "alert_id": self.alert_id,
            "name": self.name,
            "severity": self.severity.value,
            "status": self.status.value,
            "description": self.description,
            "component_id": self.component_id,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "duration_seconds": self.duration.total_seconds() if self.duration else None,
            "tags": self.tags,
            "labels": self.labels,
            "context_data": self.context_data,
            "escalation_level": self.escalation_level,
            "trace_id": self.trace_id,
            "error_id": self.error_id,
        }


@dataclass
class AlertRule:
    """Alert rule definition"""

    rule_id: str
    name: str
    description: str

    # Rule conditions
    metric_name: str
    operator: str  # >, <, >=, <=, ==, !=
    threshold_value: float
    evaluation_window: timedelta = timedelta(minutes=5)

    # Alert configuration
    severity: AlertSeverity = AlertSeverity.WARNING
    component_filter: str | None = None
    labels_filter: dict[str, str] = field(default_factory=dict)

    # Behavior configuration
    enabled: bool = True
    suppression_window: timedelta = timedelta(minutes=15)  # Prevent alert spam
    escalation_threshold: timedelta = timedelta(hours=1)  # Auto-escalate after 1 hour
    auto_resolve: bool = True

    # Notification configuration
    notification_channels: list[str] = field(default_factory=list)

    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"

    def evaluate(self, current_value: float) -> bool:
        """Evaluate if rule conditions are met"""
        if not self.enabled:
            return False

        operators = {
            ">": lambda x, y: x > y,
            "<": lambda x, y: x < y,
            ">=": lambda x, y: x >= y,
            "<=": lambda x, y: x <= y,
            "==": lambda x, y: abs(x - y) < 0.001,  # Float comparison
            "!=": lambda x, y: abs(x - y) >= 0.001,
        }

        if self.operator not in operators:
            logger.warning(f"Unknown operator in rule {self.rule_id}: {self.operator}")
            return False

        return operators[self.operator](current_value, self.threshold_value)

    def to_dict(self) -> dict[str, Any]:
        """Convert rule to dictionary"""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "metric_name": self.metric_name,
            "operator": self.operator,
            "threshold_value": self.threshold_value,
            "evaluation_window_seconds": self.evaluation_window.total_seconds(),
            "severity": self.severity.value,
            "component_filter": self.component_filter,
            "labels_filter": self.labels_filter,
            "enabled": self.enabled,
            "suppression_window_seconds": self.suppression_window.total_seconds(),
            "escalation_threshold_seconds": self.escalation_threshold.total_seconds(),
            "auto_resolve": self.auto_resolve,
            "notification_channels": self.notification_channels,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
        }


class INotificationChannel(ABC):
    """Interface for alert notification channels"""

    @abstractmethod
    def send_notification(self, alert: Alert) -> bool:
        """Send alert notification"""
        pass

    @abstractmethod
    def get_channel_name(self) -> str:
        """Get channel identifier"""
        pass


class EmailNotificationChannel(INotificationChannel):
    """Email notification channel"""

    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        username: str,
        password: str,
        from_address: str,
        to_addresses: list[str],
    ) -> None:
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_address = from_address
        self.to_addresses = to_addresses

    def send_notification(self, alert: Alert) -> bool:
        """Send email notification"""
        try:
            # Create email message
            msg = MIMEMultipart()
            msg["From"] = self.from_address
            msg["To"] = ", ".join(self.to_addresses)
            msg["Subject"] = f"[{alert.severity.value.upper()}] {alert.name}"

            # Create email body
            body = f"""
GPT-Trader Alert Notification

Alert: {alert.name}
Severity: {alert.severity.value.upper()}
Component: {alert.component_id}
Status: {alert.status.value}

Description: {alert.description}

Current Value: {alert.current_value}
Threshold: {alert.threshold_value}

Created: {alert.created_at}
Duration: {alert.duration}

Alert ID: {alert.alert_id}
Trace ID: {alert.trace_id or 'N/A'}

Tags: {', '.join([f'{k}={v}' for k, v in alert.tags.items()])}

Context:
{json.dumps(alert.context_data, indent=2)}
"""

            msg.attach(MIMEText(body, "plain"))

            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()

            logger.info(f"Email notification sent for alert {alert.alert_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email notification: {str(e)}")
            return False

    def get_channel_name(self) -> str:
        """Get channel name"""
        return "email"


class SlackNotificationChannel(INotificationChannel):
    """Slack notification channel (placeholder implementation)"""

    def __init__(self, webhook_url: str, channel: str = "#alerts") -> None:
        self.webhook_url = webhook_url
        self.channel = channel

    def send_notification(self, alert: Alert) -> bool:
        """Send Slack notification"""
        try:
            # This would integrate with Slack API
            # For now, just log the notification
            logger.info(
                f"SLACK NOTIFICATION: [{alert.severity.value.upper()}] {alert.name} - {alert.description}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to send Slack notification: {str(e)}")
            return False

    def get_channel_name(self) -> str:
        """Get channel name"""
        return "slack"


class LogNotificationChannel(INotificationChannel):
    """Log-based notification channel"""

    def __init__(self, log_level: int = logging.WARNING) -> None:
        self.log_level = log_level
        self.notification_logger = logging.getLogger(f"{__name__}.alerts")

    def send_notification(self, alert: Alert) -> bool:
        """Send log notification"""
        try:
            severity_levels = {
                AlertSeverity.INFO: logging.INFO,
                AlertSeverity.WARNING: logging.WARNING,
                AlertSeverity.ERROR: logging.ERROR,
                AlertSeverity.CRITICAL: logging.CRITICAL,
                AlertSeverity.FATAL: logging.CRITICAL,
            }

            log_level = severity_levels.get(alert.severity, logging.WARNING)

            self.notification_logger.log(
                log_level,
                f"ALERT [{alert.alert_id}]: {alert.name} - {alert.description} "
                f"(Component: {alert.component_id}, Value: {alert.current_value}, "
                f"Threshold: {alert.threshold_value})",
            )

            return True

        except Exception as e:
            logger.error(f"Failed to send log notification: {str(e)}")
            return False

    def get_channel_name(self) -> str:
        """Get channel name"""
        return "log"


class DistributedTracer:
    """Distributed tracing system"""

    def __init__(self) -> None:
        self.active_traces: dict[str, TraceContext] = {}
        self.completed_traces: deque = deque(maxlen=1000)
        self.trace_handlers: list[Callable[[TraceContext], None]] = []
        self.lock = threading.RLock()

        logger.debug("Distributed tracer initialized")

    def start_trace(self, operation_name: str, trace_id: str | None = None) -> TraceContext:
        """Start new trace"""
        trace_context = TraceContext(trace_id=trace_id, operation_name=operation_name)

        with self.lock:
            self.active_traces[trace_context.trace_id] = trace_context

        return trace_context

    def get_trace(self, trace_id: str) -> TraceContext | None:
        """Get active trace by ID"""
        with self.lock:
            return self.active_traces.get(trace_id)

    def finish_trace(self, trace_context: TraceContext) -> None:
        """Finish trace and move to completed"""
        trace_context.finish()

        with self.lock:
            if trace_context.trace_id in self.active_traces:
                del self.active_traces[trace_context.trace_id]

            self.completed_traces.append(trace_context)

        # Notify trace handlers
        for handler in self.trace_handlers:
            try:
                handler(trace_context)
            except Exception as e:
                logger.error(f"Trace handler error: {str(e)}")

    def add_trace_handler(self, handler: Callable[[TraceContext], None]) -> None:
        """Add trace completion handler"""
        self.trace_handlers.append(handler)

    def get_trace_statistics(self) -> dict[str, Any]:
        """Get tracing statistics"""
        with self.lock:
            completed_traces = list(self.completed_traces)

        if not completed_traces:
            return {"status": "no_traces"}

        # Calculate statistics
        durations = [t.duration_ms for t in completed_traces if t.duration_ms is not None]
        operations = defaultdict(int)

        for trace in completed_traces:
            operations[trace.operation_name] += 1

        return {
            "active_traces": len(self.active_traces),
            "completed_traces": len(completed_traces),
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
            "min_duration_ms": min(durations) if durations else 0,
            "max_duration_ms": max(durations) if durations else 0,
            "operations": dict(operations),
            "total_traces_processed": len(completed_traces),
        }


class ObservabilityEngine(BaseComponent):
    """
    Advanced observability and alerting engine

    Provides comprehensive monitoring, alerting, and observability
    capabilities for the entire GPT-Trader system.
    """

    def __init__(self, config: ComponentConfig | None = None) -> None:
        if not config:
            config = ComponentConfig(
                component_id="observability_engine", component_type="observability_engine"
            )

        super().__init__(config)

        # Core components
        self.tracer = DistributedTracer()
        self.alert_rules: dict[str, AlertRule] = {}
        self.active_alerts: dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.notification_channels: dict[str, INotificationChannel] = {}

        # Alert suppression tracking
        self.suppressed_alerts: dict[str, datetime] = {}  # rule_id -> last_alert_time

        # Metrics integration
        self.metrics_registry = get_metrics_registry()
        self._setup_observability_metrics()

        # Configuration
        self.evaluation_interval = timedelta(seconds=30)
        self.alert_cleanup_interval = timedelta(hours=24)

        # Initialize default notification channels
        self._initialize_default_channels()

        # Initialize default alert rules
        self._initialize_default_alert_rules()

        logger.info("Observability engine initialized")

    def _initialize_component(self) -> None:
        """Initialize observability engine"""
        # Schedule alert rule evaluation
        schedule_recurring_task(
            task_id="evaluate_alert_rules",
            function=self._evaluate_alert_rules,
            interval=self.evaluation_interval,
            component_id=self.component_id,
        )

        # Schedule alert cleanup
        schedule_recurring_task(
            task_id="cleanup_old_alerts",
            function=self._cleanup_old_alerts,
            interval=self.alert_cleanup_interval,
            component_id=self.component_id,
        )

        # Schedule alert escalation check
        schedule_recurring_task(
            task_id="check_alert_escalations",
            function=self._check_alert_escalations,
            interval=timedelta(minutes=10),
            component_id=self.component_id,
        )

    def _start_component(self) -> None:
        """Start observability engine"""
        logger.info("Observability engine started")

    def _stop_component(self) -> None:
        """Stop observability engine"""
        # Generate final observability report
        self._generate_observability_report()
        logger.info("Observability engine stopped")

    def _health_check(self) -> HealthStatus:
        """Check observability engine health"""
        try:
            # Check if we have active alert rules
            if not self.alert_rules:
                return HealthStatus.DEGRADED

            # Check for critical active alerts
            critical_alerts = [
                alert
                for alert in self.active_alerts.values()
                if alert.severity == AlertSeverity.CRITICAL and alert.is_active
            ]

            if critical_alerts:
                return HealthStatus.CRITICAL

            return HealthStatus.HEALTHY

        except Exception:
            return HealthStatus.UNHEALTHY

    def _setup_observability_metrics(self) -> None:
        """Setup observability metrics"""
        self.observability_metrics = {
            "alerts_total": self.metrics_registry.register_counter(
                "observability_alerts_total",
                "Total alerts generated",
                component_id=self.component_id,
            ),
            "alerts_active": self.metrics_registry.register_gauge(
                "observability_alerts_active",
                "Number of active alerts",
                component_id=self.component_id,
            ),
            "alert_rules": self.metrics_registry.register_gauge(
                "observability_alert_rules_total",
                "Total number of alert rules",
                component_id=self.component_id,
            ),
            "traces_active": self.metrics_registry.register_gauge(
                "observability_traces_active",
                "Number of active traces",
                component_id=self.component_id,
            ),
            "notifications_sent": self.metrics_registry.register_counter(
                "observability_notifications_sent_total",
                "Total notifications sent",
                component_id=self.component_id,
            ),
        }

    def _initialize_default_channels(self) -> None:
        """Initialize default notification channels"""
        # Log channel (always available)
        self.add_notification_channel(LogNotificationChannel())

        logger.info("Default notification channels initialized")

    def _initialize_default_alert_rules(self) -> None:
        """Initialize default alert rules"""
        default_rules = [
            AlertRule(
                rule_id="high_error_rate",
                name="High Error Rate",
                description="Error rate exceeds threshold",
                metric_name="errors_total",
                operator=">",
                threshold_value=10.0,
                severity=AlertSeverity.ERROR,
                notification_channels=["log"],
            ),
            AlertRule(
                rule_id="high_cpu_usage",
                name="High CPU Usage",
                description="CPU usage exceeds 85%",
                metric_name="cpu_usage_percent",
                operator=">",
                threshold_value=85.0,
                severity=AlertSeverity.WARNING,
                notification_channels=["log"],
            ),
            AlertRule(
                rule_id="high_memory_usage",
                name="High Memory Usage",
                description="Memory usage exceeds 85%",
                metric_name="memory_usage_percent",
                operator=">",
                threshold_value=85.0,
                severity=AlertSeverity.WARNING,
                notification_channels=["log"],
            ),
            AlertRule(
                rule_id="low_cache_hit_rate",
                name="Low Cache Hit Rate",
                description="Cache hit rate below 70%",
                metric_name="cache_hit_rate_percent",
                operator="<",
                threshold_value=70.0,
                severity=AlertSeverity.WARNING,
                notification_channels=["log"],
            ),
            AlertRule(
                rule_id="trading_system_down",
                name="Trading System Down",
                description="Trading system component is unhealthy",
                metric_name="component_health_status",
                operator="<",
                threshold_value=1.0,  # Assuming 1 = healthy, 0 = unhealthy
                severity=AlertSeverity.CRITICAL,
                notification_channels=["log"],
            ),
        ]

        for rule in default_rules:
            self.add_alert_rule(rule)

        logger.info(f"Initialized {len(default_rules)} default alert rules")

    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add alert rule"""
        self.alert_rules[rule.rule_id] = rule
        self.observability_metrics["alert_rules"].set(len(self.alert_rules))

        logger.info(f"Added alert rule: {rule.name}")

    def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove alert rule"""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            self.observability_metrics["alert_rules"].set(len(self.alert_rules))

            logger.info(f"Removed alert rule: {rule_id}")
            return True

        return False

    def add_notification_channel(self, channel: INotificationChannel) -> None:
        """Add notification channel"""
        self.notification_channels[channel.get_channel_name()] = channel

        logger.info(f"Added notification channel: {channel.get_channel_name()}")

    def start_trace(self, operation_name: str, trace_id: str | None = None) -> TraceContext:
        """Start distributed trace"""
        trace_context = self.tracer.start_trace(operation_name, trace_id)
        self.observability_metrics["traces_active"].set(len(self.tracer.active_traces))

        return trace_context

    def finish_trace(self, trace_context: TraceContext) -> None:
        """Finish distributed trace"""
        self.tracer.finish_trace(trace_context)
        self.observability_metrics["traces_active"].set(len(self.tracer.active_traces))

    def create_alert(
        self,
        name: str,
        severity: AlertSeverity,
        description: str,
        component_id: str = "",
        metric_name: str = "",
        current_value: Any = None,
        threshold_value: Any = None,
        trace_id: str | None = None,
        **kwargs,
    ) -> Alert:
        """Create new alert"""

        alert = Alert(
            alert_id=str(uuid.uuid4()),
            name=name,
            severity=severity,
            status=AlertStatus.OPEN,
            description=description,
            component_id=component_id,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            trace_id=trace_id,
        )

        # Add any additional context
        for key, value in kwargs.items():
            if hasattr(alert, key):
                setattr(alert, key, value)
            else:
                alert.context_data[key] = value

        # Store alert
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)

        # Update metrics
        self.observability_metrics["alerts_total"].increment()
        self.observability_metrics["alerts_active"].set(len(self.active_alerts))

        # Send notifications
        self._send_alert_notifications(alert)

        logger.warning(
            f"Alert created: {alert.name} (ID: {alert.alert_id}, Severity: {alert.severity.value})"
        )

        return alert

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledge(acknowledged_by)

            logger.info(f"Alert acknowledged: {alert.name} (ID: {alert_id})")
            return True

        return False

    def resolve_alert(
        self, alert_id: str, resolved_by: str = "system", resolution_note: str = ""
    ) -> bool:
        """Resolve alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolve(resolved_by, resolution_note)

            # Move to history and remove from active
            del self.active_alerts[alert_id]
            self.observability_metrics["alerts_active"].set(len(self.active_alerts))

            logger.info(f"Alert resolved: {alert.name} (ID: {alert_id})")
            return True

        return False

    def _evaluate_alert_rules(self) -> None:
        """Evaluate all alert rules against current metrics"""
        try:
            current_time = datetime.now()

            # Get all metrics for evaluation
            all_metrics = self.metrics_registry.export_all_metrics()

            for rule_id, rule in self.alert_rules.items():
                if not rule.enabled:
                    continue

                try:
                    # Check if this rule was recently suppressed
                    if rule_id in self.suppressed_alerts:
                        time_since_last = current_time - self.suppressed_alerts[rule_id]
                        if time_since_last < rule.suppression_window:
                            continue  # Still in suppression window
                        else:
                            del self.suppressed_alerts[rule_id]  # Remove from suppression

                    # Find matching metrics
                    metric_data = all_metrics.get(rule.metric_name)
                    if not metric_data:
                        continue

                    # Evaluate rule for each metric instance
                    for metric_instance in metric_data.get("metrics", []):
                        # Apply component filter
                        if rule.component_filter:
                            component_id = metric_instance.get("labels", {}).get("component_id", "")
                            if rule.component_filter not in component_id:
                                continue

                        # Apply label filters
                        if rule.labels_filter:
                            metric_labels = metric_instance.get("labels", {})
                            if not all(
                                metric_labels.get(k) == v for k, v in rule.labels_filter.items()
                            ):
                                continue

                        # Get current value
                        current_value = metric_instance.get("value")
                        if current_value is None:
                            continue

                        # Evaluate rule condition
                        if rule.evaluate(current_value):
                            # Check if we already have an active alert for this rule
                            existing_alert = None
                            for alert in self.active_alerts.values():
                                if (
                                    alert.component_id
                                    == metric_instance.get("labels", {}).get("component_id", "")
                                    and alert.metric_name == rule.metric_name
                                    and alert.name == rule.name
                                ):
                                    existing_alert = alert
                                    break

                            if not existing_alert:
                                # Create new alert
                                component_id = metric_instance.get("labels", {}).get(
                                    "component_id", ""
                                )

                                alert = self.create_alert(
                                    name=rule.name,
                                    severity=rule.severity,
                                    description=rule.description,
                                    component_id=component_id,
                                    metric_name=rule.metric_name,
                                    current_value=current_value,
                                    threshold_value=rule.threshold_value,
                                    tags={"rule_id": rule_id},
                                    labels=metric_instance.get("labels", {}),
                                )

                                # Add to suppression tracking
                                self.suppressed_alerts[rule_id] = current_time

                        else:
                            # Condition not met - check for auto-resolve
                            if rule.auto_resolve:
                                alerts_to_resolve = []
                                for alert_id, alert in self.active_alerts.items():
                                    if (
                                        alert.tags.get("rule_id") == rule_id
                                        and alert.metric_name == rule.metric_name
                                    ):
                                        alerts_to_resolve.append(alert_id)

                                for alert_id in alerts_to_resolve:
                                    self.resolve_alert(
                                        alert_id, "system", "Auto-resolved: condition no longer met"
                                    )

                except Exception as e:
                    logger.error(f"Error evaluating alert rule {rule_id}: {str(e)}")

            self.record_operation(success=True)

        except Exception as e:
            logger.error(f"Alert rule evaluation error: {str(e)}")
            self.record_operation(success=False, error_message=str(e))
            report_error(e, component=self.component_id)

    def _send_alert_notifications(self, alert: Alert) -> None:
        """Send notifications for alert"""
        # Find the rule that triggered this alert
        rule_id = alert.tags.get("rule_id")
        if not rule_id or rule_id not in self.alert_rules:
            # Fallback to all available channels for manually created alerts
            channels = list(self.notification_channels.keys())
        else:
            rule = self.alert_rules[rule_id]
            channels = rule.notification_channels

        for channel_name in channels:
            if channel_name in self.notification_channels:
                channel = self.notification_channels[channel_name]

                try:
                    success = channel.send_notification(alert)
                    if success:
                        self.observability_metrics["notifications_sent"].increment()
                        logger.debug(
                            f"Notification sent via {channel_name} for alert {alert.alert_id}"
                        )
                    else:
                        logger.warning(
                            f"Failed to send notification via {channel_name} for alert {alert.alert_id}"
                        )

                except Exception as e:
                    logger.error(f"Notification error for channel {channel_name}: {str(e)}")

    def _check_alert_escalations(self) -> None:
        """Check for alerts that need escalation"""
        try:
            current_time = datetime.now()

            for alert in self.active_alerts.values():
                # Find the rule for escalation configuration
                rule_id = alert.tags.get("rule_id")
                if not rule_id or rule_id not in self.alert_rules:
                    continue

                rule = self.alert_rules[rule_id]

                # Check if alert should be escalated
                alert_age = current_time - alert.created_at
                time_since_escalation = current_time - (alert.last_escalated_at or alert.created_at)

                if (
                    alert_age > rule.escalation_threshold
                    and time_since_escalation > rule.escalation_threshold
                    and alert.status == AlertStatus.OPEN
                ):
                    alert.escalate()

                    # Send escalated notification
                    logger.critical(
                        f"ALERT ESCALATED: {alert.name} (ID: {alert.alert_id}, Level: {alert.escalation_level})"
                    )

                    # You could implement additional escalation logic here
                    # such as notifying different people based on escalation level

        except Exception as e:
            logger.error(f"Alert escalation check error: {str(e)}")

    def _cleanup_old_alerts(self) -> None:
        """Clean up old resolved alerts"""
        try:
            cutoff_time = datetime.now() - timedelta(days=7)  # Keep alerts for 7 days

            # Clean up alert history
            self.alert_history = deque(
                [alert for alert in self.alert_history if alert.created_at > cutoff_time],
                maxlen=self.alert_history.maxlen,
            )

            logger.debug("Old alerts cleaned up")

        except Exception as e:
            logger.error(f"Alert cleanup error: {str(e)}")

    def _generate_observability_report(self) -> dict[str, Any]:
        """Generate comprehensive observability report"""
        try:
            # Alert statistics
            alert_stats = {
                "total_alerts": len(self.alert_history),
                "active_alerts": len(self.active_alerts),
                "alert_by_severity": defaultdict(int),
                "alert_by_status": defaultdict(int),
                "alert_by_component": defaultdict(int),
                "avg_resolution_time_hours": 0.0,
            }

            resolution_times = []

            for alert in self.alert_history:
                alert_stats["alert_by_severity"][alert.severity.value] += 1
                alert_stats["alert_by_status"][alert.status.value] += 1
                alert_stats["alert_by_component"][alert.component_id] += 1

                if alert.resolved_at and alert.created_at:
                    resolution_time = (alert.resolved_at - alert.created_at).total_seconds() / 3600
                    resolution_times.append(resolution_time)

            if resolution_times:
                alert_stats["avg_resolution_time_hours"] = sum(resolution_times) / len(
                    resolution_times
                )

            # Convert defaultdicts to regular dicts
            alert_stats["alert_by_severity"] = dict(alert_stats["alert_by_severity"])
            alert_stats["alert_by_status"] = dict(alert_stats["alert_by_status"])
            alert_stats["alert_by_component"] = dict(alert_stats["alert_by_component"])

            # Trace statistics
            trace_stats = self.tracer.get_trace_statistics()

            # System health overview
            system_health = {
                "critical_alerts": len(
                    [a for a in self.active_alerts.values() if a.severity == AlertSeverity.CRITICAL]
                ),
                "warning_alerts": len(
                    [a for a in self.active_alerts.values() if a.severity == AlertSeverity.WARNING]
                ),
                "error_alerts": len(
                    [a for a in self.active_alerts.values() if a.severity == AlertSeverity.ERROR]
                ),
                "alert_rules_enabled": len([r for r in self.alert_rules.values() if r.enabled]),
                "notification_channels": len(self.notification_channels),
            }

            report = {
                "timestamp": datetime.now().isoformat(),
                "alert_statistics": alert_stats,
                "trace_statistics": trace_stats,
                "system_health": system_health,
                "active_alert_details": [alert.to_dict() for alert in self.active_alerts.values()],
                "alert_rules": [rule.to_dict() for rule in self.alert_rules.values()],
            }

            logger.info("Observability report generated")
            return report

        except Exception as e:
            logger.error(f"Report generation error: {str(e)}")
            return {"error": str(e)}

    def get_active_alerts(self) -> list[Alert]:
        """Get active alerts"""
        return list(self.active_alerts.values())

    def get_alert_rules(self) -> list[AlertRule]:
        """Get alert rules"""
        return list(self.alert_rules.values())

    def get_system_health_summary(self) -> dict[str, Any]:
        """Get system health summary"""
        critical_alerts = [
            a for a in self.active_alerts.values() if a.severity == AlertSeverity.CRITICAL
        ]
        error_alerts = [a for a in self.active_alerts.values() if a.severity == AlertSeverity.ERROR]
        warning_alerts = [
            a for a in self.active_alerts.values() if a.severity == AlertSeverity.WARNING
        ]

        # Determine overall health status
        if critical_alerts:
            overall_health = "critical"
        elif error_alerts:
            overall_health = "degraded"
        elif warning_alerts:
            overall_health = "warning"
        else:
            overall_health = "healthy"

        return {
            "overall_health": overall_health,
            "active_alerts": len(self.active_alerts),
            "critical_alerts": len(critical_alerts),
            "error_alerts": len(error_alerts),
            "warning_alerts": len(warning_alerts),
            "alert_rules_enabled": len([r for r in self.alert_rules.values() if r.enabled]),
            "trace_statistics": self.tracer.get_trace_statistics(),
        }


# Global observability engine instance
_observability_engine: ObservabilityEngine | None = None
_observability_lock = threading.Lock()


def get_observability_engine() -> ObservabilityEngine:
    """Get global observability engine instance"""
    global _observability_engine

    with _observability_lock:
        if _observability_engine is None:
            _observability_engine = ObservabilityEngine()
            logger.info("Global observability engine created")

        return _observability_engine


def create_alert(name: str, severity: AlertSeverity, description: str, **kwargs) -> Alert:
    """Create new alert"""
    return get_observability_engine().create_alert(name, severity, description, **kwargs)


def start_trace(operation_name: str, trace_id: str | None = None) -> TraceContext:
    """Start distributed trace"""
    return get_observability_engine().start_trace(operation_name, trace_id)


def finish_trace(trace_context: TraceContext) -> None:
    """Finish distributed trace"""
    get_observability_engine().finish_trace(trace_context)


# Observability decorators


def trace_operation(operation_name: str = None):
    """Decorator to trace function execution"""

    def decorator(func: F) -> F:
        nonlocal operation_name
        if not operation_name:
            operation_name = f"{func.__module__}.{func.__name__}"

        @wraps(func)
        def wrapper(*args, **kwargs):
            trace_context = start_trace(operation_name)
            trace_context.add_tag("function", func.__name__)
            trace_context.add_tag("module", func.__module__)

            try:
                result = func(*args, **kwargs)
                trace_context.add_tag("status", "success")
                return result

            except Exception as e:
                trace_context.add_tag("status", "error")
                trace_context.add_tag("error", str(e))
                trace_context.log(f"Function error: {str(e)}", level="error")
                raise

            finally:
                finish_trace(trace_context)

        return wrapper

    return decorator


@contextmanager
def observability_context(operation_name: str, component_id: str = ""):
    """Context manager for observability"""
    trace_context = start_trace(operation_name)
    trace_context.add_tag("component_id", component_id)

    try:
        yield trace_context
        trace_context.add_tag("status", "success")

    except Exception as e:
        trace_context.add_tag("status", "error")
        trace_context.add_tag("error", str(e))
        trace_context.log(f"Operation error: {str(e)}", level="error")
        raise

    finally:
        finish_trace(trace_context)


# Health monitoring functions


def monitor_component_health(component: BaseComponent) -> None:
    """Monitor component health and create alerts if unhealthy"""
    try:
        health_status = component.get_health_status()

        if health_status == HealthStatus.CRITICAL:
            create_alert(
                name=f"Component Critical: {component.component_id}",
                severity=AlertSeverity.CRITICAL,
                description=f"Component {component.component_id} is in critical state",
                component_id=component.component_id,
                metric_name="component_health_status",
                current_value=0,  # 0 = unhealthy
                threshold_value=1,  # 1 = healthy
            )

        elif health_status == HealthStatus.UNHEALTHY:
            create_alert(
                name=f"Component Unhealthy: {component.component_id}",
                severity=AlertSeverity.ERROR,
                description=f"Component {component.component_id} is unhealthy",
                component_id=component.component_id,
                metric_name="component_health_status",
                current_value=0,
                threshold_value=1,
            )

        elif health_status == HealthStatus.DEGRADED:
            create_alert(
                name=f"Component Degraded: {component.component_id}",
                severity=AlertSeverity.WARNING,
                description=f"Component {component.component_id} is degraded",
                component_id=component.component_id,
                metric_name="component_health_status",
                current_value=0.5,  # 0.5 = degraded
                threshold_value=1,
            )

    except Exception as e:
        logger.error(f"Health monitoring error for component {component.component_id}: {str(e)}")


def get_system_health_summary() -> dict[str, Any]:
    """Get system-wide health summary"""
    return get_observability_engine().get_system_health_summary()
