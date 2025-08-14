"""
Production Monitoring and Alerting System

Comprehensive monitoring for production trading systems:
- Real-time performance metrics
- System health monitoring
- Alert management and notifications
- Anomaly detection
- Dashboard and reporting
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np


class MetricType(Enum):
    """Types of metrics to monitor"""

    COUNTER = "counter"  # Monotonically increasing
    GAUGE = "gauge"  # Point-in-time value
    HISTOGRAM = "histogram"  # Distribution of values
    RATE = "rate"  # Events per time unit


class AlertSeverity(Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status"""

    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SILENCED = "silenced"


@dataclass
class Metric:
    """Single metric data point"""

    name: str
    value: float
    timestamp: datetime
    type: MetricType
    tags: dict[str, str] = field(default_factory=dict)
    unit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "type": self.type.value,
            "tags": self.tags,
            "unit": self.unit,
        }


@dataclass
class Alert:
    """Alert instance"""

    id: str
    name: str
    message: str
    severity: AlertSeverity
    timestamp: datetime
    status: AlertStatus = AlertStatus.ACTIVE
    metric_name: str | None = None
    threshold: float | None = None
    actual_value: float | None = None
    tags: dict[str, str] = field(default_factory=dict)
    resolved_at: datetime | None = None
    acknowledged_at: datetime | None = None
    acknowledged_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "message": self.message,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "metric_name": self.metric_name,
            "threshold": self.threshold,
            "actual_value": self.actual_value,
            "tags": self.tags,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
        }


@dataclass
class AlertRule:
    """Rule for generating alerts"""

    name: str
    metric_name: str
    condition: str  # "gt", "lt", "eq", "ne"
    threshold: float
    severity: AlertSeverity
    window_seconds: int = 60
    min_occurrences: int = 1
    cooldown_seconds: int = 300
    enabled: bool = True
    tags: dict[str, str] = field(default_factory=dict)

    def evaluate(self, value: float) -> bool:
        """Check if rule is violated"""
        if self.condition == "gt":
            return value > self.threshold
        elif self.condition == "lt":
            return value < self.threshold
        elif self.condition == "eq":
            return abs(value - self.threshold) < 0.0001
        elif self.condition == "ne":
            return abs(value - self.threshold) >= 0.0001
        else:
            return False


class MetricsCollector:
    """
    Collects and aggregates system metrics.

    Features:
    - Time-series storage
    - Statistical aggregation
    - Rolling windows
    - Export to various formats
    """

    def __init__(self, retention_hours: int = 24) -> None:
        self.retention_hours = retention_hours
        self.metrics = {}  # name -> deque of Metric objects
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)

        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()

    def record(self, metric: Metric) -> None:
        """Record a metric"""
        with self.lock:
            if metric.name not in self.metrics:
                self.metrics[metric.name] = deque()

            self.metrics[metric.name].append(metric)

    def record_counter(self, name: str, value: float = 1, tags: dict[str, str] = None) -> None:
        """Record counter metric"""
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            type=MetricType.COUNTER,
            tags=tags or {},
        )
        self.record(metric)

    def record_gauge(self, name: str, value: float, tags: dict[str, str] = None) -> None:
        """Record gauge metric"""
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            type=MetricType.GAUGE,
            tags=tags or {},
        )
        self.record(metric)

    def get_metric_series(
        self, name: str, start_time: datetime | None = None, end_time: datetime | None = None
    ) -> list[Metric]:
        """Get time series for metric"""
        with self.lock:
            if name not in self.metrics:
                return []

            series = list(self.metrics[name])

            # Filter by time range
            if start_time:
                series = [m for m in series if m.timestamp >= start_time]
            if end_time:
                series = [m for m in series if m.timestamp <= end_time]

            return series

    def get_metric_stats(self, name: str, window_seconds: int = 300) -> dict[str, float]:
        """Get statistics for metric over window"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(seconds=window_seconds)

        series = self.get_metric_series(name, start_time, end_time)

        if not series:
            return {}

        values = [m.value for m in series]

        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "stddev": statistics.stdev(values) if len(values) > 1 else 0,
            "rate": len(values) / window_seconds,
        }

    def get_all_metrics(self) -> dict[str, Any]:
        """Get all current metrics"""
        with self.lock:
            result = {}

            for name, series in self.metrics.items():
                if series:
                    latest = series[-1]
                    result[name] = {
                        "value": latest.value,
                        "timestamp": latest.timestamp.isoformat(),
                        "type": latest.type.value,
                        "stats": self.get_metric_stats(name),
                    }

            return result

    def _cleanup_loop(self) -> None:
        """Remove old metrics periodically"""
        while True:
            time.sleep(3600)  # Run hourly

            cutoff = datetime.utcnow() - timedelta(hours=self.retention_hours)

            with self.lock:
                for name in self.metrics:
                    # Remove old entries
                    while self.metrics[name] and self.metrics[name][0].timestamp < cutoff:
                        self.metrics[name].popleft()


class AlertManager:
    """
    Manages alerts and notifications.

    Features:
    - Rule-based alerting
    - Alert deduplication
    - Notification routing
    - Alert history
    """

    def __init__(self) -> None:
        self.rules = {}  # name -> AlertRule
        self.active_alerts = {}  # alert_id -> Alert
        self.alert_history = deque(maxlen=1000)
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        self.notification_handlers = []
        self.last_alert_times = {}  # rule_name -> timestamp

    def add_rule(self, rule: AlertRule) -> None:
        """Add alert rule"""
        with self.lock:
            self.rules[rule.name] = rule
            self.logger.info(f"Added alert rule: {rule.name}")

    def remove_rule(self, name: str) -> None:
        """Remove alert rule"""
        with self.lock:
            if name in self.rules:
                del self.rules[name]
                self.logger.info(f"Removed alert rule: {name}")

    def add_notification_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add notification handler"""
        self.notification_handlers.append(handler)

    def check_rules(self, metrics: dict[str, Metric]) -> None:
        """Check all rules against current metrics"""
        with self.lock:
            for _rule_name, rule in self.rules.items():
                if not rule.enabled:
                    continue

                # Check if metric exists
                if rule.metric_name not in metrics:
                    continue

                metric = metrics[rule.metric_name]

                # Check if rule is violated
                if rule.evaluate(metric.value):
                    self._trigger_alert(rule, metric)
                else:
                    self._resolve_alert(rule, metric)

    def _trigger_alert(self, rule: AlertRule, metric: Metric) -> None:
        """Trigger new alert or update existing"""
        alert_id = f"{rule.name}_{rule.metric_name}"

        # Check cooldown
        if rule.name in self.last_alert_times:
            last_time = self.last_alert_times[rule.name]
            if (datetime.utcnow() - last_time).total_seconds() < rule.cooldown_seconds:
                return  # Still in cooldown

        # Check if alert already exists
        if alert_id in self.active_alerts:
            # Update existing alert
            alert = self.active_alerts[alert_id]
            alert.actual_value = metric.value
            alert.timestamp = datetime.utcnow()
        else:
            # Create new alert
            alert = Alert(
                id=alert_id,
                name=rule.name,
                message=f"{rule.metric_name} {rule.condition} {rule.threshold}: {metric.value:.2f}",
                severity=rule.severity,
                timestamp=datetime.utcnow(),
                metric_name=rule.metric_name,
                threshold=rule.threshold,
                actual_value=metric.value,
                tags={**rule.tags, **metric.tags},
            )

            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            self.last_alert_times[rule.name] = datetime.utcnow()

            # Send notifications
            self._notify(alert)

            self.logger.warning(f"Alert triggered: {alert.message}")

    def _resolve_alert(self, rule: AlertRule, metric: Metric) -> None:
        """Resolve alert if it exists"""
        alert_id = f"{rule.name}_{rule.metric_name}"

        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.utcnow()

            # Move to history
            del self.active_alerts[alert_id]

            self.logger.info(f"Alert resolved: {alert.name}")

    def _notify(self, alert: Alert) -> None:
        """Send notifications for alert"""
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Notification handler error: {e}")

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> None:
        """Acknowledge an alert"""
        with self.lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = datetime.utcnow()
                alert.acknowledged_by = acknowledged_by

                self.logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")

    def get_active_alerts(self) -> list[Alert]:
        """Get all active alerts"""
        with self.lock:
            return list(self.active_alerts.values())

    def get_alert_history(self, hours: int = 24) -> list[Alert]:
        """Get alert history"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        with self.lock:
            return [a for a in self.alert_history if a.timestamp >= cutoff]


class SystemMonitor:
    """
    Comprehensive system monitoring.

    Monitors:
    - Trading performance
    - System resources
    - Data pipeline health
    - Model performance
    - Risk metrics
    """

    def __init__(self) -> None:
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.logger = logging.getLogger(__name__)
        self.monitoring_thread = None
        self.is_running = False

        # Initialize default alert rules
        self._init_default_rules()

    def _init_default_rules(self) -> None:
        """Initialize default monitoring rules"""
        default_rules = [
            AlertRule(
                name="high_cpu_usage",
                metric_name="system.cpu_percent",
                condition="gt",
                threshold=90.0,
                severity=AlertSeverity.WARNING,
                window_seconds=60,
            ),
            AlertRule(
                name="high_memory_usage",
                metric_name="system.memory_percent",
                condition="gt",
                threshold=85.0,
                severity=AlertSeverity.WARNING,
                window_seconds=60,
            ),
            AlertRule(
                name="high_error_rate",
                metric_name="errors.rate",
                condition="gt",
                threshold=10.0,
                severity=AlertSeverity.ERROR,
                window_seconds=300,
            ),
            AlertRule(
                name="low_data_rate",
                metric_name="data.ingestion_rate",
                condition="lt",
                threshold=1.0,
                severity=AlertSeverity.WARNING,
                window_seconds=60,
            ),
            AlertRule(
                name="high_latency",
                metric_name="latency.p99",
                condition="gt",
                threshold=1000.0,  # milliseconds
                severity=AlertSeverity.WARNING,
                window_seconds=60,
            ),
            AlertRule(
                name="trading_loss",
                metric_name="trading.daily_pnl",
                condition="lt",
                threshold=-1000.0,
                severity=AlertSeverity.ERROR,
                window_seconds=3600,
            ),
        ]

        for rule in default_rules:
            self.alert_manager.add_rule(rule)

    def start(self) -> None:
        """Start monitoring"""
        if self.is_running:
            return

        self.is_running = True
        self.monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitoring_thread.start()

        self.logger.info("System monitoring started")

    def stop(self) -> None:
        """Stop monitoring"""
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

        self.logger.info("System monitoring stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Collect system metrics
                self._collect_system_metrics()

                # Check alert rules
                current_metrics = self._get_current_metrics()
                self.alert_manager.check_rules(current_metrics)

                # Sleep interval
                time.sleep(10)  # Collect every 10 seconds

            except Exception as e:
                self.logger.error(f"Monitoring error: {e}\n{traceback.format_exc()}")
                self.metrics_collector.record_counter("monitoring.errors")

    def _collect_system_metrics(self) -> None:
        """Collect system resource metrics"""
        try:
            import psutil

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics_collector.record_gauge("system.cpu_percent", cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics_collector.record_gauge("system.memory_percent", memory.percent)
            self.metrics_collector.record_gauge(
                "system.memory_available_gb", memory.available / (1024**3)
            )

            # Disk usage
            disk = psutil.disk_usage("/")
            self.metrics_collector.record_gauge("system.disk_percent", disk.percent)

            # Network I/O
            net_io = psutil.net_io_counters()
            self.metrics_collector.record_counter("system.network_bytes_sent", net_io.bytes_sent)
            self.metrics_collector.record_counter("system.network_bytes_recv", net_io.bytes_recv)

        except ImportError:
            # psutil not available, use dummy metrics
            self.metrics_collector.record_gauge("system.cpu_percent", np.random.uniform(20, 40))
            self.metrics_collector.record_gauge("system.memory_percent", np.random.uniform(30, 50))

    def _get_current_metrics(self) -> dict[str, Metric]:
        """Get latest value for each metric"""
        result = {}

        for name, series in self.metrics_collector.metrics.items():
            if series:
                result[name] = series[-1]

        return result

    def record_trading_metrics(
        self, portfolio_value: float, daily_pnl: float, position_count: int, trade_count: int
    ) -> None:
        """Record trading performance metrics"""
        self.metrics_collector.record_gauge("trading.portfolio_value", portfolio_value)
        self.metrics_collector.record_gauge("trading.daily_pnl", daily_pnl)
        self.metrics_collector.record_gauge("trading.position_count", position_count)
        self.metrics_collector.record_counter("trading.trade_count", trade_count)

    def record_model_metrics(
        self, model_name: str, accuracy: float, latency_ms: float, predictions_count: int
    ) -> None:
        """Record ML model performance metrics"""
        tags = {"model": model_name}
        self.metrics_collector.record_gauge("model.accuracy", accuracy, tags)
        self.metrics_collector.record_gauge("model.latency_ms", latency_ms, tags)
        self.metrics_collector.record_counter("model.predictions", predictions_count, tags)

    def record_data_metrics(
        self, ingestion_rate: float, processing_rate: float, error_count: int
    ) -> None:
        """Record data pipeline metrics"""
        self.metrics_collector.record_gauge("data.ingestion_rate", ingestion_rate)
        self.metrics_collector.record_gauge("data.processing_rate", processing_rate)
        self.metrics_collector.record_counter("data.errors", error_count)

    def get_dashboard_data(self) -> dict[str, Any]:
        """Get data for monitoring dashboard"""
        return {
            "metrics": self.metrics_collector.get_all_metrics(),
            "active_alerts": [a.to_dict() for a in self.alert_manager.get_active_alerts()],
            "alert_history": [a.to_dict() for a in self.alert_manager.get_alert_history(24)],
            "system_status": self._get_system_status(),
        }

    def _get_system_status(self) -> dict[str, Any]:
        """Get overall system status"""
        active_alerts = self.alert_manager.get_active_alerts()

        # Determine overall status
        if any(a.severity == AlertSeverity.CRITICAL for a in active_alerts):
            status = "critical"
        elif any(a.severity == AlertSeverity.ERROR for a in active_alerts):
            status = "error"
        elif any(a.severity == AlertSeverity.WARNING for a in active_alerts):
            status = "warning"
        else:
            status = "healthy"

        return {
            "status": status,
            "active_alert_count": len(active_alerts),
            "uptime_seconds": time.time(),  # Simplified
            "last_check": datetime.utcnow().isoformat(),
        }


def demo_monitoring() -> None:
    """Demo monitoring system"""
    print("🚀 Production Monitoring Demo")
    print("=" * 50)

    # Create monitor
    monitor = SystemMonitor()

    # Add custom notification handler
    def print_alert(alert: Alert) -> None:
        print(f"\n🚨 ALERT: [{alert.severity.value.upper()}] {alert.message}")

    monitor.alert_manager.add_notification_handler(print_alert)

    # Start monitoring
    monitor.start()
    print("📊 Monitoring started")

    # Simulate some metrics
    for i in range(10):
        # Trading metrics
        portfolio_value = 100000 + np.random.normal(0, 1000)
        daily_pnl = np.random.normal(100, 500)
        monitor.record_trading_metrics(portfolio_value, daily_pnl, 5, i)

        # Model metrics
        monitor.record_model_metrics(
            "transformer",
            accuracy=0.65 + np.random.uniform(-0.1, 0.1),
            latency_ms=50 + np.random.uniform(-10, 20),
            predictions_count=100,
        )

        # Data metrics
        monitor.record_data_metrics(
            ingestion_rate=100 + np.random.uniform(-20, 20),
            processing_rate=95 + np.random.uniform(-15, 15),
            error_count=np.random.poisson(1),
        )

        # Simulate high CPU occasionally
        if i == 5:
            monitor.metrics_collector.record_gauge("system.cpu_percent", 95.0)

        time.sleep(2)

    # Get dashboard data
    dashboard = monitor.get_dashboard_data()

    print("\n📊 Dashboard Summary:")
    print(f"   System Status: {dashboard['system_status']['status']}")
    print(f"   Active Alerts: {dashboard['system_status']['active_alert_count']}")

    print("\n📈 Key Metrics:")
    for name, data in list(dashboard["metrics"].items())[:5]:
        print(f"   {name}: {data['value']:.2f}")

    # Stop monitoring
    monitor.stop()
    print("\n✅ Monitoring stopped")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Run demo
    demo_monitoring()
