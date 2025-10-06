"""
Main monitoring orchestration - entry point for the slice.

Complete isolation - everything needed is local.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bot_v2.orchestration.core import IBotRuntime

from bot_v2.monitoring.alerts import Alert, AlertDispatcher
from bot_v2.monitoring.alerts_manager import AlertManager
from bot_v2.monitoring.interfaces import (
    ComponentHealth,
    ComponentStatus,
    MonitorConfig,
    PerformanceMetrics,
    ResourceUsage,
    SystemHealth,
    TradeMetrics,
)
from bot_v2.monitoring.system.collectors import (
    ComponentCollector,
    PerformanceCollector,
    ResourceCollector,
)

logger = logging.getLogger(__name__)


class MonitoringSystem:
    """Main monitoring system."""

    def __init__(
        self,
        bot: IBotRuntime | None = None,
        config: MonitorConfig | None = None,
        *,
        profile: Any | None = None,
        alert_config_path: str | Path | None = None,
        alert_manager: AlertManager | None = None,
        monitor_settings: Mapping[str, object] | None = None,
        alert_settings: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize monitoring system."""

        self._bot = bot
        self.config = config or self._build_monitor_config(monitor_settings)
        self.is_running: bool = False
        self.thread: threading.Thread | None = None

        # Collectors
        self.resource_collector = ResourceCollector()
        self.performance_collector = PerformanceCollector()
        self.component_collector = ComponentCollector()

        if alert_manager is not None:
            self.alert_manager = alert_manager
            alerts_enabled = self.config.enable_notifications
        else:
            effective_profile = profile if profile is not None else os.getenv("PERPS_PROFILE")
            if alert_settings is not None:
                self.alert_manager = AlertManager.from_settings(alert_settings)
                alerts_enabled = bool(alert_settings.get("enabled", True))
            elif self.config.enable_notifications:
                self.alert_manager = AlertManager.from_profile_yaml(
                    path=alert_config_path, profile=effective_profile
                )
                alerts_enabled = True
            else:
                self.alert_manager = AlertManager(dispatcher=AlertDispatcher())
                alerts_enabled = False

        self._alerts_enabled = alerts_enabled and self.config.enable_notifications

        # Metrics history
        self.health_history: list[SystemHealth] = []
        self.max_history = 1000

    def start(self) -> None:
        """Start monitoring."""
        if self.is_running:
            return

        self.is_running = True
        thread = threading.Thread(target=self._monitoring_loop)
        thread.daemon = True
        thread.start()
        self.thread = thread

        logger.info(
            "Monitoring system started",
            extra={
                "check_interval_seconds": self.config.check_interval_seconds,
                "alerts_enabled": self._alerts_enabled,
            },
        )

    def stop(self) -> None:
        """Stop monitoring."""
        if not self.is_running:
            return

        self.is_running = False
        if self.thread:
            self.thread.join(timeout=5)

        logger.info("Monitoring system stopped")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_running:
            try:
                # Collect metrics
                health = self._collect_health()

                # Store history
                self.health_history.append(health)
                if len(self.health_history) > self.max_history:
                    self.health_history.pop(0)

                # Check for alerts
                self._check_alerts(health)

                # Sleep until next check
                time.sleep(self.config.check_interval_seconds)

            except Exception as e:
                logger.error(
                    "Monitoring loop error",
                    extra={"error": str(e)},
                    exc_info=True,
                )
                time.sleep(self.config.check_interval_seconds)

    def _collect_health(self) -> SystemHealth:
        """Collect current system health."""
        # Collect resource usage
        resources = self.resource_collector.collect()

        # Collect performance metrics
        performance = self.performance_collector.collect()

        # Collect component health
        components = self.component_collector.collect()

        # Determine overall status
        overall_status = self._determine_overall_status(components, resources, performance)

        # Count active alerts
        active_alerts = len(self.alert_manager.get_active_alerts())

        return SystemHealth(
            timestamp=datetime.now(),
            overall_status=overall_status,
            components=components,
            resource_usage=resources,
            performance=performance,
            active_alerts=active_alerts,
        )

    def _determine_overall_status(
        self,
        components: dict[str, ComponentHealth],
        resources: ResourceUsage,
        performance: PerformanceMetrics,
    ) -> ComponentStatus:
        """Determine overall system status."""
        # Check for critical components
        critical_unhealthy = any(c.status == ComponentStatus.UNHEALTHY for c in components.values())

        if critical_unhealthy:
            return ComponentStatus.UNHEALTHY

        # Check resource usage
        if resources.is_high_usage():
            return ComponentStatus.DEGRADED

        # Check performance
        if performance.is_degraded():
            return ComponentStatus.DEGRADED

        # Check for any degraded components
        any_degraded = any(c.status == ComponentStatus.DEGRADED for c in components.values())

        if any_degraded:
            return ComponentStatus.DEGRADED

        return ComponentStatus.HEALTHY

    def _check_alerts(self, health: SystemHealth) -> None:
        """Check for alert conditions."""
        from bot_v2.monitoring.alerts import AlertLevel

        # Resource alerts
        if health.resource_usage.cpu_percent >= self.config.alert_threshold_cpu:
            self.alert_manager.create_alert(
                level=AlertLevel.WARNING,
                source="System",
                message=f"High CPU usage: {health.resource_usage.cpu_percent:.1f}%",
                context={"cpu_percent": health.resource_usage.cpu_percent},
                dispatch=self._alerts_enabled,
            )

        if health.resource_usage.memory_percent >= self.config.alert_threshold_memory:
            self.alert_manager.create_alert(
                level=AlertLevel.WARNING,
                source="System",
                message=f"High memory usage: {health.resource_usage.memory_percent:.1f}%",
                context={"memory_percent": health.resource_usage.memory_percent},
                dispatch=self._alerts_enabled,
            )

        if health.resource_usage.disk_percent >= self.config.alert_threshold_disk:
            self.alert_manager.create_alert(
                level=AlertLevel.ERROR,
                source="System",
                message=f"High disk usage: {health.resource_usage.disk_percent:.1f}%",
                context={"disk_percent": health.resource_usage.disk_percent},
                dispatch=self._alerts_enabled,
            )

        # Performance alerts
        if health.performance.error_rate > self.config.alert_threshold_error_rate:
            self.alert_manager.create_alert(
                level=AlertLevel.ERROR,
                source="Performance",
                message=f"High error rate: {health.performance.error_rate:.2%}",
                context={"error_rate": health.performance.error_rate},
                dispatch=self._alerts_enabled,
            )

        if health.performance.avg_response_time_ms > self.config.alert_threshold_response_time_ms:
            self.alert_manager.create_alert(
                level=AlertLevel.WARNING,
                source="Performance",
                message=f"Slow response time: {health.performance.avg_response_time_ms:.1f}ms",
                context={"response_time_ms": health.performance.avg_response_time_ms},
                dispatch=self._alerts_enabled,
            )

        # Component alerts
        for name, component in health.components.items():
            if component.status == ComponentStatus.UNHEALTHY:
                self.alert_manager.create_alert(
                    level=AlertLevel.CRITICAL,
                    source=name,
                    message=f"Component unhealthy: {name}",
                    context=component.details,
                    dispatch=self._alerts_enabled,
                )

    def _build_monitor_config(self, monitor_settings: Mapping[str, object] | None) -> MonitorConfig:
        if monitor_settings is None:
            return MonitorConfig()

        config = MonitorConfig()

        metrics = monitor_settings.get("metrics")
        if isinstance(metrics, Mapping):
            interval = metrics.get("interval_seconds")
            if isinstance(interval, (int, float)):
                config.check_interval_seconds = int(interval)

        thresholds = monitor_settings.get("thresholds")
        if isinstance(thresholds, Mapping):
            cpu = thresholds.get("cpu_percent")
            if isinstance(cpu, (int, float)):
                config.alert_threshold_cpu = float(cpu)
            memory = thresholds.get("memory_percent")
            if isinstance(memory, (int, float)):
                config.alert_threshold_memory = float(memory)
            disk = thresholds.get("disk_percent")
            if isinstance(disk, (int, float)):
                config.alert_threshold_disk = float(disk)
            error_rate = thresholds.get("error_rate")
            if isinstance(error_rate, (int, float)):
                config.alert_threshold_error_rate = float(error_rate)
            response_ms = thresholds.get("response_time_ms")
            if isinstance(response_ms, (int, float)):
                config.alert_threshold_response_time_ms = float(response_ms)

        notifications = monitor_settings.get("enable_notifications")
        if isinstance(notifications, bool):
            config.enable_notifications = notifications

        retention = monitor_settings.get("retention_days")
        if isinstance(retention, (int, float)):
            config.retention_days = int(retention)

        return config

    def get_current_health(self) -> SystemHealth | None:
        """Get current system health."""
        if self.health_history:
            return self.health_history[-1]
        return None

    def get_alerts(self) -> list[Alert]:
        """Get all alerts."""
        return self.alert_manager.get_all_alerts()

    def get_active_alerts(self) -> list[Alert]:
        """Get active alerts."""
        return self.alert_manager.get_active_alerts()

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        return self.alert_manager.acknowledge_alert(alert_id)

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        return self.alert_manager.resolve_alert(alert_id)

    def get_trade_metrics(self) -> TradeMetrics:
        """Get trading-specific metrics."""
        # This would integrate with trading systems
        return TradeMetrics(
            total_trades_today=42,
            successful_trades=38,
            failed_trades=4,
            total_volume=150000.0,
            total_pnl=2500.0,
            win_rate=0.65,
            avg_execution_time_ms=125.5,
            slippage_bps=2.3,
        )


# Global monitoring instance
_monitor: MonitoringSystem | None = None


def start_monitoring(config: MonitorConfig | None = None) -> None:
    """
    Start system monitoring.

    Args:
        config: Optional monitoring configuration
    """
    global _monitor

    if _monitor and _monitor.is_running:
        logger.warning("Monitoring already running")
        return

    _monitor = MonitoringSystem(config=config)
    _monitor.start()


def stop_monitoring() -> None:
    """Stop system monitoring."""
    global _monitor

    if not _monitor:
        return

    _monitor.stop()
    _monitor = None


def get_system_health() -> SystemHealth | None:
    """
    Get current system health.

    Returns:
        SystemHealth object or None
    """
    if not _monitor:
        return None

    return _monitor.get_current_health()


def get_alerts() -> list[Alert]:
    """
    Get all system alerts.

    Returns:
        List of Alert objects
    """
    if not _monitor:
        return []

    return _monitor.get_alerts()


def get_active_alerts() -> list[Alert]:
    """
    Get active system alerts.

    Returns:
        List of active Alert objects
    """
    if not _monitor:
        return []

    return _monitor.get_active_alerts()


def acknowledge_alert(alert_id: str) -> bool:
    """
    Acknowledge an alert.

    Args:
        alert_id: Alert ID to acknowledge

    Returns:
        True if acknowledged successfully
    """
    if not _monitor:
        return False

    return _monitor.acknowledge_alert(alert_id)


def resolve_alert(alert_id: str) -> bool:
    """
    Resolve an alert.

    Args:
        alert_id: Alert ID to resolve

    Returns:
        True if resolved successfully
    """
    if not _monitor:
        return False

    return _monitor.resolve_alert(alert_id)
