"""
Main monitoring orchestration - entry point for the slice.

Complete isolation - everything needed is local.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional
import threading
import time
import psutil
import random

from .types import (
    SystemHealth, Alert, PerformanceMetrics, ResourceUsage,
    ComponentHealth, ComponentStatus, AlertLevel, MonitorConfig,
    TradeMetrics
)
from .collectors import ResourceCollector, PerformanceCollector, ComponentCollector
from .alerting import AlertManager


class MonitoringSystem:
    """Main monitoring system."""
    
    def __init__(self, config: Optional[MonitorConfig] = None):
        """
        Initialize monitoring system.
        
        Args:
            config: Monitoring configuration
        """
        self.config = config or MonitorConfig()
        self.is_running = False
        self.thread = None
        
        # Collectors
        self.resource_collector = ResourceCollector()
        self.performance_collector = PerformanceCollector()
        self.component_collector = ComponentCollector()
        
        # Alert manager
        self.alert_manager = AlertManager(self.config)
        
        # Metrics history
        self.health_history: List[SystemHealth] = []
        self.max_history = 1000
    
    def start(self):
        """Start monitoring."""
        if self.is_running:
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._monitoring_loop)
        self.thread.daemon = True
        self.thread.start()
        
        print("✅ Monitoring system started")
        print(f"   Check interval: {self.config.check_interval_seconds}s")
        print(f"   Alerts enabled: {self.config.enable_notifications}")
    
    def stop(self):
        """Stop monitoring."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=5)
        
        print("⏹️ Monitoring system stopped")
    
    def _monitoring_loop(self):
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
                print(f"Monitoring error: {e}")
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
            active_alerts=active_alerts
        )
    
    def _determine_overall_status(
        self,
        components: Dict[str, ComponentHealth],
        resources: ResourceUsage,
        performance: PerformanceMetrics
    ) -> ComponentStatus:
        """Determine overall system status."""
        # Check for critical components
        critical_unhealthy = any(
            c.status == ComponentStatus.UNHEALTHY 
            for c in components.values()
        )
        
        if critical_unhealthy:
            return ComponentStatus.UNHEALTHY
        
        # Check resource usage
        if resources.is_high_usage():
            return ComponentStatus.DEGRADED
        
        # Check performance
        if performance.is_degraded():
            return ComponentStatus.DEGRADED
        
        # Check for any degraded components
        any_degraded = any(
            c.status == ComponentStatus.DEGRADED
            for c in components.values()
        )
        
        if any_degraded:
            return ComponentStatus.DEGRADED
        
        return ComponentStatus.HEALTHY
    
    def _check_alerts(self, health: SystemHealth):
        """Check for alert conditions."""
        # Resource alerts
        if health.resource_usage.cpu_percent > self.config.alert_threshold_cpu:
            self.alert_manager.create_alert(
                level=AlertLevel.WARNING,
                component="System",
                message=f"High CPU usage: {health.resource_usage.cpu_percent:.1f}%",
                details={"cpu_percent": health.resource_usage.cpu_percent}
            )
        
        if health.resource_usage.memory_percent > self.config.alert_threshold_memory:
            self.alert_manager.create_alert(
                level=AlertLevel.WARNING,
                component="System",
                message=f"High memory usage: {health.resource_usage.memory_percent:.1f}%",
                details={"memory_percent": health.resource_usage.memory_percent}
            )
        
        if health.resource_usage.disk_percent > self.config.alert_threshold_disk:
            self.alert_manager.create_alert(
                level=AlertLevel.ERROR,
                component="System",
                message=f"High disk usage: {health.resource_usage.disk_percent:.1f}%",
                details={"disk_percent": health.resource_usage.disk_percent}
            )
        
        # Performance alerts
        if health.performance.error_rate > self.config.alert_threshold_error_rate:
            self.alert_manager.create_alert(
                level=AlertLevel.ERROR,
                component="Performance",
                message=f"High error rate: {health.performance.error_rate:.2%}",
                details={"error_rate": health.performance.error_rate}
            )
        
        if health.performance.avg_response_time_ms > self.config.alert_threshold_response_time_ms:
            self.alert_manager.create_alert(
                level=AlertLevel.WARNING,
                component="Performance",
                message=f"Slow response time: {health.performance.avg_response_time_ms:.1f}ms",
                details={"response_time_ms": health.performance.avg_response_time_ms}
            )
        
        # Component alerts
        for name, component in health.components.items():
            if component.status == ComponentStatus.UNHEALTHY:
                self.alert_manager.create_alert(
                    level=AlertLevel.CRITICAL,
                    component=name,
                    message=f"Component unhealthy: {name}",
                    details=component.details
                )
    
    def get_current_health(self) -> Optional[SystemHealth]:
        """Get current system health."""
        if self.health_history:
            return self.health_history[-1]
        return None
    
    def get_alerts(self) -> List[Alert]:
        """Get all alerts."""
        return self.alert_manager.get_all_alerts()
    
    def get_active_alerts(self) -> List[Alert]:
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
            slippage_bps=2.3
        )


# Global monitoring instance
_monitor: Optional[MonitoringSystem] = None


def start_monitoring(config: Optional[MonitorConfig] = None):
    """
    Start system monitoring.
    
    Args:
        config: Optional monitoring configuration
    """
    global _monitor
    
    if _monitor and _monitor.is_running:
        print("Monitoring already running")
        return
    
    _monitor = MonitoringSystem(config)
    _monitor.start()


def stop_monitoring():
    """Stop system monitoring."""
    global _monitor
    
    if not _monitor:
        return
    
    _monitor.stop()
    _monitor = None


def get_system_health() -> Optional[SystemHealth]:
    """
    Get current system health.
    
    Returns:
        SystemHealth object or None
    """
    if not _monitor:
        return None
    
    return _monitor.get_current_health()


def get_alerts() -> List[Alert]:
    """
    Get all system alerts.
    
    Returns:
        List of Alert objects
    """
    if not _monitor:
        return []
    
    return _monitor.get_alerts()


def get_active_alerts() -> List[Alert]:
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