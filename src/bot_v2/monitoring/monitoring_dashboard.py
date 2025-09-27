"""
Real-time monitoring dashboard for bot_v2 orchestration system.
Tracks orchestrator performance, workflow execution, and slice health status.

EXPERIMENTAL: Internal monitoring helper used for local snapshots.
Not a production UI/dashboard.
"""

import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import deque, defaultdict


@dataclass
class MetricSnapshot:
    """Represents a point-in-time metric measurement."""
    timestamp: datetime
    metric_name: str
    value: float
    tags: Dict[str, str] = field(default_factory=dict)


class MonitoringDashboard:
    """Real-time monitoring dashboard for orchestration system."""

    # Marker used by tooling and documentation
    __experimental__ = True
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history = deque(maxlen=history_size)
        self.slice_status = {}
        self.workflow_status = {}
        self.resource_metrics = {}
        self.alert_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'error_rate': 0.05,
            'response_time_ms': 5000
        }
        self._lock = threading.Lock()
        self.start_time = datetime.now()
        self._resource_monitor_active = False
        self._start_resource_monitoring()
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict] = None):
        """Record a metric measurement with optional tags."""
        with self._lock:
            snapshot = MetricSnapshot(
                timestamp=datetime.now(),
                metric_name=name,
                value=value,
                tags=tags or {}
            )
            self.metrics_history.append(snapshot)
            
            # Check for alert conditions
            self._check_alert_conditions(name, value)
    
    def update_slice_status(self, slice_name: str, status: str, details: Optional[Dict] = None):
        """Update the health status of a feature slice."""
        with self._lock:
            self.slice_status[slice_name] = {
                'status': status,
                'last_update': datetime.now(),
                'details': details or {},
                'uptime': self._calculate_slice_uptime(slice_name)
            }
    
    def track_workflow(self, workflow_id: str, status: str, progress: float = 0.0, 
                      metadata: Optional[Dict] = None):
        """Track workflow execution status and progress."""
        with self._lock:
            self.workflow_status[workflow_id] = {
                'status': status,
                'progress': progress,
                'updated_at': datetime.now(),
                'metadata': metadata or {},
                'duration': self._calculate_workflow_duration(workflow_id)
            }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data snapshot."""
        with self._lock:
            return {
                'system_info': {
                    'uptime': str(datetime.now() - self.start_time),
                    'total_metrics': len(self.metrics_history),
                    'monitoring_since': self.start_time.isoformat()
                },
                'slice_health': self._calculate_slice_health(),
                'workflow_summary': self._get_workflow_summary(),
                'active_workflows': self._get_active_workflows(),
                'recent_metrics': self._get_recent_metrics(minutes=5),
                'resource_usage': self.resource_metrics.copy(),
                'performance_stats': self._calculate_performance_stats(),
                'alerts': self._get_active_alerts()
            }
    
    def get_slice_metrics(self, slice_name: str, minutes: int = 15) -> List[Dict]:
        """Get metrics for a specific slice over time period."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        slice_metrics = []
        
        for snapshot in self.metrics_history:
            if (snapshot.timestamp > cutoff and 
                snapshot.tags.get('slice') == slice_name):
                slice_metrics.append({
                    'timestamp': snapshot.timestamp.isoformat(),
                    'name': snapshot.metric_name,
                    'value': snapshot.value,
                    'tags': snapshot.tags
                })
        
        return slice_metrics
    
    def _start_resource_monitoring(self):
        """Start background resource monitoring thread."""
        if self._resource_monitor_active:
            return
            
        self._resource_monitor_active = True
        
        def monitor_resources():
            while self._resource_monitor_active:
                try:
                    # CPU and memory usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    disk = psutil.disk_usage('/')
                    
                    with self._lock:
                        self.resource_metrics.update({
                            'cpu_percent': cpu_percent,
                            'memory_percent': memory.percent,
                            'memory_available_gb': memory.available / (1024**3),
                            'disk_usage_percent': disk.percent,
                            'disk_free_gb': disk.free / (1024**3),
                            'last_updated': datetime.now().isoformat()
                        })
                    
                    # Record as metrics for trending
                    self.record_metric('system.cpu_percent', cpu_percent, {'type': 'resource'})
                    self.record_metric('system.memory_percent', memory.percent, {'type': 'resource'})
                    
                    time.sleep(5)  # Monitor every 5 seconds
                except Exception:
                    pass  # Continue monitoring even if some metrics fail
        
        monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        monitor_thread.start()
    
    def _calculate_slice_health(self) -> Dict[str, Dict]:
        """Calculate health scores and details for all slices."""
        health = {}
        
        for slice_name, status in self.slice_status.items():
            score = 1.0 if status['status'] == 'healthy' else 0.5 if status['status'] == 'degraded' else 0.0
            
            health[slice_name] = {
                'score': score,
                'status': status['status'],
                'last_update': status['last_update'].isoformat(),
                'uptime': status.get('uptime', 'unknown'),
                'details': status.get('details', {})
            }
        
        return health
    
    def _get_workflow_summary(self) -> Dict[str, int]:
        """Get summary statistics of workflow statuses."""
        summary = defaultdict(int)
        
        for workflow in self.workflow_status.values():
            summary[workflow['status']] += 1
        
        return dict(summary)
    
    def _get_active_workflows(self) -> List[Dict]:
        """Get currently active workflows."""
        active = []
        cutoff = datetime.now() - timedelta(minutes=10)
        
        for wf_id, status in self.workflow_status.items():
            if (status['updated_at'] > cutoff and 
                status['status'] not in ['completed', 'failed', 'cancelled']):
                active.append({
                    'id': wf_id,
                    'status': status['status'],
                    'progress': status['progress'],
                    'duration': status.get('duration', 'unknown'),
                    'metadata': status.get('metadata', {})
                })
        
        return active
    
    def _get_recent_metrics(self, minutes: int = 5) -> List[Dict]:
        """Get metrics from recent time window."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        recent = []
        
        for snapshot in self.metrics_history:
            if snapshot.timestamp > cutoff:
                recent.append({
                    'timestamp': snapshot.timestamp.isoformat(),
                    'name': snapshot.metric_name,
                    'value': snapshot.value,
                    'tags': snapshot.tags
                })
        
        return recent[-50:]  # Limit to last 50 for performance
    
    def _calculate_performance_stats(self) -> Dict[str, Any]:
        """Calculate performance statistics from recent metrics."""
        stats = {
            'avg_response_time': 0.0,
            'error_rate': 0.0,
            'throughput_per_min': 0.0,
            'success_rate': 1.0
        }
        
        # Calculate from recent metrics
        cutoff = datetime.now() - timedelta(minutes=15)
        response_times = []
        total_requests = 0
        failed_requests = 0
        
        for snapshot in self.metrics_history:
            if snapshot.timestamp > cutoff:
                if snapshot.metric_name == 'response_time_ms':
                    response_times.append(snapshot.value)
                elif snapshot.metric_name == 'request_count':
                    total_requests += snapshot.value
                elif snapshot.metric_name == 'error_count':
                    failed_requests += snapshot.value
        
        if response_times:
            stats['avg_response_time'] = sum(response_times) / len(response_times)
        
        if total_requests > 0:
            stats['error_rate'] = failed_requests / total_requests
            stats['success_rate'] = 1.0 - stats['error_rate']
            stats['throughput_per_min'] = total_requests / 15  # 15-minute window
        
        return stats
    
    def _get_active_alerts(self) -> List[Dict]:
        """Get active system alerts."""
        alerts = []
        
        # Check resource thresholds
        if self.resource_metrics:
            cpu = self.resource_metrics.get('cpu_percent', 0)
            memory = self.resource_metrics.get('memory_percent', 0)
            
            if cpu > self.alert_thresholds['cpu_percent']:
                alerts.append({
                    'type': 'resource',
                    'severity': 'warning',
                    'message': f'High CPU usage: {cpu:.1f}%',
                    'timestamp': datetime.now().isoformat()
                })
            
            if memory > self.alert_thresholds['memory_percent']:
                alerts.append({
                    'type': 'resource',
                    'severity': 'warning',
                    'message': f'High memory usage: {memory:.1f}%',
                    'timestamp': datetime.now().isoformat()
                })
        
        # Check for unhealthy slices
        for slice_name, health in self._calculate_slice_health().items():
            if health['score'] < 1.0:
                alerts.append({
                    'type': 'slice_health',
                    'severity': 'error' if health['score'] == 0.0 else 'warning',
                    'message': f'Slice {slice_name} is {health["status"]}',
                    'timestamp': datetime.now().isoformat()
                })
        
        return alerts
    
    def _check_alert_conditions(self, metric_name: str, value: float):
        """Check if metric value triggers any alerts."""
        # This would integrate with alerting system in production
        pass
    
    def _calculate_slice_uptime(self, slice_name: str) -> str:
        """Calculate uptime for a specific slice."""
        # Simplified - in production would track actual start times
        return "99.9%"
    
    def _calculate_workflow_duration(self, workflow_id: str) -> str:
        """Calculate duration of a workflow."""
        # Simplified - would track actual start times
        return "< 1min"
    
    def generate_summary(self) -> str:
        """Generate a text summary of current system status."""
        data = self.get_dashboard_data()
        summary = []
        
        summary.append("=== Bot V2 Monitoring Dashboard ===")
        summary.append(f"Uptime: {data['system_info']['uptime']}")
        summary.append(f"Total Metrics Collected: {data['system_info']['total_metrics']}")
        summary.append("")
        
        summary.append("Active Workflows:")
        for workflow in data['active_workflows']:
            summary.append(f"  - {workflow['id']}: {workflow['status']} ({workflow['progress']:.0%})")
        
        summary.append("")
        summary.append("Slice Health:")
        for slice_name, health in data['slice_health'].items():
            status_icon = "✅" if health['score'] == 1.0 else "⚠️" if health['score'] == 0.5 else "❌"
            summary.append(f"  {status_icon} {slice_name}: {health['status']} ({health['score']:.0%})")
        
        if data['resource_usage']:
            summary.append("")
            summary.append("Resource Usage:")
            cpu = data['resource_usage'].get('cpu_percent', 0)
            memory = data['resource_usage'].get('memory_percent', 0)
            summary.append(f"  CPU: {cpu:.1f}% | Memory: {memory:.1f}%")
        
        alerts = data['alerts']
        if alerts:
            summary.append("")
            summary.append(f"Active Alerts ({len(alerts)}):")
            for alert in alerts[:5]:  # Show top 5 alerts
                summary.append(f"  ⚠️ {alert['message']}")
        
        return "\n".join(summary)
    
    def stop_monitoring(self):
        """Stop background monitoring threads."""
        self._resource_monitor_active = False


# Global dashboard instance
_dashboard = None


def get_dashboard() -> MonitoringDashboard:
    """Get the global monitoring dashboard instance."""
    global _dashboard
    if _dashboard is None:
        _dashboard = MonitoringDashboard()
    return _dashboard


def record_metric(name: str, value: float, tags: Optional[Dict] = None):
    """Convenience function to record a metric."""
    get_dashboard().record_metric(name, value, tags)


def update_slice_status(slice_name: str, status: str, details: Optional[Dict] = None):
    """Convenience function to update slice status."""
    get_dashboard().update_slice_status(slice_name, status, details)


def track_workflow(workflow_id: str, status: str, progress: float = 0.0):
    """Convenience function to track workflow."""
    get_dashboard().track_workflow(workflow_id, status, progress)
