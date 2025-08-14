"""
Anomaly Alert Generation and Management System
Phase 3, Week 3-4: RISK-013, RISK-014, RISK-015
Alert generation, visualization, and investigation tools
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict, deque
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert delivery channels"""
    LOG = "log"
    EMAIL = "email"
    WEBHOOK = "webhook"
    DASHBOARD = "dashboard"
    SMS = "sms"


@dataclass
class Alert:
    """Alert object"""
    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    
    # Alert details
    title: str
    message: str
    source: str  # Component that generated alert
    metric_name: Optional[str] = None
    
    # Values
    observed_value: Optional[float] = None
    threshold_value: Optional[float] = None
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # Actions
    recommended_action: Optional[str] = None
    auto_resolved: bool = False
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity.value,
            'title': self.title,
            'message': self.message,
            'source': self.source,
            'metric_name': self.metric_name,
            'observed_value': self.observed_value,
            'threshold_value': self.threshold_value,
            'tags': self.tags,
            'acknowledged': self.acknowledged
        }
    
    def to_text(self) -> str:
        """Convert to text format"""
        text = f"[{self.severity.value.upper()}] {self.title}\n"
        text += f"Time: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
        text += f"Source: {self.source}\n"
        text += f"Message: {self.message}\n"
        
        if self.metric_name:
            text += f"Metric: {self.metric_name}\n"
        if self.observed_value is not None:
            text += f"Value: {self.observed_value:.4f}\n"
        if self.threshold_value is not None:
            text += f"Threshold: {self.threshold_value:.4f}\n"
        if self.recommended_action:
            text += f"Action: {self.recommended_action}\n"
        
        return text


@dataclass
class AlertConfig:
    """Alert system configuration"""
    # Thresholds
    cooldown_period: int = 300  # Seconds between same alerts
    max_alerts_per_hour: int = 50
    
    # Aggregation
    aggregate_window: int = 60  # Seconds to aggregate similar alerts
    deduplication_window: int = 300  # Seconds to deduplicate
    
    # Channels
    enabled_channels: List[AlertChannel] = field(default_factory=lambda: [AlertChannel.LOG, AlertChannel.DASHBOARD])
    
    # Email settings (if email channel enabled)
    email_recipients: List[str] = field(default_factory=list)
    email_sender: str = ""
    smtp_server: str = ""
    smtp_port: int = 587
    
    # Webhook settings (if webhook channel enabled)
    webhook_url: str = ""
    
    # Severity mapping
    severity_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'info': 0.3,
        'warning': 0.5,
        'error': 0.7,
        'critical': 0.9
    })


class AlertGenerator:
    """
    Alert generation system for anomalies and risk events.
    
    Features:
    - Multi-channel delivery
    - Alert aggregation and deduplication
    - Severity-based routing
    - Cooldown periods
    - Action recommendations
    """
    
    def __init__(self, config: Optional[AlertConfig] = None):
        """
        Initialize alert generator.
        
        Args:
            config: Alert configuration
        """
        self.config = config or AlertConfig()
        
        # Alert history
        self.alerts: List[Alert] = []
        self.alert_counts: Dict[str, int] = defaultdict(int)
        self.last_alert_time: Dict[str, datetime] = {}
        
        # Alert queues by channel
        self.channel_queues: Dict[AlertChannel, deque] = {
            channel: deque(maxlen=100) for channel in AlertChannel
        }
        
        # Callbacks
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Alert ID counter
        self._alert_counter = 0
    
    def generate_alert(self,
                      title: str,
                      message: str,
                      severity: Union[AlertSeverity, float],
                      source: str,
                      metric_name: Optional[str] = None,
                      observed_value: Optional[float] = None,
                      threshold_value: Optional[float] = None,
                      context: Optional[Dict] = None,
                      tags: Optional[List[str]] = None) -> Optional[Alert]:
        """
        Generate an alert.
        
        Args:
            title: Alert title
            message: Alert message
            severity: Severity level or score (0-1)
            source: Source component
            metric_name: Associated metric
            observed_value: Observed value
            threshold_value: Threshold value
            context: Additional context
            tags: Alert tags
            
        Returns:
            Generated alert or None if suppressed
        """
        # Convert float severity to enum
        if isinstance(severity, float):
            severity = self._severity_from_score(severity)
        
        # Check cooldown
        alert_key = f"{source}_{metric_name}_{severity.value}"
        if self._is_in_cooldown(alert_key):
            logger.debug(f"Alert suppressed due to cooldown: {alert_key}")
            return None
        
        # Check rate limit
        if self._exceeds_rate_limit():
            logger.warning("Alert rate limit exceeded")
            return None
        
        # Generate alert ID
        self._alert_counter += 1
        alert_id = f"ALERT_{datetime.now().strftime('%Y%m%d')}_{self._alert_counter:05d}"
        
        # Create alert
        alert = Alert(
            alert_id=alert_id,
            timestamp=datetime.now(),
            severity=severity,
            title=title,
            message=message,
            source=source,
            metric_name=metric_name,
            observed_value=observed_value,
            threshold_value=threshold_value,
            context=context or {},
            tags=tags or []
        )
        
        # Add recommended action
        alert.recommended_action = self._get_recommended_action(alert)
        
        # Store alert
        self.alerts.append(alert)
        self.alert_counts[alert_key] += 1
        self.last_alert_time[alert_key] = alert.timestamp
        
        # Route to channels
        self._route_alert(alert)
        
        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
        
        return alert
    
    def generate_anomaly_alert(self, anomaly: Any) -> Optional[Alert]:
        """
        Generate alert from anomaly object.
        
        Args:
            anomaly: Anomaly object (from anomaly_detector.py)
            
        Returns:
            Generated alert
        """
        # Map anomaly severity to alert severity
        if anomaly.severity > 0.9:
            severity = AlertSeverity.CRITICAL
        elif anomaly.severity > 0.7:
            severity = AlertSeverity.ERROR
        elif anomaly.severity > 0.5:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.INFO
        
        # Create alert message
        message = f"{anomaly.description}"
        if anomaly.deviation:
            message += f" (deviation: {anomaly.deviation:.2f})"
        
        return self.generate_alert(
            title=f"Anomaly Detected: {anomaly.anomaly_type.value}",
            message=message,
            severity=severity,
            source="anomaly_detector",
            metric_name=anomaly.metric_name,
            observed_value=anomaly.observed_value,
            threshold_value=anomaly.expected_value,
            context={'anomaly_type': anomaly.anomaly_type.value,
                    'detection_method': anomaly.detection_method.value},
            tags=[anomaly.anomaly_type.value, anomaly.detection_method.value]
        )
    
    def _severity_from_score(self, score: float) -> AlertSeverity:
        """Convert severity score to enum"""
        thresholds = self.config.severity_thresholds
        
        if score >= thresholds['critical']:
            return AlertSeverity.CRITICAL
        elif score >= thresholds['error']:
            return AlertSeverity.ERROR
        elif score >= thresholds['warning']:
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO
    
    def _is_in_cooldown(self, alert_key: str) -> bool:
        """Check if alert is in cooldown period"""
        if alert_key not in self.last_alert_time:
            return False
        
        time_since_last = (datetime.now() - self.last_alert_time[alert_key]).total_seconds()
        return time_since_last < self.config.cooldown_period
    
    def _exceeds_rate_limit(self) -> bool:
        """Check if rate limit is exceeded"""
        hour_ago = datetime.now() - timedelta(hours=1)
        recent_alerts = [a for a in self.alerts if a.timestamp > hour_ago]
        return len(recent_alerts) >= self.config.max_alerts_per_hour
    
    def _get_recommended_action(self, alert: Alert) -> str:
        """Get recommended action for alert"""
        actions = {
            AlertSeverity.INFO: "Monitor situation",
            AlertSeverity.WARNING: "Review metrics and prepare response",
            AlertSeverity.ERROR: "Investigate immediately and consider intervention",
            AlertSeverity.CRITICAL: "Take immediate action to mitigate risk"
        }
        
        base_action = actions[alert.severity]
        
        # Add specific actions based on context
        if alert.metric_name:
            if 'var' in alert.metric_name.lower():
                base_action += ". Consider reducing position sizes."
            elif 'drawdown' in alert.metric_name.lower():
                base_action += ". Review stop-loss levels."
            elif 'exposure' in alert.metric_name.lower():
                base_action += ". Rebalance portfolio if needed."
        
        return base_action
    
    def _route_alert(self, alert: Alert):
        """Route alert to configured channels"""
        for channel in self.config.enabled_channels:
            self.channel_queues[channel].append(alert)
            
            if channel == AlertChannel.LOG:
                self._send_to_log(alert)
            elif channel == AlertChannel.EMAIL:
                self._send_email(alert)
            elif channel == AlertChannel.WEBHOOK:
                self._send_webhook(alert)
            elif channel == AlertChannel.DASHBOARD:
                # Dashboard pulls from queue
                pass
    
    def _send_to_log(self, alert: Alert):
        """Send alert to log"""
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }[alert.severity]
        
        logger.log(log_level, alert.to_text())
    
    def _send_email(self, alert: Alert):
        """Send alert via email"""
        if not self.config.email_recipients:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.email_sender
            msg['To'] = ', '.join(self.config.email_recipients)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            body = alert.to_text()
            msg.attach(MIMEText(body, 'plain'))
            
            # Note: Actual email sending would require SMTP setup
            logger.info(f"Email alert queued: {alert.alert_id}")
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def _send_webhook(self, alert: Alert):
        """Send alert to webhook"""
        if not self.config.webhook_url:
            return
        
        try:
            # Note: Actual webhook sending would require requests library
            payload = alert.to_dict()
            logger.info(f"Webhook alert queued: {alert.alert_id}")
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system"):
        """
        Acknowledge an alert.
        
        Args:
            alert_id: Alert ID
            acknowledged_by: Who acknowledged
        """
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now()
                logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                return
    
    def get_active_alerts(self) -> List[Alert]:
        """Get active (unacknowledged) alerts"""
        return [a for a in self.alerts if not a.acknowledged and not a.auto_resolved]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics"""
        active_alerts = self.get_active_alerts()
        
        # Count by severity
        severity_counts = defaultdict(int)
        for alert in active_alerts:
            severity_counts[alert.severity.value] += 1
        
        # Recent alert rate
        hour_ago = datetime.now() - timedelta(hours=1)
        recent_alerts = [a for a in self.alerts if a.timestamp > hour_ago]
        
        return {
            'total_alerts': len(self.alerts),
            'active_alerts': len(active_alerts),
            'alerts_last_hour': len(recent_alerts),
            'by_severity': dict(severity_counts),
            'top_sources': self._get_top_sources(),
            'alert_rate': len(recent_alerts) / 60  # Per minute
        }
    
    def _get_top_sources(self, n: int = 5) -> List[Tuple[str, int]]:
        """Get top alert sources"""
        source_counts = defaultdict(int)
        for alert in self.alerts:
            source_counts[alert.source] += 1
        
        return sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:n]
    
    def add_callback(self, callback: Callable[[Alert], None]):
        """Add alert callback"""
        self.alert_callbacks.append(callback)


class AnomalyVisualizer:
    """
    Visualization tools for anomalies and alerts.
    
    Creates charts and dashboards for anomaly analysis.
    """
    
    def __init__(self):
        """Initialize visualizer"""
        sns.set_style("whitegrid")
        self.figure_size = (12, 6)
    
    def plot_anomaly_timeline(self,
                             data: pd.DataFrame,
                             anomalies: List[Any],
                             metric_name: str,
                             save_path: Optional[str] = None):
        """
        Plot time series with anomalies highlighted.
        
        Args:
            data: Time series data
            anomalies: List of anomaly objects
            metric_name: Name of metric to plot
            save_path: Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figure_size, 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # Main plot
        ax1.plot(data.index, data[metric_name], label=metric_name, alpha=0.7)
        
        # Mark anomalies
        anomaly_times = [a.timestamp for a in anomalies if a.metric_name == metric_name]
        anomaly_values = [a.observed_value for a in anomalies if a.metric_name == metric_name]
        
        if anomaly_times:
            ax1.scatter(anomaly_times, anomaly_values, color='red', 
                       s=50, zorder=5, label='Anomalies')
        
        ax1.set_ylabel(metric_name)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Anomaly score plot
        if anomalies:
            scores = [a.severity for a in anomalies]
            times = [a.timestamp for a in anomalies]
            ax2.bar(times, scores, color='orange', alpha=0.6)
            ax2.set_ylabel('Severity')
            ax2.set_xlabel('Time')
            ax2.set_ylim(0, 1)
        
        plt.suptitle(f'Anomaly Detection: {metric_name}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        plt.show()
    
    def plot_alert_heatmap(self,
                          alerts: List[Alert],
                          save_path: Optional[str] = None):
        """
        Create heatmap of alerts by hour and severity.
        
        Args:
            alerts: List of alerts
            save_path: Path to save figure
        """
        # Prepare data
        alert_data = []
        for alert in alerts:
            alert_data.append({
                'hour': alert.timestamp.hour,
                'day': alert.timestamp.strftime('%Y-%m-%d'),
                'severity': alert.severity.value
            })
        
        if not alert_data:
            logger.warning("No alerts to visualize")
            return
        
        df = pd.DataFrame(alert_data)
        
        # Create pivot table
        pivot = df.pivot_table(
            index='hour',
            columns='severity',
            values='day',
            aggfunc='count',
            fill_value=0
        )
        
        # Plot heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot.T, annot=True, fmt='d', cmap='YlOrRd')
        plt.title('Alert Distribution by Hour and Severity')
        plt.xlabel('Hour of Day')
        plt.ylabel('Severity')
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        plt.show()
    
    def plot_anomaly_distribution(self,
                                 anomalies: List[Any],
                                 save_path: Optional[str] = None):
        """
        Plot distribution of anomaly types and methods.
        
        Args:
            anomalies: List of anomaly objects
            save_path: Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figure_size)
        
        # Anomaly types
        types = [a.anomaly_type.value for a in anomalies]
        type_counts = pd.Series(types).value_counts()
        
        ax1.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
        ax1.set_title('Anomaly Types')
        
        # Detection methods
        methods = [a.detection_method.value for a in anomalies]
        method_counts = pd.Series(methods).value_counts()
        
        ax2.bar(method_counts.index, method_counts.values)
        ax2.set_xlabel('Detection Method')
        ax2.set_ylabel('Count')
        ax2.set_title('Detection Methods')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.suptitle('Anomaly Distribution Analysis')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        plt.show()
    
    def create_dashboard_html(self,
                            alerts: List[Alert],
                            anomalies: List[Any],
                            save_path: str = "anomaly_dashboard.html"):
        """
        Create HTML dashboard for anomalies and alerts.
        
        Args:
            alerts: List of alerts
            anomalies: List of anomalies
            save_path: Path to save HTML
        """
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Anomaly Detection Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
                .summary { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }
                .card { background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .metric { font-size: 24px; font-weight: bold; color: #2c3e50; }
                .label { color: #7f8c8d; font-size: 12px; text-transform: uppercase; }
                .alert { padding: 10px; margin: 10px 0; border-left: 4px solid; border-radius: 3px; }
                .alert-critical { border-color: #e74c3c; background: #ffebee; }
                .alert-error { border-color: #f39c12; background: #fff3e0; }
                .alert-warning { border-color: #f1c40f; background: #fffde7; }
                .alert-info { border-color: #3498db; background: #e3f2fd; }
                table { width: 100%; border-collapse: collapse; background: white; }
                th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background: #ecf0f1; }
            </style>
        </head>
        <body>
        """
        
        # Header
        html += f"""
        <div class="header">
            <h1>Anomaly Detection Dashboard</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """
        
        # Summary cards
        active_alerts = [a for a in alerts if not a.acknowledged]
        critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
        
        html += """
        <div class="summary">
            <div class="card">
                <div class="label">Total Anomalies</div>
                <div class="metric">{}</div>
            </div>
            <div class="card">
                <div class="label">Active Alerts</div>
                <div class="metric">{}</div>
            </div>
            <div class="card">
                <div class="label">Critical Alerts</div>
                <div class="metric" style="color: #e74c3c;">{}</div>
            </div>
            <div class="card">
                <div class="label">Avg Severity</div>
                <div class="metric">{:.2f}</div>
            </div>
        </div>
        """.format(
            len(anomalies),
            len(active_alerts),
            len(critical_alerts),
            np.mean([a.severity for a in anomalies]) if anomalies else 0
        )
        
        # Recent alerts
        html += "<h2>Recent Alerts</h2>"
        for alert in sorted(active_alerts, key=lambda x: x.timestamp, reverse=True)[:10]:
            html += f"""
            <div class="alert alert-{alert.severity.value}">
                <strong>{alert.title}</strong><br>
                {alert.message}<br>
                <small>{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | {alert.source}</small>
            </div>
            """
        
        # Anomaly table
        html += """
        <h2>Recent Anomalies</h2>
        <table>
            <tr>
                <th>Time</th>
                <th>Type</th>
                <th>Method</th>
                <th>Metric</th>
                <th>Severity</th>
                <th>Description</th>
            </tr>
        """
        
        for anomaly in sorted(anomalies, key=lambda x: x.timestamp, reverse=True)[:20]:
            html += f"""
            <tr>
                <td>{anomaly.timestamp.strftime('%H:%M:%S')}</td>
                <td>{anomaly.anomaly_type.value}</td>
                <td>{anomaly.detection_method.value}</td>
                <td>{anomaly.metric_name}</td>
                <td>{anomaly.severity:.2f}</td>
                <td>{anomaly.description}</td>
            </tr>
            """
        
        html += """
        </table>
        </body>
        </html>
        """
        
        with open(save_path, 'w') as f:
            f.write(html)
        
        logger.info(f"Dashboard saved to {save_path}")


class AnomalyInvestigator:
    """
    Tools for investigating detected anomalies.
    
    Provides root cause analysis and correlation detection.
    """
    
    def __init__(self):
        """Initialize investigator"""
        self.investigation_results: Dict[str, Any] = {}
    
    def investigate(self,
                   anomaly: Any,
                   historical_data: pd.DataFrame,
                   context_window: int = 100) -> Dict[str, Any]:
        """
        Investigate an anomaly.
        
        Args:
            anomaly: Anomaly to investigate
            historical_data: Historical data
            context_window: Window size for context analysis
            
        Returns:
            Investigation results
        """
        results = {
            'anomaly_id': getattr(anomaly, 'timestamp', datetime.now()).isoformat(),
            'type': anomaly.anomaly_type.value,
            'severity': anomaly.severity,
            'metric': anomaly.metric_name,
            'context': {},
            'correlations': {},
            'similar_events': [],
            'possible_causes': []
        }
        
        # Analyze context
        if anomaly.metric_name in historical_data.columns:
            metric_data = historical_data[anomaly.metric_name]
            
            # Statistical context
            results['context']['mean'] = metric_data.mean()
            results['context']['std'] = metric_data.std()
            results['context']['percentile'] = (
                (metric_data <= anomaly.observed_value).mean() * 100
            )
            
            # Recent trend
            if len(metric_data) > context_window:
                recent = metric_data.iloc[-context_window:]
                results['context']['recent_trend'] = (
                    'increasing' if recent.iloc[-1] > recent.iloc[0] else 'decreasing'
                )
                results['context']['recent_volatility'] = recent.std()
        
        # Find correlations
        if len(historical_data.columns) > 1:
            correlations = {}
            for col in historical_data.columns:
                if col != anomaly.metric_name:
                    corr = historical_data[anomaly.metric_name].corr(historical_data[col])
                    if abs(corr) > 0.5:  # Strong correlation
                        correlations[col] = corr
            
            results['correlations'] = correlations
        
        # Identify possible causes
        causes = self._identify_causes(anomaly, results)
        results['possible_causes'] = causes
        
        # Store results
        self.investigation_results[results['anomaly_id']] = results
        
        return results
    
    def _identify_causes(self, anomaly: Any, context: Dict) -> List[str]:
        """Identify possible causes for anomaly"""
        causes = []
        
        # Type-specific causes
        if anomaly.anomaly_type.value == 'outlier':
            causes.append("Sudden market event or data error")
            if context.get('correlations'):
                causes.append(f"Correlated with: {list(context['correlations'].keys())}")
        
        elif anomaly.anomaly_type.value == 'trend':
            causes.append("Regime change or systematic shift")
            if context['context'].get('recent_trend') == 'increasing':
                causes.append("Sustained upward pressure")
            else:
                causes.append("Sustained downward pressure")
        
        elif anomaly.anomaly_type.value == 'volatility':
            causes.append("Market uncertainty or news event")
            causes.append("Liquidity changes")
        
        elif anomaly.anomaly_type.value == 'microstructure':
            causes.append("Order flow imbalance")
            causes.append("Market maker behavior change")
        
        return causes
    
    def generate_report(self, investigation_id: str) -> str:
        """
        Generate investigation report.
        
        Args:
            investigation_id: Investigation ID
            
        Returns:
            Report text
        """
        if investigation_id not in self.investigation_results:
            return "Investigation not found"
        
        results = self.investigation_results[investigation_id]
        
        report = f"""
        ANOMALY INVESTIGATION REPORT
        ============================
        
        Anomaly Type: {results['type']}
        Severity: {results['severity']:.2f}
        Metric: {results['metric']}
        
        CONTEXT ANALYSIS
        ----------------
        Historical Mean: {results['context'].get('mean', 'N/A'):.4f}
        Historical Std: {results['context'].get('std', 'N/A'):.4f}
        Percentile: {results['context'].get('percentile', 'N/A'):.1f}%
        Recent Trend: {results['context'].get('recent_trend', 'N/A')}
        
        CORRELATIONS
        ------------
        """
        
        for metric, corr in results['correlations'].items():
            report += f"{metric}: {corr:.3f}\n"
        
        report += """
        
        POSSIBLE CAUSES
        ---------------
        """
        
        for cause in results['possible_causes']:
            report += f"• {cause}\n"
        
        return report


def demonstrate_alert_system():
    """Demonstrate alert and visualization system"""
    print("Alert System Demonstration")
    print("=" * 60)
    
    # Create alert generator
    config = AlertConfig()
    generator = AlertGenerator(config)
    
    # Generate sample alerts
    print("\nGenerating sample alerts...")
    
    alerts = []
    
    # Critical alert
    alert1 = generator.generate_alert(
        title="VaR Limit Breach",
        message="95% VaR exceeded risk limit",
        severity=AlertSeverity.CRITICAL,
        source="risk_monitor",
        metric_name="var_95",
        observed_value=0.08,
        threshold_value=0.05
    )
    if alert1:
        alerts.append(alert1)
    
    # Warning alert
    alert2 = generator.generate_alert(
        title="Unusual Volume Pattern",
        message="Trading volume 3x above average",
        severity=AlertSeverity.WARNING,
        source="market_monitor",
        metric_name="volume",
        observed_value=1000000,
        threshold_value=300000
    )
    if alert2:
        alerts.append(alert2)
    
    # Info alert
    alert3 = generator.generate_alert(
        title="Model Update",
        message="New model deployed successfully",
        severity=AlertSeverity.INFO,
        source="model_manager"
    )
    if alert3:
        alerts.append(alert3)
    
    # Display alerts
    print(f"\nGenerated {len(alerts)} alerts")
    for alert in alerts:
        print(f"\n{alert.to_text()}")
    
    # Alert summary
    summary = generator.get_alert_summary()
    print("\nAlert Summary:")
    print(f"  Total: {summary['total_alerts']}")
    print(f"  Active: {summary['active_alerts']}")
    print(f"  Last Hour: {summary['alerts_last_hour']}")
    
    # Create visualizer
    print("\n" + "=" * 40)
    print("Creating visualizations...")
    
    visualizer = AnomalyVisualizer()
    
    # Create sample anomalies for visualization
    from anomaly_detector import Anomaly, AnomalyType, DetectionMethod
    
    anomalies = [
        Anomaly(
            timestamp=datetime.now() - timedelta(minutes=30),
            anomaly_type=AnomalyType.OUTLIER,
            detection_method=DetectionMethod.ISOLATION_FOREST,
            severity=0.7,
            confidence=0.8,
            metric_name="returns",
            observed_value=0.05,
            description="Large return spike"
        ),
        Anomaly(
            timestamp=datetime.now() - timedelta(minutes=15),
            anomaly_type=AnomalyType.TREND,
            detection_method=DetectionMethod.CUSUM,
            severity=0.5,
            confidence=0.7,
            metric_name="volume",
            observed_value=500000,
            description="Volume trend change"
        )
    ]
    
    # Create HTML dashboard
    visualizer.create_dashboard_html(alerts, anomalies, "anomaly_dashboard.html")
    print("Dashboard saved to anomaly_dashboard.html")
    
    # Investigation
    print("\n" + "=" * 40)
    print("Investigating anomalies...")
    
    investigator = AnomalyInvestigator()
    
    # Create sample data for investigation
    sample_data = pd.DataFrame({
        'returns': np.random.normal(0, 0.01, 100),
        'volume': np.random.lognormal(10, 1, 100),
        'volatility': np.random.gamma(2, 0.01, 100)
    })
    
    for anomaly in anomalies[:1]:  # Investigate first anomaly
        results = investigator.investigate(anomaly, sample_data)
        report = investigator.generate_report(results['anomaly_id'])
        print(report)
    
    print("\n✅ Alert System operational!")


if __name__ == "__main__":
    demonstrate_alert_system()