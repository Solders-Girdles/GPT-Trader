"""
Performance Metrics Collection System
Phase 3, Week 1: MON-010
Comprehensive system for collecting and analyzing model performance metrics
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from collections import defaultdict, deque
import threading
import time
import psutil
import tracemalloc

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics to collect"""
    # Model Performance
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    
    # Regression Metrics
    MSE = "mean_squared_error"
    MAE = "mean_absolute_error"
    RMSE = "root_mean_squared_error"
    R2 = "r2_score"
    
    # Trading Metrics
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    
    # System Metrics
    LATENCY = "latency_ms"
    THROUGHPUT = "throughput_rps"
    CPU_USAGE = "cpu_usage_percent"
    MEMORY_USAGE = "memory_usage_mb"
    
    # Data Quality
    MISSING_DATA = "missing_data_ratio"
    OUTLIERS = "outlier_ratio"
    DATA_FRESHNESS = "data_freshness_seconds"


@dataclass
class MetricSnapshot:
    """Single metric measurement"""
    metric_type: MetricType
    value: float
    timestamp: datetime
    model_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'metric_type': self.metric_type.value,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'model_id': self.model_id,
            'metadata': self.metadata
        }


@dataclass
class PerformanceReport:
    """Aggregated performance report"""
    model_id: str
    period_start: datetime
    period_end: datetime
    
    # Aggregated metrics
    metrics_summary: Dict[str, Dict[str, float]]  # metric -> {mean, std, min, max}
    
    # Trends
    trends: Dict[str, float]  # metric -> trend slope
    
    # Comparisons
    baseline_comparison: Dict[str, float]  # metric -> % change from baseline
    
    # Alerts
    anomalies: List[str]
    warnings: List[str]
    
    # Sample counts
    n_predictions: int
    n_errors: int
    
    def to_dict(self) -> Dict:
        return {
            'model_id': self.model_id,
            'period': {
                'start': self.period_start.isoformat(),
                'end': self.period_end.isoformat()
            },
            'metrics_summary': self.metrics_summary,
            'trends': self.trends,
            'baseline_comparison': self.baseline_comparison,
            'anomalies': self.anomalies,
            'warnings': self.warnings,
            'statistics': {
                'n_predictions': self.n_predictions,
                'n_errors': self.n_errors,
                'error_rate': self.n_errors / max(1, self.n_predictions)
            }
        }


class PerformanceMetricsCollector:
    """
    Comprehensive system for collecting and analyzing performance metrics.
    
    Features:
    - Real-time metric collection
    - Multiple metric types (model, system, data quality)
    - Trend analysis
    - Anomaly detection
    - Baseline comparisons
    - Automatic aggregation
    """
    
    def __init__(self,
                 buffer_size: int = 10000,
                 aggregation_interval: int = 60,
                 anomaly_threshold: float = 3.0):
        """
        Initialize metrics collector.
        
        Args:
            buffer_size: Size of metric buffer per model
            aggregation_interval: Seconds between aggregations
            anomaly_threshold: Z-score threshold for anomaly detection
        """
        self.buffer_size = buffer_size
        self.aggregation_interval = aggregation_interval
        self.anomaly_threshold = anomaly_threshold
        
        # Metric storage
        self.metrics_buffer: Dict[str, Dict[MetricType, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=buffer_size))
        )
        self.aggregated_metrics: Dict[str, List[PerformanceReport]] = defaultdict(list)
        
        # Baselines
        self.baselines: Dict[str, Dict[MetricType, float]] = {}
        
        # System monitoring
        self.system_monitor_thread = None
        self.monitoring_active = False
        
        # Performance tracking
        self.start_time = datetime.now()
        self.total_metrics_collected = 0
        
        logger.info(f"PerformanceMetricsCollector initialized with {buffer_size} buffer size")
    
    def start_monitoring(self) -> None:
        """Start system monitoring thread"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.system_monitor_thread = threading.Thread(
                target=self._system_monitoring_loop,
                daemon=True
            )
            self.system_monitor_thread.start()
            logger.info("System monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop system monitoring"""
        self.monitoring_active = False
        if self.system_monitor_thread:
            self.system_monitor_thread.join(timeout=5)
        logger.info("System monitoring stopped")
    
    def collect_prediction_metrics(self,
                                  model_id: str,
                                  predictions: np.ndarray,
                                  actuals: np.ndarray,
                                  probabilities: Optional[np.ndarray] = None,
                                  metadata: Optional[Dict] = None) -> None:
        """
        Collect metrics from model predictions.
        
        Args:
            model_id: Model identifier
            predictions: Predicted values
            actuals: Actual values
            probabilities: Prediction probabilities (for classification)
            metadata: Additional metadata
        """
        timestamp = datetime.now()
        
        # Calculate metrics based on task type
        if len(np.unique(actuals)) <= 10:  # Classification
            self._collect_classification_metrics(
                model_id, predictions, actuals, probabilities, timestamp, metadata
            )
        else:  # Regression
            self._collect_regression_metrics(
                model_id, predictions, actuals, timestamp, metadata
            )
        
        self.total_metrics_collected += 1
    
    def _collect_classification_metrics(self,
                                       model_id: str,
                                       predictions: np.ndarray,
                                       actuals: np.ndarray,
                                       probabilities: Optional[np.ndarray],
                                       timestamp: datetime,
                                       metadata: Optional[Dict]) -> None:
        """Collect classification metrics"""
        # Accuracy
        accuracy = np.mean(predictions == actuals)
        self._add_metric(model_id, MetricType.ACCURACY, accuracy, timestamp, metadata)
        
        # For binary classification
        if len(np.unique(actuals)) == 2:
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            precision = precision_score(actuals, predictions, zero_division=0)
            recall = recall_score(actuals, predictions, zero_division=0)
            f1 = f1_score(actuals, predictions, zero_division=0)
            
            self._add_metric(model_id, MetricType.PRECISION, precision, timestamp, metadata)
            self._add_metric(model_id, MetricType.RECALL, recall, timestamp, metadata)
            self._add_metric(model_id, MetricType.F1_SCORE, f1, timestamp, metadata)
            
            # ROC-AUC if probabilities available
            if probabilities is not None and len(probabilities.shape) > 1:
                from sklearn.metrics import roc_auc_score
                try:
                    auc = roc_auc_score(actuals, probabilities[:, 1])
                    self._add_metric(model_id, MetricType.ROC_AUC, auc, timestamp, metadata)
                except:
                    pass
    
    def _collect_regression_metrics(self,
                                   model_id: str,
                                   predictions: np.ndarray,
                                   actuals: np.ndarray,
                                   timestamp: datetime,
                                   metadata: Optional[Dict]) -> None:
        """Collect regression metrics"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mse)
        
        self._add_metric(model_id, MetricType.MSE, mse, timestamp, metadata)
        self._add_metric(model_id, MetricType.MAE, mae, timestamp, metadata)
        self._add_metric(model_id, MetricType.RMSE, rmse, timestamp, metadata)
        
        # R2 score
        if len(actuals) > 1 and np.var(actuals) > 0:
            r2 = r2_score(actuals, predictions)
            self._add_metric(model_id, MetricType.R2, r2, timestamp, metadata)
    
    def collect_trading_metrics(self,
                               model_id: str,
                               returns: np.ndarray,
                               positions: Optional[np.ndarray] = None) -> None:
        """
        Collect trading-specific metrics.
        
        Args:
            model_id: Model identifier
            returns: Array of returns
            positions: Array of positions (1, 0, -1)
        """
        timestamp = datetime.now()
        
        if len(returns) < 2:
            return
        
        # Sharpe Ratio (annualized)
        if np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
            self._add_metric(model_id, MetricType.SHARPE_RATIO, sharpe, timestamp)
        
        # Maximum Drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_dd = np.min(drawdown)
        self._add_metric(model_id, MetricType.MAX_DRAWDOWN, abs(max_dd), timestamp)
        
        # Win Rate
        if positions is not None:
            winning_trades = np.sum(returns[positions != 0] > 0)
            total_trades = np.sum(positions != 0)
            if total_trades > 0:
                win_rate = winning_trades / total_trades
                self._add_metric(model_id, MetricType.WIN_RATE, win_rate, timestamp)
        
        # Profit Factor
        gains = returns[returns > 0]
        losses = abs(returns[returns < 0])
        if len(losses) > 0 and np.sum(losses) > 0:
            profit_factor = np.sum(gains) / np.sum(losses)
            self._add_metric(model_id, MetricType.PROFIT_FACTOR, profit_factor, timestamp)
    
    def collect_system_metrics(self,
                              model_id: str,
                              latency_ms: Optional[float] = None,
                              throughput_rps: Optional[float] = None) -> None:
        """
        Collect system performance metrics.
        
        Args:
            model_id: Model identifier
            latency_ms: Response latency in milliseconds
            throughput_rps: Throughput in requests per second
        """
        timestamp = datetime.now()
        
        if latency_ms is not None:
            self._add_metric(model_id, MetricType.LATENCY, latency_ms, timestamp)
        
        if throughput_rps is not None:
            self._add_metric(model_id, MetricType.THROUGHPUT, throughput_rps, timestamp)
        
        # Collect current system stats
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        
        self._add_metric(model_id, MetricType.CPU_USAGE, cpu_percent, timestamp)
        self._add_metric(model_id, MetricType.MEMORY_USAGE, memory_mb, timestamp)
    
    def collect_data_quality_metrics(self,
                                    model_id: str,
                                    data: pd.DataFrame) -> None:
        """
        Collect data quality metrics.
        
        Args:
            model_id: Model identifier
            data: Input data DataFrame
        """
        timestamp = datetime.now()
        
        # Missing data ratio
        missing_ratio = data.isnull().sum().sum() / data.size
        self._add_metric(model_id, MetricType.MISSING_DATA, missing_ratio, timestamp)
        
        # Outlier detection (simple z-score method)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            from scipy import stats
            z_scores = np.abs(stats.zscore(data[numeric_cols].fillna(0)))
            outlier_ratio = (z_scores > 3).sum().sum() / z_scores.size
            self._add_metric(model_id, MetricType.OUTLIERS, outlier_ratio, timestamp)
    
    def _add_metric(self,
                   model_id: str,
                   metric_type: MetricType,
                   value: float,
                   timestamp: datetime,
                   metadata: Optional[Dict] = None) -> None:
        """Add metric to buffer"""
        snapshot = MetricSnapshot(
            metric_type=metric_type,
            value=value,
            timestamp=timestamp,
            model_id=model_id,
            metadata=metadata or {}
        )
        
        self.metrics_buffer[model_id][metric_type].append(snapshot)
    
    def set_baseline(self,
                    model_id: str,
                    baseline_metrics: Dict[str, float]) -> None:
        """
        Set baseline metrics for comparison.
        
        Args:
            model_id: Model identifier
            baseline_metrics: Dictionary of metric name to baseline value
        """
        self.baselines[model_id] = {}
        for metric_name, value in baseline_metrics.items():
            try:
                metric_type = MetricType(metric_name)
                self.baselines[model_id][metric_type] = value
            except ValueError:
                logger.warning(f"Unknown metric type: {metric_name}")
        
        logger.info(f"Set {len(self.baselines[model_id])} baseline metrics for {model_id}")
    
    def get_current_metrics(self,
                           model_id: str,
                           window_minutes: int = 5) -> Dict[str, Any]:
        """
        Get current metrics for a model.
        
        Args:
            model_id: Model identifier
            window_minutes: Time window for metrics
            
        Returns:
            Dictionary of current metrics
        """
        if model_id not in self.metrics_buffer:
            return {'error': f'No metrics for model {model_id}'}
        
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        current_metrics = {}
        
        for metric_type, snapshots in self.metrics_buffer[model_id].items():
            recent = [s for s in snapshots if s.timestamp > cutoff_time]
            
            if recent:
                values = [s.value for s in recent]
                current_metrics[metric_type.value] = {
                    'current': values[-1],
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
                
                # Add baseline comparison if available
                if model_id in self.baselines and metric_type in self.baselines[model_id]:
                    baseline = self.baselines[model_id][metric_type]
                    current_metrics[metric_type.value]['baseline'] = baseline
                    current_metrics[metric_type.value]['change_from_baseline'] = (
                        (values[-1] - baseline) / baseline if baseline != 0 else 0
                    )
        
        return current_metrics
    
    def aggregate_metrics(self,
                         model_id: str,
                         period_hours: int = 1) -> PerformanceReport:
        """
        Aggregate metrics over a time period.
        
        Args:
            model_id: Model identifier
            period_hours: Aggregation period in hours
            
        Returns:
            Aggregated performance report
        """
        period_end = datetime.now()
        period_start = period_end - timedelta(hours=period_hours)
        
        metrics_summary = {}
        trends = {}
        anomalies = []
        warnings = []
        n_predictions = 0
        n_errors = 0
        
        for metric_type, snapshots in self.metrics_buffer[model_id].items():
            # Filter to period
            period_snapshots = [
                s for s in snapshots 
                if period_start <= s.timestamp <= period_end
            ]
            
            if not period_snapshots:
                continue
            
            values = [s.value for s in period_snapshots]
            timestamps = [s.timestamp for s in period_snapshots]
            
            # Calculate summary statistics
            metrics_summary[metric_type.value] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
            
            # Calculate trend
            if len(values) >= 3:
                x = np.arange(len(values))
                slope, _ = np.polyfit(x, values, 1)
                trends[metric_type.value] = slope
            
            # Detect anomalies
            if len(values) >= 10:
                z_scores = np.abs(stats.zscore(values))
                anomaly_indices = np.where(z_scores > self.anomaly_threshold)[0]
                if len(anomaly_indices) > 0:
                    anomalies.append(
                        f"{metric_type.value}: {len(anomaly_indices)} anomalies detected"
                    )
            
            # Count predictions and errors
            if metric_type == MetricType.ACCURACY:
                n_predictions = len(values)
                n_errors = int(np.sum([1 - v for v in values]))
        
        # Calculate baseline comparison
        baseline_comparison = {}
        if model_id in self.baselines:
            for metric_type, baseline in self.baselines[model_id].items():
                if metric_type.value in metrics_summary:
                    current = metrics_summary[metric_type.value]['mean']
                    change = (current - baseline) / baseline if baseline != 0 else 0
                    baseline_comparison[metric_type.value] = change
                    
                    # Add warning if significant degradation
                    if change < -0.05:  # 5% degradation
                        warnings.append(
                            f"{metric_type.value}: {change:.1%} below baseline"
                        )
        
        report = PerformanceReport(
            model_id=model_id,
            period_start=period_start,
            period_end=period_end,
            metrics_summary=metrics_summary,
            trends=trends,
            baseline_comparison=baseline_comparison,
            anomalies=anomalies,
            warnings=warnings,
            n_predictions=n_predictions,
            n_errors=n_errors
        )
        
        self.aggregated_metrics[model_id].append(report)
        
        return report
    
    def detect_anomalies(self,
                        model_id: str,
                        metric_type: MetricType,
                        window_size: int = 100) -> List[Tuple[datetime, float]]:
        """
        Detect anomalies in a specific metric.
        
        Args:
            model_id: Model identifier
            metric_type: Type of metric
            window_size: Window size for anomaly detection
            
        Returns:
            List of (timestamp, value) tuples for anomalies
        """
        if model_id not in self.metrics_buffer:
            return []
        
        if metric_type not in self.metrics_buffer[model_id]:
            return []
        
        snapshots = list(self.metrics_buffer[model_id][metric_type])
        
        if len(snapshots) < window_size:
            return []
        
        # Use recent window for baseline
        recent = snapshots[-window_size:]
        values = [s.value for s in recent]
        
        mean = np.mean(values)
        std = np.std(values)
        
        anomalies = []
        for snapshot in recent:
            z_score = abs((snapshot.value - mean) / std) if std > 0 else 0
            if z_score > self.anomaly_threshold:
                anomalies.append((snapshot.timestamp, snapshot.value))
        
        return anomalies
    
    def _system_monitoring_loop(self) -> None:
        """Background thread for system monitoring"""
        while self.monitoring_active:
            try:
                # Collect system metrics for all active models
                for model_id in list(self.metrics_buffer.keys()):
                    self.collect_system_metrics(model_id)
                
                # Sleep for interval
                time.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
    
    def export_metrics(self,
                      model_id: str,
                      filepath: str,
                      format: str = 'json') -> None:
        """
        Export metrics to file.
        
        Args:
            model_id: Model identifier
            filepath: Output file path
            format: 'json' or 'csv'
        """
        if format == 'json':
            # Export as JSON
            data = {
                'model_id': model_id,
                'export_time': datetime.now().isoformat(),
                'current_metrics': self.get_current_metrics(model_id),
                'reports': [r.to_dict() for r in self.aggregated_metrics.get(model_id, [])]
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        elif format == 'csv':
            # Export as CSV
            rows = []
            for metric_type, snapshots in self.metrics_buffer[model_id].items():
                for snapshot in snapshots:
                    rows.append({
                        'timestamp': snapshot.timestamp,
                        'metric': metric_type.value,
                        'value': snapshot.value,
                        'model_id': model_id
                    })
            
            df = pd.DataFrame(rows)
            df.to_csv(filepath, index=False)
        
        logger.info(f"Exported metrics for {model_id} to {filepath}")
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for all models"""
        summary = {
            'collector_uptime': (datetime.now() - self.start_time).total_seconds(),
            'total_metrics_collected': self.total_metrics_collected,
            'models_monitored': len(self.metrics_buffer),
            'models': {}
        }
        
        for model_id in self.metrics_buffer:
            total_snapshots = sum(
                len(snapshots) 
                for snapshots in self.metrics_buffer[model_id].values()
            )
            
            summary['models'][model_id] = {
                'total_snapshots': total_snapshots,
                'metrics_tracked': len(self.metrics_buffer[model_id]),
                'reports_generated': len(self.aggregated_metrics.get(model_id, []))
            }
        
        return summary


def create_metrics_collector() -> PerformanceMetricsCollector:
    """Create and initialize a metrics collector"""
    collector = PerformanceMetricsCollector()
    collector.start_monitoring()
    return collector


if __name__ == "__main__":
    # Example usage
    collector = create_metrics_collector()
    
    # Simulate collecting metrics
    np.random.seed(42)
    model_id = "test_model"
    
    # Set baseline
    collector.set_baseline(model_id, {
        'accuracy': 0.75,
        'latency_ms': 50
    })
    
    # Collect metrics over time
    for i in range(100):
        # Model predictions
        n = 100
        accuracy = 0.75 + np.random.normal(0, 0.02)
        predictions = np.random.choice([0, 1], n)
        actuals = predictions.copy()
        wrong = np.random.choice(n, int((1-accuracy)*n), replace=False)
        actuals[wrong] = 1 - actuals[wrong]
        
        collector.collect_prediction_metrics(model_id, predictions, actuals)
        
        # System metrics
        collector.collect_system_metrics(
            model_id,
            latency_ms=50 + np.random.normal(0, 5),
            throughput_rps=100 + np.random.normal(0, 10)
        )
        
        # Trading metrics (every 10 iterations)
        if i % 10 == 0:
            returns = np.random.normal(0.001, 0.02, 20)
            collector.collect_trading_metrics(model_id, returns)
        
        time.sleep(0.1)
    
    # Get current metrics
    current = collector.get_current_metrics(model_id)
    print("\nCurrent Metrics:")
    for metric, values in current.items():
        if isinstance(values, dict):
            print(f"  {metric}: {values.get('current', 0):.3f} "
                  f"(mean={values.get('mean', 0):.3f})")
    
    # Generate report
    report = collector.aggregate_metrics(model_id)
    print(f"\nPerformance Report:")
    print(f"  Period: {report.period_start} to {report.period_end}")
    print(f"  Predictions: {report.n_predictions}")
    print(f"  Errors: {report.n_errors}")
    print(f"  Anomalies: {report.anomalies}")
    print(f"  Warnings: {report.warnings}")
    
    # Export metrics
    collector.export_metrics(model_id, "metrics_export.json")
    
    # Get summary
    summary = collector.get_summary_statistics()
    print(f"\nCollector Summary:")
    print(f"  Uptime: {summary['collector_uptime']:.1f} seconds")
    print(f"  Total metrics: {summary['total_metrics_collected']}")
    print(f"  Models monitored: {summary['models_monitored']}")
    
    # Stop monitoring
    collector.stop_monitoring()