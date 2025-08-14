"""
Model Degradation Monitoring System
Phase 2.5 - Day 7

Tracks model performance degradation and triggers retraining when needed.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from collections import deque
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger(__name__)


class DegradationStatus(Enum):
    """Model degradation status levels"""
    HEALTHY = "healthy"           # Performance within expected range
    WARNING = "warning"           # Slight degradation detected
    DEGRADED = "degraded"         # Significant degradation
    CRITICAL = "critical"         # Severe degradation, immediate action needed
    RETRAINING = "retraining"     # Model is being retrained


class DegradationType(Enum):
    """Types of degradation"""
    GRADUAL = "gradual"           # Slow decline over time
    SUDDEN = "sudden"             # Rapid performance drop
    CYCLICAL = "cyclical"         # Periodic performance variations
    CONCEPT_DRIFT = "concept_drift"  # Underlying data distribution change
    DATA_QUALITY = "data_quality"    # Input data quality issues


@dataclass
class DegradationMetrics:
    """Metrics for tracking degradation"""
    timestamp: datetime
    model_id: str
    
    # Performance metrics
    current_accuracy: float
    baseline_accuracy: float
    rolling_accuracy: float  # Moving average
    
    # Degradation indicators
    accuracy_drop: float  # From baseline
    accuracy_trend: float  # Slope of recent performance
    volatility: float  # Standard deviation of recent performance
    
    # Statistical tests
    drift_p_value: float  # Statistical significance of drift
    is_significant: bool  # Statistically significant degradation
    
    # Confidence
    prediction_confidence: float  # Average model confidence
    confidence_trend: float  # Trend in confidence
    
    # Data quality
    missing_features_ratio: float
    outlier_ratio: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'model_id': self.model_id,
            'current_accuracy': self.current_accuracy,
            'baseline_accuracy': self.baseline_accuracy,
            'rolling_accuracy': self.rolling_accuracy,
            'accuracy_drop': self.accuracy_drop,
            'accuracy_trend': self.accuracy_trend,
            'volatility': self.volatility,
            'drift_p_value': self.drift_p_value,
            'is_significant': self.is_significant,
            'prediction_confidence': self.prediction_confidence,
            'confidence_trend': self.confidence_trend,
            'missing_features_ratio': self.missing_features_ratio,
            'outlier_ratio': self.outlier_ratio
        }


@dataclass
class RetrainingTrigger:
    """Configuration for retraining triggers"""
    # Performance thresholds
    accuracy_drop_threshold: float = 0.05  # 5% drop
    min_accuracy_threshold: float = 0.55   # Absolute minimum
    
    # Time-based triggers
    max_days_without_retrain: int = 30
    min_days_between_retrain: int = 7
    
    # Statistical triggers
    drift_significance_level: float = 0.05  # p-value threshold
    volatility_threshold: float = 0.1       # Performance volatility
    
    # Data quality triggers
    max_missing_features: float = 0.1  # 10% missing
    max_outlier_ratio: float = 0.05    # 5% outliers
    
    # Trend triggers
    negative_trend_periods: int = 5  # Consecutive periods of decline
    
    # Automatic actions
    auto_retrain: bool = True
    auto_rollback: bool = True  # Rollback to previous model if new one performs worse
    
    # Notification settings
    alert_on_degradation: bool = True
    alert_on_retrain: bool = True


class ModelDegradationMonitor:
    """
    Monitors model performance degradation and triggers retraining.
    
    Features:
    - Real-time performance tracking
    - Multiple degradation detection methods
    - Automatic retraining triggers
    - Performance history analysis
    - Statistical significance testing
    """
    
    def __init__(self, 
                 baseline_window: int = 30,
                 monitoring_window: int = 7,
                 triggers: Optional[RetrainingTrigger] = None):
        """
        Initialize degradation monitor.
        
        Args:
            baseline_window: Days to establish baseline performance
            monitoring_window: Days for rolling performance calculation
            triggers: Retraining trigger configuration
        """
        self.baseline_window = baseline_window
        self.monitoring_window = monitoring_window
        self.triggers = triggers or RetrainingTrigger()
        
        # Performance history
        self.performance_history: Dict[str, deque] = {}
        self.prediction_history: Dict[str, List] = {}
        self.confidence_history: Dict[str, deque] = {}
        
        # Model metadata
        self.model_baselines: Dict[str, float] = {}
        self.last_retrain_date: Dict[str, datetime] = {}
        self.degradation_status: Dict[str, DegradationStatus] = {}
        
        # Alerts
        self.alert_history: List[Dict] = []
        
        logger.info(f"ModelDegradationMonitor initialized with {baseline_window} day baseline")
    
    def update_performance(self,
                          model_id: str,
                          y_true: np.ndarray,
                          y_pred: np.ndarray,
                          confidence: Optional[np.ndarray] = None,
                          features: Optional[pd.DataFrame] = None):
        """
        Update model performance metrics.
        
        Args:
            model_id: Model identifier
            y_true: True labels
            y_pred: Predicted labels
            confidence: Prediction confidence scores
            features: Input features for quality analysis
        """
        # Calculate performance
        accuracy = accuracy_score(y_true, y_pred)
        
        # Initialize history if needed
        if model_id not in self.performance_history:
            self.performance_history[model_id] = deque(maxlen=self.baseline_window)
            self.confidence_history[model_id] = deque(maxlen=self.baseline_window)
            self.prediction_history[model_id] = []
        
        # Update history
        self.performance_history[model_id].append({
            'timestamp': datetime.now(),
            'accuracy': accuracy,
            'n_samples': len(y_true)
        })
        
        # Update confidence if provided
        if confidence is not None:
            avg_confidence = np.mean(confidence)
            self.confidence_history[model_id].append(avg_confidence)
        
        # Store predictions for analysis
        self.prediction_history[model_id].append({
            'y_true': y_true,
            'y_pred': y_pred,
            'timestamp': datetime.now()
        })
        
        # Check for degradation
        metrics = self._calculate_degradation_metrics(model_id, features)
        
        if metrics:
            status = self._determine_status(metrics)
            self.degradation_status[model_id] = status
            
            # Check retraining triggers
            if self._should_retrain(model_id, metrics):
                self._trigger_retraining(model_id, metrics)
            
            logger.info(f"Model {model_id}: Accuracy={accuracy:.3f}, Status={status.value}")
    
    def _calculate_degradation_metrics(self,
                                      model_id: str,
                                      features: Optional[pd.DataFrame] = None) -> Optional[DegradationMetrics]:
        """Calculate degradation metrics"""
        
        history = self.performance_history.get(model_id)
        if not history or len(history) < 2:
            return None
        
        # Current performance
        current_perf = history[-1]['accuracy']
        
        # Baseline (first N days or explicit baseline)
        if model_id not in self.model_baselines:
            baseline_perfs = [h['accuracy'] for h in list(history)[:self.baseline_window]]
            if baseline_perfs:
                self.model_baselines[model_id] = np.mean(baseline_perfs)
        
        baseline = self.model_baselines.get(model_id, current_perf)
        
        # Rolling average
        recent_perfs = [h['accuracy'] for h in list(history)[-self.monitoring_window:]]
        rolling_avg = np.mean(recent_perfs) if recent_perfs else current_perf
        
        # Performance trend (linear regression slope)
        if len(recent_perfs) >= 3:
            x = np.arange(len(recent_perfs))
            slope, _ = np.polyfit(x, recent_perfs, 1)
            trend = slope
        else:
            trend = 0
        
        # Volatility
        volatility = np.std(recent_perfs) if len(recent_perfs) > 1 else 0
        
        # Statistical test for drift
        if len(history) >= self.baseline_window:
            baseline_perfs = [h['accuracy'] for h in list(history)[:self.baseline_window]]
            recent_perfs = [h['accuracy'] for h in list(history)[-self.monitoring_window:]]
            
            if baseline_perfs and recent_perfs:
                _, p_value = stats.ttest_ind(baseline_perfs, recent_perfs)
            else:
                p_value = 1.0
        else:
            p_value = 1.0
        
        # Confidence metrics
        confidence_history = self.confidence_history.get(model_id, [])
        if confidence_history:
            current_confidence = confidence_history[-1] if confidence_history else 0.5
            
            if len(confidence_history) >= 3:
                x = np.arange(len(confidence_history))
                conf_trend, _ = np.polyfit(x, list(confidence_history), 1)
            else:
                conf_trend = 0
        else:
            current_confidence = 0.5
            conf_trend = 0
        
        # Data quality metrics
        missing_ratio = 0
        outlier_ratio = 0
        
        if features is not None:
            missing_ratio = features.isnull().sum().sum() / features.size
            
            # Simple outlier detection (values beyond 3 std)
            numeric_features = features.select_dtypes(include=[np.number])
            if not numeric_features.empty:
                z_scores = np.abs(stats.zscore(numeric_features.fillna(0)))
                outlier_ratio = (z_scores > 3).sum().sum() / numeric_features.size
        
        return DegradationMetrics(
            timestamp=datetime.now(),
            model_id=model_id,
            current_accuracy=current_perf,
            baseline_accuracy=baseline,
            rolling_accuracy=rolling_avg,
            accuracy_drop=baseline - current_perf,
            accuracy_trend=trend,
            volatility=volatility,
            drift_p_value=p_value,
            is_significant=p_value < self.triggers.drift_significance_level,
            prediction_confidence=current_confidence,
            confidence_trend=conf_trend,
            missing_features_ratio=missing_ratio,
            outlier_ratio=outlier_ratio
        )
    
    def _determine_status(self, metrics: DegradationMetrics) -> DegradationStatus:
        """Determine degradation status from metrics"""
        
        # Critical: Below minimum threshold
        if metrics.current_accuracy < self.triggers.min_accuracy_threshold:
            return DegradationStatus.CRITICAL
        
        # Degraded: Significant drop or high volatility
        if (metrics.accuracy_drop > self.triggers.accuracy_drop_threshold or
            metrics.volatility > self.triggers.volatility_threshold or
            metrics.is_significant):
            return DegradationStatus.DEGRADED
        
        # Warning: Negative trend or moderate drop
        if (metrics.accuracy_trend < -0.01 or  # Declining trend
            metrics.accuracy_drop > self.triggers.accuracy_drop_threshold * 0.5):
            return DegradationStatus.WARNING
        
        return DegradationStatus.HEALTHY
    
    def _should_retrain(self, model_id: str, metrics: DegradationMetrics) -> bool:
        """Check if model should be retrained"""
        
        if not self.triggers.auto_retrain:
            return False
        
        # Check minimum time between retrains
        last_retrain = self.last_retrain_date.get(model_id)
        if last_retrain:
            days_since_retrain = (datetime.now() - last_retrain).days
            if days_since_retrain < self.triggers.min_days_between_retrain:
                return False
        
        # Check triggers
        triggers = [
            # Performance triggers
            metrics.accuracy_drop > self.triggers.accuracy_drop_threshold,
            metrics.current_accuracy < self.triggers.min_accuracy_threshold,
            metrics.is_significant,  # Statistically significant drift
            
            # Trend triggers
            metrics.accuracy_trend < -0.02,  # Strong negative trend
            
            # Data quality triggers
            metrics.missing_features_ratio > self.triggers.max_missing_features,
            metrics.outlier_ratio > self.triggers.max_outlier_ratio,
            
            # Time trigger
            last_retrain and (datetime.now() - last_retrain).days > self.triggers.max_days_without_retrain
        ]
        
        return any(triggers)
    
    def _trigger_retraining(self, model_id: str, metrics: DegradationMetrics):
        """Trigger model retraining"""
        
        logger.warning(f"Triggering retraining for model {model_id}")
        
        # Update status
        self.degradation_status[model_id] = DegradationStatus.RETRAINING
        self.last_retrain_date[model_id] = datetime.now()
        
        # Create alert
        alert = {
            'timestamp': datetime.now(),
            'model_id': model_id,
            'type': 'retraining_triggered',
            'metrics': metrics.to_dict(),
            'reason': self._get_trigger_reason(metrics)
        }
        
        self.alert_history.append(alert)
        
        if self.triggers.alert_on_retrain:
            self._send_alert(alert)
        
        # In production, this would trigger actual retraining
        logger.info(f"Retraining triggered for {model_id}: {alert['reason']}")
    
    def _get_trigger_reason(self, metrics: DegradationMetrics) -> str:
        """Get human-readable trigger reason"""
        reasons = []
        
        if metrics.accuracy_drop > self.triggers.accuracy_drop_threshold:
            reasons.append(f"Accuracy drop: {metrics.accuracy_drop:.2%}")
        
        if metrics.current_accuracy < self.triggers.min_accuracy_threshold:
            reasons.append(f"Below minimum accuracy: {metrics.current_accuracy:.2%}")
        
        if metrics.is_significant:
            reasons.append(f"Statistically significant drift (p={metrics.drift_p_value:.3f})")
        
        if metrics.accuracy_trend < -0.02:
            reasons.append(f"Strong negative trend: {metrics.accuracy_trend:.3f}")
        
        if metrics.missing_features_ratio > self.triggers.max_missing_features:
            reasons.append(f"High missing features: {metrics.missing_features_ratio:.1%}")
        
        if metrics.outlier_ratio > self.triggers.max_outlier_ratio:
            reasons.append(f"High outlier ratio: {metrics.outlier_ratio:.1%}")
        
        return "; ".join(reasons) if reasons else "Unknown"
    
    def _send_alert(self, alert: Dict):
        """Send alert (placeholder for actual alerting)"""
        logger.warning(f"ALERT: {alert['type']} for {alert['model_id']}: {alert.get('reason', '')}")
    
    def get_degradation_report(self, model_id: str) -> Dict[str, Any]:
        """Get comprehensive degradation report for a model"""
        
        metrics = self._calculate_degradation_metrics(model_id)
        if not metrics:
            return {'status': 'insufficient_data'}
        
        history = self.performance_history.get(model_id, [])
        
        # Calculate degradation type
        degradation_type = self._identify_degradation_type(model_id)
        
        report = {
            'model_id': model_id,
            'status': self.degradation_status.get(model_id, DegradationStatus.HEALTHY).value,
            'degradation_type': degradation_type.value if degradation_type else None,
            'current_metrics': metrics.to_dict(),
            'performance_summary': {
                'current': metrics.current_accuracy,
                'baseline': metrics.baseline_accuracy,
                'rolling_avg': metrics.rolling_accuracy,
                'drop': metrics.accuracy_drop,
                'trend': metrics.accuracy_trend,
                'volatility': metrics.volatility
            },
            'statistical_tests': {
                'drift_detected': metrics.is_significant,
                'p_value': metrics.drift_p_value,
                'confidence': 1 - metrics.drift_p_value
            },
            'data_quality': {
                'missing_features': metrics.missing_features_ratio,
                'outliers': metrics.outlier_ratio
            },
            'history_length': len(history),
            'last_retrain': self.last_retrain_date.get(model_id),
            'alerts': [a for a in self.alert_history if a['model_id'] == model_id][-5:]  # Last 5 alerts
        }
        
        return report
    
    def _identify_degradation_type(self, model_id: str) -> Optional[DegradationType]:
        """Identify the type of degradation"""
        
        history = self.performance_history.get(model_id, [])
        if len(history) < 10:
            return None
        
        accuracies = [h['accuracy'] for h in history]
        
        # Check for sudden drop
        if len(accuracies) >= 2:
            recent_drop = accuracies[-2] - accuracies[-1]
            if recent_drop > 0.1:  # 10% sudden drop
                return DegradationType.SUDDEN
        
        # Check for gradual decline
        if len(accuracies) >= 10:
            first_half = np.mean(accuracies[:len(accuracies)//2])
            second_half = np.mean(accuracies[len(accuracies)//2:])
            if first_half - second_half > 0.05:  # 5% gradual decline
                return DegradationType.GRADUAL
        
        # Check for cyclical pattern (simplified)
        if len(accuracies) >= 20:
            # Look for periodic variations
            fft = np.fft.fft(accuracies - np.mean(accuracies))
            power = np.abs(fft) ** 2
            if np.max(power[1:len(power)//2]) > np.mean(power) * 3:
                return DegradationType.CYCLICAL
        
        # Check for concept drift (statistical test)
        metrics = self._calculate_degradation_metrics(model_id)
        if metrics and metrics.is_significant:
            return DegradationType.CONCEPT_DRIFT
        
        return None
    
    def plot_degradation_history(self, model_id: str):
        """Plot degradation history for a model"""
        import matplotlib.pyplot as plt
        
        history = self.performance_history.get(model_id, [])
        if len(history) < 2:
            logger.warning(f"Insufficient history for model {model_id}")
            return None
        
        # Extract data
        timestamps = [h['timestamp'] for h in history]
        accuracies = [h['accuracy'] for h in history]
        
        # Calculate rolling average
        window = min(self.monitoring_window, len(accuracies))
        rolling_avg = pd.Series(accuracies).rolling(window).mean()
        
        # Get baseline
        baseline = self.model_baselines.get(model_id, np.mean(accuracies))
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot 1: Performance over time
        ax1.plot(timestamps, accuracies, 'b-', alpha=0.5, label='Actual')
        ax1.plot(timestamps, rolling_avg, 'b-', linewidth=2, label=f'{window}-day MA')
        ax1.axhline(y=baseline, color='g', linestyle='--', label='Baseline')
        ax1.axhline(y=self.triggers.min_accuracy_threshold, color='r', linestyle='--', label='Min Threshold')
        
        # Mark degradation zones
        for i, acc in enumerate(accuracies):
            if acc < self.triggers.min_accuracy_threshold:
                ax1.axvspan(timestamps[i], timestamps[min(i+1, len(timestamps)-1)], 
                           alpha=0.2, color='red')
            elif baseline - acc > self.triggers.accuracy_drop_threshold:
                ax1.axvspan(timestamps[i], timestamps[min(i+1, len(timestamps)-1)], 
                           alpha=0.2, color='orange')
        
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'Model Performance: {model_id}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Degradation metrics
        if len(accuracies) >= 3:
            # Calculate rolling metrics
            drops = [baseline - acc for acc in accuracies]
            volatilities = pd.Series(accuracies).rolling(window).std()
            
            ax2.plot(timestamps, drops, 'r-', alpha=0.7, label='Accuracy Drop')
            ax2.plot(timestamps, volatilities, 'purple', alpha=0.7, label='Volatility')
            ax2.axhline(y=self.triggers.accuracy_drop_threshold, color='r', linestyle='--', alpha=0.5)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Metric Value')
            ax2.set_title('Degradation Metrics')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Mark retrain events
        for retrain_date in [self.last_retrain_date.get(model_id)] if model_id in self.last_retrain_date else []:
            if retrain_date:
                for ax in [ax1, ax2]:
                    ax.axvline(x=retrain_date, color='green', linestyle=':', alpha=0.7, label='Retrain')
        
        plt.tight_layout()
        
        # Add status to title
        status = self.degradation_status.get(model_id, DegradationStatus.HEALTHY)
        fig.suptitle(f'Model Degradation Analysis: {model_id} (Status: {status.value})', 
                    fontsize=14, y=1.02)
        
        return fig
    
    def save_monitoring_state(self, filepath: str):
        """Save monitoring state to file"""
        state = {
            'model_baselines': self.model_baselines,
            'last_retrain_dates': {k: v.isoformat() for k, v in self.last_retrain_date.items()},
            'degradation_status': {k: v.value for k, v in self.degradation_status.items()},
            'alert_history': self.alert_history[-100:],  # Keep last 100 alerts
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Monitoring state saved to {filepath}")
    
    def load_monitoring_state(self, filepath: str):
        """Load monitoring state from file"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.model_baselines = state.get('model_baselines', {})
        
        # Parse dates
        for model_id, date_str in state.get('last_retrain_dates', {}).items():
            self.last_retrain_date[model_id] = datetime.fromisoformat(date_str)
        
        # Parse status
        for model_id, status_str in state.get('degradation_status', {}).items():
            self.degradation_status[model_id] = DegradationStatus(status_str)
        
        self.alert_history = state.get('alert_history', [])
        
        logger.info(f"Monitoring state loaded from {filepath}")


def create_degradation_monitor(baseline_window: int = 30,
                              monitoring_window: int = 7,
                              triggers: Optional[RetrainingTrigger] = None) -> ModelDegradationMonitor:
    """Create degradation monitor instance"""
    return ModelDegradationMonitor(baseline_window, monitoring_window, triggers)


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Create monitor
    triggers = RetrainingTrigger(
        accuracy_drop_threshold=0.05,
        min_accuracy_threshold=0.55,
        auto_retrain=True
    )
    
    monitor = create_degradation_monitor(
        baseline_window=30,
        monitoring_window=7,
        triggers=triggers
    )
    
    # Simulate model performance over time
    np.random.seed(42)
    model_id = "test_model"
    
    # Simulate 60 days of performance
    for day in range(60):
        # Simulate degradation after day 30
        if day < 30:
            # Good performance
            accuracy = np.random.normal(0.70, 0.02)
        elif day < 45:
            # Gradual degradation
            accuracy = np.random.normal(0.68 - (day - 30) * 0.005, 0.03)
        else:
            # Recovery after retrain
            accuracy = np.random.normal(0.69, 0.02)
        
        # Ensure valid range
        accuracy = np.clip(accuracy, 0, 1)
        
        # Generate fake predictions
        n_samples = 100
        y_true = np.random.randint(0, 2, n_samples)
        y_pred = (np.random.random(n_samples) < accuracy).astype(int)
        confidence = np.random.beta(5, 2, n_samples)  # Skewed confidence
        
        # Update monitor
        monitor.update_performance(model_id, y_true, y_pred, confidence)
        
        # Check status every 10 days
        if day % 10 == 0 and day > 0:
            report = monitor.get_degradation_report(model_id)
            print(f"Day {day}: Status={report['status']}, "
                  f"Accuracy={report['performance_summary']['current']:.3f}")
    
    # Final report
    report = monitor.get_degradation_report(model_id)
    print("\nFinal Degradation Report:")
    print(f"  Status: {report['status']}")
    print(f"  Degradation Type: {report.get('degradation_type', 'None')}")
    print(f"  Current Accuracy: {report['performance_summary']['current']:.3f}")
    print(f"  Baseline Accuracy: {report['performance_summary']['baseline']:.3f}")
    print(f"  Accuracy Drop: {report['performance_summary']['drop']:.3f}")
    print(f"  Trend: {report['performance_summary']['trend']:.4f}")
    print(f"  Drift Detected: {report['statistical_tests']['drift_detected']}")
    
    # Plot history
    fig = monitor.plot_degradation_history(model_id)
    if fig:
        plt.savefig('degradation_history.png')
        print("\nDegradation history plot saved to degradation_history.png")
    
    # Save state
    monitor.save_monitoring_state('monitor_state.json')
    print("Monitor state saved to monitor_state.json")