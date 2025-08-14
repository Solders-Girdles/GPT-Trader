"""
Advanced Model Degradation Detection System
Phase 3, Week 1-2: Model Performance Monitoring
Task: MON-001 to MON-008
"""

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class DegradationType(Enum):
    """Types of model degradation detected"""

    NONE = "none"
    ACCURACY_DRIFT = "accuracy_drift"
    FEATURE_DRIFT = "feature_drift"
    CONFIDENCE_DECAY = "confidence_decay"
    ERROR_PATTERN_CHANGE = "error_pattern_change"
    CONCEPT_DRIFT = "concept_drift"


@dataclass
class DegradationAlert:
    """Alert generated when degradation is detected"""

    alert_id: str
    timestamp: datetime
    degradation_type: DegradationType
    severity: str  # "low", "medium", "high", "critical"
    metric_name: str
    baseline_value: float
    current_value: float
    deviation: float
    confidence: float
    recommended_action: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DegradationMetrics:
    """Comprehensive degradation metrics"""

    accuracy_trend: list[float]
    feature_drift_scores: dict[str, float]
    confidence_distribution: dict[str, float]
    error_patterns: dict[str, int]
    ks_test_results: dict[str, tuple[float, float]]  # (statistic, p-value)
    cusum_values: list[float]
    degradation_score: float  # 0-1, higher is worse
    status: DegradationType


class AdvancedDegradationDetector:
    """
    Monitors multiple degradation signals:
    - Accuracy drift using CUSUM charts
    - Feature distribution shift using KS test
    - Prediction confidence decay
    - Error pattern changes
    """

    def __init__(
        self,
        metrics_window: int = 1000,
        drift_threshold: float = 0.05,
        confidence_threshold: float = 0.55,
        cusum_h: float = 4.0,
        cusum_k: float = 0.5,
    ):
        """
        Initialize the degradation detector.

        Args:
            metrics_window: Rolling window size for metrics
            drift_threshold: KS test p-value threshold for drift detection
            confidence_threshold: Minimum acceptable confidence
            cusum_h: CUSUM control limit
            cusum_k: CUSUM reference value (slack)
        """
        self.metrics_window = metrics_window
        self.drift_threshold = drift_threshold
        self.confidence_threshold = confidence_threshold
        self.cusum_h = cusum_h
        self.cusum_k = cusum_k

        # Storage for metrics
        self.accuracy_history = deque(maxlen=metrics_window)
        self.confidence_history = deque(maxlen=metrics_window)
        self.feature_distributions = {}
        self.baseline_distributions = {}
        self.error_patterns = deque(maxlen=metrics_window)

        # CUSUM tracking
        self.cusum_pos = 0
        self.cusum_neg = 0
        self.target_accuracy = None

        # Alert management
        self.alerts = []
        self.last_alert_time = {}

        logger.info(f"Initialized AdvancedDegradationDetector with window={metrics_window}")

    def update_baseline(
        self,
        features: pd.DataFrame,
        predictions: np.ndarray,
        actuals: np.ndarray,
        confidences: np.ndarray,
    ) -> None:
        """
        Update baseline distributions for comparison.

        Args:
            features: Feature values
            predictions: Model predictions
            actuals: Actual values
            confidences: Prediction confidences
        """
        # Store baseline feature distributions
        for col in features.columns:
            self.baseline_distributions[col] = features[col].values

        # Calculate baseline accuracy
        baseline_accuracy = np.mean(predictions == actuals)
        self.target_accuracy = baseline_accuracy

        # Initialize CUSUM with baseline
        self.cusum_pos = 0
        self.cusum_neg = 0

        logger.info(f"Updated baseline with accuracy={baseline_accuracy:.3f}")

    def detect_feature_drift(self, features: pd.DataFrame) -> dict[str, tuple[float, float]]:
        """
        MON-001: Implement Kolmogorov-Smirnov test for feature drift.

        Args:
            features: Current feature values

        Returns:
            Dictionary of feature -> (KS statistic, p-value)
        """
        ks_results = {}

        for col in features.columns:
            if col not in self.baseline_distributions:
                continue

            # Get current and baseline distributions
            current_dist = features[col].values
            baseline_dist = self.baseline_distributions[col]

            # Perform KS test
            ks_stat, p_value = stats.ks_2samp(baseline_dist, current_dist)
            ks_results[col] = (ks_stat, p_value)

            # Check for significant drift
            if p_value < self.drift_threshold:
                logger.warning(
                    f"Feature drift detected in {col}: " f"KS={ks_stat:.3f}, p={p_value:.4f}"
                )

        return ks_results

    def update_cusum(self, accuracy: float) -> tuple[float, float]:
        """
        MON-002: Add CUSUM charts for accuracy monitoring.

        Args:
            accuracy: Current accuracy value

        Returns:
            Tuple of (positive CUSUM, negative CUSUM)
        """
        if self.target_accuracy is None:
            self.target_accuracy = accuracy
            return 0, 0

        # Calculate deviation from target
        deviation = accuracy - self.target_accuracy

        # Update positive CUSUM (detects increase)
        self.cusum_pos = max(0, self.cusum_pos + deviation - self.cusum_k)

        # Update negative CUSUM (detects decrease)
        self.cusum_neg = max(0, self.cusum_neg - deviation - self.cusum_k)

        # Check for out-of-control signal
        if self.cusum_neg > self.cusum_h:
            logger.warning(
                f"CUSUM alert: Accuracy degradation detected "
                f"(CUSUM-={self.cusum_neg:.2f} > {self.cusum_h})"
            )

        return self.cusum_pos, self.cusum_neg

    def track_confidence(self, confidences: np.ndarray) -> dict[str, float]:
        """
        MON-003: Create prediction confidence tracking system.

        Args:
            confidences: Array of prediction confidences

        Returns:
            Dictionary of confidence statistics
        """
        self.confidence_history.extend(confidences)

        confidence_stats = {
            "mean": np.mean(confidences),
            "std": np.std(confidences),
            "min": np.min(confidences),
            "max": np.max(confidences),
            "below_threshold": np.mean(confidences < self.confidence_threshold),
        }

        # Check for confidence decay
        if len(self.confidence_history) > 100:
            recent = list(self.confidence_history)[-100:]
            older = (
                list(self.confidence_history)[-200:-100]
                if len(self.confidence_history) > 200
                else recent
            )

            recent_mean = np.mean(recent)
            older_mean = np.mean(older)

            if recent_mean < older_mean * 0.95:  # 5% decay
                logger.warning(f"Confidence decay detected: {older_mean:.3f} -> {recent_mean:.3f}")
                confidence_stats["decay_detected"] = True
                confidence_stats["decay_rate"] = (older_mean - recent_mean) / older_mean

        return confidence_stats

    def analyze_error_patterns(
        self, predictions: np.ndarray, actuals: np.ndarray, features: pd.DataFrame | None = None
    ) -> dict[str, Any]:
        """
        MON-004: Build error pattern analyzer.

        Args:
            predictions: Model predictions
            actuals: Actual values
            features: Optional features for detailed analysis

        Returns:
            Dictionary of error pattern statistics
        """
        errors = predictions != actuals
        self.error_patterns.extend(errors)

        error_analysis = {
            "error_rate": np.mean(errors),
            "consecutive_errors": self._count_consecutive_errors(errors),
            "error_clustering": self._calculate_error_clustering(errors),
        }

        # Analyze errors by feature if provided
        if features is not None:
            error_by_feature = {}
            for col in features.columns:
                # Group errors by feature quartiles
                quartiles = pd.qcut(features[col], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
                for quartile in ["Q1", "Q2", "Q3", "Q4"]:
                    mask = quartiles == quartile
                    if mask.any():
                        error_by_feature[f"{col}_{quartile}"] = np.mean(errors[mask])

            error_analysis["error_by_feature"] = error_by_feature

        return error_analysis

    def _count_consecutive_errors(self, errors: np.ndarray) -> int:
        """Count maximum consecutive errors"""
        max_consecutive = 0
        current_consecutive = 0

        for error in errors:
            if error:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def _calculate_error_clustering(self, errors: np.ndarray) -> float:
        """Calculate error clustering coefficient"""
        if len(errors) < 2:
            return 0.0

        # Calculate runs test for randomness
        n_errors = np.sum(errors)
        n_correct = len(errors) - n_errors

        if n_errors == 0 or n_correct == 0:
            return 1.0  # Complete clustering

        # Count runs (sequences of same value)
        runs = 1
        for i in range(1, len(errors)):
            if errors[i] != errors[i - 1]:
                runs += 1

        # Expected runs under randomness
        expected_runs = (2 * n_errors * n_correct) / len(errors) + 1

        # Clustering coefficient (0 = random, 1 = clustered)
        if expected_runs > 0:
            clustering = 1 - (runs / expected_runs)
            return max(0, min(1, clustering))

        return 0.0

    def check_degradation(
        self,
        features: pd.DataFrame,
        predictions: np.ndarray,
        actuals: np.ndarray,
        confidences: np.ndarray,
    ) -> DegradationMetrics:
        """
        Comprehensive degradation check combining all methods.

        Args:
            features: Current feature values
            predictions: Model predictions
            actuals: Actual values
            confidences: Prediction confidences

        Returns:
            DegradationMetrics object with full analysis
        """
        # Calculate current accuracy
        accuracy = np.mean(predictions == actuals)
        self.accuracy_history.append(accuracy)

        # 1. Feature drift detection (KS test)
        ks_results = self.detect_feature_drift(features)

        # 2. CUSUM for accuracy monitoring
        cusum_pos, cusum_neg = self.update_cusum(accuracy)

        # 3. Confidence tracking
        confidence_stats = self.track_confidence(confidences)

        # 4. Error pattern analysis
        error_patterns = self.analyze_error_patterns(predictions, actuals, features)

        # Calculate overall degradation score
        degradation_score = self._calculate_degradation_score(
            ks_results, cusum_neg, confidence_stats, error_patterns
        )

        # Determine degradation type
        degradation_type = self._determine_degradation_type(
            ks_results, cusum_neg, confidence_stats, error_patterns
        )

        # Create metrics object
        metrics = DegradationMetrics(
            accuracy_trend=list(self.accuracy_history),
            feature_drift_scores={k: v[1] for k, v in ks_results.items()},
            confidence_distribution=confidence_stats,
            error_patterns=error_patterns,
            ks_test_results=ks_results,
            cusum_values=[cusum_pos, cusum_neg],
            degradation_score=degradation_score,
            status=degradation_type,
        )

        # Generate alerts if needed
        self._generate_alerts(metrics)

        return metrics

    def _calculate_degradation_score(
        self,
        ks_results: dict[str, tuple[float, float]],
        cusum_neg: float,
        confidence_stats: dict[str, float],
        error_patterns: dict[str, Any],
    ) -> float:
        """Calculate overall degradation score (0-1)"""
        score = 0.0

        # Feature drift contribution (0-0.3)
        if ks_results:
            drift_scores = [1 - p for _, p in ks_results.values()]
            score += 0.3 * np.mean(drift_scores)

        # CUSUM contribution (0-0.3)
        cusum_score = min(1.0, cusum_neg / (2 * self.cusum_h))
        score += 0.3 * cusum_score

        # Confidence contribution (0-0.2)
        if "below_threshold" in confidence_stats:
            score += 0.2 * confidence_stats["below_threshold"]

        # Error pattern contribution (0-0.2)
        if "error_clustering" in error_patterns:
            score += 0.2 * error_patterns["error_clustering"]

        return min(1.0, score)

    def _determine_degradation_type(
        self,
        ks_results: dict[str, tuple[float, float]],
        cusum_neg: float,
        confidence_stats: dict[str, float],
        error_patterns: dict[str, Any],
    ) -> DegradationType:
        """Determine the primary type of degradation"""
        # Check for feature drift
        if ks_results:
            significant_drifts = sum(1 for _, p in ks_results.values() if p < self.drift_threshold)
            if significant_drifts > len(ks_results) * 0.3:
                return DegradationType.FEATURE_DRIFT

        # Check for accuracy drift
        if cusum_neg > self.cusum_h:
            return DegradationType.ACCURACY_DRIFT

        # Check for confidence decay
        if confidence_stats.get("decay_detected", False):
            return DegradationType.CONFIDENCE_DECAY

        # Check for error pattern changes
        if error_patterns.get("error_clustering", 0) > 0.5:
            return DegradationType.ERROR_PATTERN_CHANGE

        # Check for concept drift (combination of factors)
        if cusum_neg > self.cusum_h * 0.5 and len(ks_results) > 0:
            avg_p_value = np.mean([p for _, p in ks_results.values()])
            if avg_p_value < 0.1:
                return DegradationType.CONCEPT_DRIFT

        return DegradationType.NONE

    def _generate_alerts(self, metrics: DegradationMetrics) -> None:
        """Generate alerts based on degradation metrics"""
        if metrics.status == DegradationType.NONE:
            return

        # Determine severity
        if metrics.degradation_score > 0.8:
            severity = "critical"
        elif metrics.degradation_score > 0.6:
            severity = "high"
        elif metrics.degradation_score > 0.4:
            severity = "medium"
        else:
            severity = "low"

        # Check cooldown period (avoid alert spam)
        alert_key = f"{metrics.status}_{severity}"
        if alert_key in self.last_alert_time:
            time_since_last = datetime.now() - self.last_alert_time[alert_key]
            if time_since_last < timedelta(hours=1):
                return

        # Create alert
        alert = DegradationAlert(
            alert_id=f"DEG_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            degradation_type=metrics.status,
            severity=severity,
            metric_name="model_degradation",
            baseline_value=self.target_accuracy if self.target_accuracy else 0.0,
            current_value=metrics.accuracy_trend[-1] if metrics.accuracy_trend else 0.0,
            deviation=metrics.degradation_score,
            confidence=1.0 - metrics.degradation_score,
            recommended_action=self._get_recommended_action(metrics.status, severity),
            details={
                "cusum_value": metrics.cusum_values[1],
                "drift_features": len(
                    [p for p in metrics.feature_drift_scores.values() if p < self.drift_threshold]
                ),
                "confidence_mean": metrics.confidence_distribution.get("mean", 0),
                "error_clustering": metrics.error_patterns.get("error_clustering", 0),
            },
        )

        self.alerts.append(alert)
        self.last_alert_time[alert_key] = datetime.now()

        logger.warning(
            f"Degradation alert generated: {alert.degradation_type.value} "
            f"(severity={severity}, score={metrics.degradation_score:.3f})"
        )

    def _get_recommended_action(self, degradation_type: DegradationType, severity: str) -> str:
        """Get recommended action based on degradation type and severity"""
        actions = {
            DegradationType.FEATURE_DRIFT: {
                "critical": "Immediate retraining required with new data",
                "high": "Schedule retraining within 24 hours",
                "medium": "Monitor closely, prepare for retraining",
                "low": "Continue monitoring, no immediate action",
            },
            DegradationType.ACCURACY_DRIFT: {
                "critical": "Switch to fallback model immediately",
                "high": "Investigate root cause and retrain",
                "medium": "Increase monitoring frequency",
                "low": "Track trend, may be temporary",
            },
            DegradationType.CONFIDENCE_DECAY: {
                "critical": "Recalibrate model immediately",
                "high": "Review calibration and thresholds",
                "medium": "Adjust confidence thresholds",
                "low": "Monitor confidence distribution",
            },
            DegradationType.ERROR_PATTERN_CHANGE: {
                "critical": "Review data quality and model assumptions",
                "high": "Analyze error clusters for patterns",
                "medium": "Investigate specific error cases",
                "low": "Document pattern changes",
            },
            DegradationType.CONCEPT_DRIFT: {
                "critical": "Full model review and possible architecture change",
                "high": "Comprehensive retraining with recent data",
                "medium": "Evaluate model assumptions",
                "low": "Prepare for gradual adaptation",
            },
        }

        return actions.get(degradation_type, {}).get(severity, "Monitor and assess")

    def get_status(self) -> dict[str, Any]:
        """Get current degradation detector status"""
        recent_alerts = [
            a for a in self.alerts if datetime.now() - a.timestamp < timedelta(hours=24)
        ]

        return {
            "operational": True,
            "metrics_collected": len(self.accuracy_history),
            "baseline_set": bool(self.baseline_distributions),
            "recent_alerts": len(recent_alerts),
            "current_accuracy": self.accuracy_history[-1] if self.accuracy_history else None,
            "cusum_status": {
                "positive": self.cusum_pos,
                "negative": self.cusum_neg,
                "threshold": self.cusum_h,
            },
            "features_monitored": len(self.baseline_distributions),
        }

    def export_metrics(self, filepath: str) -> None:
        """Export current metrics to JSON file"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "accuracy_history": list(self.accuracy_history),
            "confidence_history": list(self.confidence_history),
            "cusum_values": [self.cusum_pos, self.cusum_neg],
            "alerts": [
                {
                    "alert_id": a.alert_id,
                    "timestamp": a.timestamp.isoformat(),
                    "type": a.degradation_type.value,
                    "severity": a.severity,
                    "score": a.deviation,
                }
                for a in self.alerts
            ],
        }

        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Exported degradation metrics to {filepath}")
