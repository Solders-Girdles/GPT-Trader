"""
Anomaly Detection System
Phase 3, Week 3-4: RISK-009, RISK-010, RISK-011, RISK-012
Comprehensive anomaly detection using ML and statistical methods
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ..utils.serialization import (
    load_json,
    save_json,
    save_model,
)
from ..utils.serialization import (
    load_model as load_secure_model,
)

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of anomalies"""

    OUTLIER = "outlier"  # Single point outliers
    CONTEXTUAL = "contextual"  # Context-specific anomalies
    COLLECTIVE = "collective"  # Collective/pattern anomalies
    TREND = "trend"  # Trend breaks
    VOLATILITY = "volatility"  # Volatility regime changes
    CORRELATION = "correlation"  # Correlation breakdowns
    MICROSTRUCTURE = "microstructure"  # Market microstructure anomalies


class DetectionMethod(Enum):
    """Anomaly detection methods"""

    ISOLATION_FOREST = "isolation_forest"
    LSTM = "lstm"
    STATISTICAL = "statistical"
    EWMA = "ewma"
    CUSUM = "cusum"
    MAD = "mad"  # Median Absolute Deviation
    LOF = "lof"  # Local Outlier Factor


@dataclass
class Anomaly:
    """Detected anomaly"""

    timestamp: datetime
    anomaly_type: AnomalyType
    detection_method: DetectionMethod
    severity: float  # 0-1 scale
    confidence: float  # 0-1 scale

    # Anomaly details
    metric_name: str
    observed_value: float
    expected_value: float | None = None
    deviation: float | None = None

    # Context
    context: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "type": self.anomaly_type.value,
            "method": self.detection_method.value,
            "severity": self.severity,
            "confidence": self.confidence,
            "metric": self.metric_name,
            "value": self.observed_value,
            "expected": self.expected_value,
            "description": self.description,
        }


@dataclass
class AnomalyDetectorConfig:
    """Configuration for anomaly detection"""

    # Isolation Forest parameters
    contamination: float = 0.1  # Expected proportion of outliers
    n_estimators: int = 100
    max_samples: int | float | str = "auto"

    # Statistical parameters
    ewma_alpha: float = 0.2  # EWMA smoothing factor
    cusum_threshold: float = 5  # CUSUM threshold
    mad_threshold: float = 3  # MAD threshold multiplier

    # LSTM parameters (placeholder for RISK-010)
    lstm_sequence_length: int = 20
    lstm_threshold: float = 2.0  # Standard deviations

    # General parameters
    min_samples: int = 100  # Minimum samples for training
    update_frequency: int = 100  # Update model every N samples
    history_size: int = 1000  # Size of historical data to keep


class IsolationForestDetector:
    """
    Isolation Forest based anomaly detector.

    Isolation Forest isolates anomalies by randomly selecting features
    and split values, requiring fewer splits for anomalies.
    """

    def __init__(self, config: AnomalyDetectorConfig):
        """
        Initialize Isolation Forest detector.

        Args:
            config: Detector configuration
        """
        self.config = config
        self.model = IsolationForest(
            contamination=config.contamination,
            n_estimators=config.n_estimators,
            max_samples=config.max_samples,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names: list[str] = []
        self.training_data: pd.DataFrame | None = None

    def fit(self, data: pd.DataFrame, feature_columns: list[str] | None = None):
        """
        Train Isolation Forest on historical data.

        Args:
            data: Historical data
            feature_columns: Columns to use as features
        """
        if len(data) < self.config.min_samples:
            logger.warning(
                f"Insufficient data for training: {len(data)} < {self.config.min_samples}"
            )
            return

        # Select features
        if feature_columns:
            self.feature_names = feature_columns
        else:
            # Use all numeric columns
            self.feature_names = data.select_dtypes(include=[np.number]).columns.tolist()

        # Prepare training data
        X = data[self.feature_names].values

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model.fit(X_scaled)
        self.is_fitted = True
        self.training_data = data.copy()

        logger.info(
            f"Isolation Forest trained on {len(data)} samples with {len(self.feature_names)} features"
        )

    def detect(self, data: pd.DataFrame) -> list[Anomaly]:
        """
        Detect anomalies in new data.

        Args:
            data: New data to check

        Returns:
            List of detected anomalies
        """
        if not self.is_fitted:
            logger.warning("Model not fitted yet")
            return []

        anomalies = []

        # Prepare data
        X = data[self.feature_names].values
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = self.scaler.transform(X)

        # Predict anomalies (-1 for anomaly, 1 for normal)
        predictions = self.model.predict(X_scaled)
        anomaly_scores = self.model.score_samples(X_scaled)

        # Process anomalies
        for i, (pred, score) in enumerate(zip(predictions, anomaly_scores, strict=False)):
            if pred == -1:  # Anomaly detected
                # Calculate severity based on anomaly score
                severity = min(1.0, abs(score) / 2)  # Normalize to 0-1

                # Find the most anomalous feature
                feature_contributions = self._get_feature_contributions(X_scaled[i])
                max_contrib_idx = np.argmax(np.abs(feature_contributions))
                anomalous_feature = self.feature_names[max_contrib_idx]

                anomaly = Anomaly(
                    timestamp=(
                        data.index[i] if hasattr(data.index[i], "to_pydatetime") else datetime.now()
                    ),
                    anomaly_type=AnomalyType.OUTLIER,
                    detection_method=DetectionMethod.ISOLATION_FOREST,
                    severity=severity,
                    confidence=0.8,  # Isolation Forest confidence
                    metric_name=anomalous_feature,
                    observed_value=float(X[i, max_contrib_idx]),
                    context={"anomaly_score": float(score)},
                    description=f"Outlier detected in {anomalous_feature}",
                )
                anomalies.append(anomaly)

        return anomalies

    def _get_feature_contributions(self, sample: np.ndarray) -> np.ndarray:
        """Calculate feature contributions to anomaly score"""
        # Simplified: use absolute z-scores as proxy for contribution
        if self.training_data is not None:
            mean = self.scaler.mean_
            std = self.scaler.scale_
            z_scores = np.abs((sample - mean) / (std + 1e-10))
            return z_scores
        return np.zeros_like(sample)

    def update(self, new_data: pd.DataFrame):
        """
        Update model with new data.

        Args:
            new_data: New data for updating
        """
        if self.training_data is not None:
            # Append new data
            combined_data = pd.concat([self.training_data, new_data])

            # Keep only recent history
            if len(combined_data) > self.config.history_size:
                combined_data = combined_data.iloc[-self.config.history_size :]

            # Retrain
            self.fit(combined_data, self.feature_names)


class StatisticalAnomalyDetector:
    """
    Statistical process control based anomaly detection.

    Implements EWMA, CUSUM, and MAD-based detection methods.
    """

    def __init__(self, config: AnomalyDetectorConfig):
        """
        Initialize statistical detector.

        Args:
            config: Detector configuration
        """
        self.config = config

        # EWMA state
        self.ewma_values: dict[str, float] = {}
        self.ewma_std: dict[str, float] = {}

        # CUSUM state
        self.cusum_pos: dict[str, float] = {}
        self.cusum_neg: dict[str, float] = {}
        self.cusum_target: dict[str, float] = {}

        # Historical data for MAD
        self.history: dict[str, deque] = {}
        self.history_size = 100

    def detect_ewma(self, metric_name: str, value: float) -> Anomaly | None:
        """
        Detect anomalies using EWMA control charts.

        Args:
            metric_name: Name of the metric
            value: Current value

        Returns:
            Anomaly if detected, None otherwise
        """
        alpha = self.config.ewma_alpha

        # Initialize if first value
        if metric_name not in self.ewma_values:
            self.ewma_values[metric_name] = value
            self.ewma_std[metric_name] = 0
            return None

        # Update EWMA
        ewma_old = self.ewma_values[metric_name]
        ewma_new = alpha * value + (1 - alpha) * ewma_old
        self.ewma_values[metric_name] = ewma_new

        # Update standard deviation estimate
        deviation = abs(value - ewma_old)
        self.ewma_std[metric_name] = alpha * deviation + (1 - alpha) * self.ewma_std[metric_name]

        # Check for anomaly (3-sigma rule)
        if self.ewma_std[metric_name] > 0:
            z_score = abs(value - ewma_new) / self.ewma_std[metric_name]

            if z_score > 3:
                return Anomaly(
                    timestamp=datetime.now(),
                    anomaly_type=AnomalyType.OUTLIER,
                    detection_method=DetectionMethod.EWMA,
                    severity=min(1.0, z_score / 6),  # Normalize to 0-1
                    confidence=0.7,
                    metric_name=metric_name,
                    observed_value=value,
                    expected_value=ewma_new,
                    deviation=z_score,
                    description=f"EWMA anomaly: {z_score:.1f} standard deviations from expected",
                )

        return None

    def detect_cusum(self, metric_name: str, value: float) -> Anomaly | None:
        """
        Detect anomalies using CUSUM control charts.

        Args:
            metric_name: Name of the metric
            value: Current value

        Returns:
            Anomaly if detected, None otherwise
        """
        # Initialize if first value
        if metric_name not in self.cusum_pos:
            self.cusum_pos[metric_name] = 0
            self.cusum_neg[metric_name] = 0
            self.cusum_target[metric_name] = value
            return None

        target = self.cusum_target[metric_name]
        k = 0.5  # Slack parameter
        h = self.config.cusum_threshold  # Decision threshold

        # Update CUSUM statistics
        self.cusum_pos[metric_name] = max(0, self.cusum_pos[metric_name] + value - target - k)
        self.cusum_neg[metric_name] = max(0, self.cusum_neg[metric_name] - value + target - k)

        # Check for anomaly
        if self.cusum_pos[metric_name] > h:
            # Positive shift detected
            return Anomaly(
                timestamp=datetime.now(),
                anomaly_type=AnomalyType.TREND,
                detection_method=DetectionMethod.CUSUM,
                severity=min(1.0, self.cusum_pos[metric_name] / (2 * h)),
                confidence=0.75,
                metric_name=metric_name,
                observed_value=value,
                expected_value=target,
                deviation=self.cusum_pos[metric_name],
                description=f"CUSUM: Upward shift detected (score: {self.cusum_pos[metric_name]:.2f})",
            )
        elif self.cusum_neg[metric_name] > h:
            # Negative shift detected
            return Anomaly(
                timestamp=datetime.now(),
                anomaly_type=AnomalyType.TREND,
                detection_method=DetectionMethod.CUSUM,
                severity=min(1.0, self.cusum_neg[metric_name] / (2 * h)),
                confidence=0.75,
                metric_name=metric_name,
                observed_value=value,
                expected_value=target,
                deviation=-self.cusum_neg[metric_name],
                description=f"CUSUM: Downward shift detected (score: {self.cusum_neg[metric_name]:.2f})",
            )

        # Update target using exponential smoothing
        self.cusum_target[metric_name] = 0.95 * target + 0.05 * value

        return None

    def detect_mad(self, metric_name: str, value: float) -> Anomaly | None:
        """
        Detect anomalies using Median Absolute Deviation.

        Args:
            metric_name: Name of the metric
            value: Current value

        Returns:
            Anomaly if detected, None otherwise
        """
        # Initialize history if needed
        if metric_name not in self.history:
            self.history[metric_name] = deque(maxlen=self.history_size)

        history = self.history[metric_name]
        history.append(value)

        # Need minimum samples
        if len(history) < 20:
            return None

        # Calculate MAD
        values = np.array(history)
        median = np.median(values)
        mad = np.median(np.abs(values - median))

        # Modified Z-score
        if mad > 0:
            modified_z_score = 0.6745 * (value - median) / mad

            if abs(modified_z_score) > self.config.mad_threshold:
                return Anomaly(
                    timestamp=datetime.now(),
                    anomaly_type=AnomalyType.OUTLIER,
                    detection_method=DetectionMethod.MAD,
                    severity=min(1.0, abs(modified_z_score) / (2 * self.config.mad_threshold)),
                    confidence=0.8,
                    metric_name=metric_name,
                    observed_value=value,
                    expected_value=median,
                    deviation=modified_z_score,
                    description=f"MAD anomaly: Modified Z-score = {modified_z_score:.2f}",
                )

        return None

    def detect(self, metrics: dict[str, float]) -> list[Anomaly]:
        """
        Detect anomalies across multiple metrics.

        Args:
            metrics: Dictionary of metric values

        Returns:
            List of detected anomalies
        """
        anomalies = []

        for metric_name, value in metrics.items():
            # Try multiple detection methods

            # EWMA detection
            anomaly = self.detect_ewma(metric_name, value)
            if anomaly:
                anomalies.append(anomaly)

            # CUSUM detection
            anomaly = self.detect_cusum(metric_name, value)
            if anomaly:
                anomalies.append(anomaly)

            # MAD detection
            anomaly = self.detect_mad(metric_name, value)
            if anomaly:
                anomalies.append(anomaly)

        return anomalies


class MarketMicrostructureAnalyzer:
    """
    Analyze market microstructure for anomalies.

    Detects anomalies in bid-ask spread, order flow, volume patterns, etc.
    """

    def __init__(self):
        """Initialize microstructure analyzer"""
        self.spread_history = deque(maxlen=100)
        self.volume_history = deque(maxlen=100)
        self.trade_imbalance_history = deque(maxlen=100)

    def analyze_spread(self, bid: float, ask: float) -> Anomaly | None:
        """
        Analyze bid-ask spread for anomalies.

        Args:
            bid: Best bid price
            ask: Best ask price

        Returns:
            Anomaly if detected
        """
        spread = ask - bid
        spread_pct = spread / ((bid + ask) / 2) if bid > 0 else 0

        self.spread_history.append(spread_pct)

        if len(self.spread_history) >= 20:
            mean_spread = np.mean(self.spread_history)
            std_spread = np.std(self.spread_history)

            if std_spread > 0:
                z_score = (spread_pct - mean_spread) / std_spread

                if abs(z_score) > 3:
                    return Anomaly(
                        timestamp=datetime.now(),
                        anomaly_type=AnomalyType.MICROSTRUCTURE,
                        detection_method=DetectionMethod.STATISTICAL,
                        severity=min(1.0, abs(z_score) / 5),
                        confidence=0.7,
                        metric_name="bid_ask_spread",
                        observed_value=spread_pct,
                        expected_value=mean_spread,
                        deviation=z_score,
                        description=f"Abnormal spread: {spread_pct:.2%} (Z={z_score:.1f})",
                    )

        return None

    def analyze_volume(self, volume: float, avg_volume: float) -> Anomaly | None:
        """
        Analyze volume patterns for anomalies.

        Args:
            volume: Current volume
            avg_volume: Average volume

        Returns:
            Anomaly if detected
        """
        if avg_volume > 0:
            volume_ratio = volume / avg_volume
            self.volume_history.append(volume_ratio)

            # Check for volume spike
            if volume_ratio > 3:  # 3x average volume
                return Anomaly(
                    timestamp=datetime.now(),
                    anomaly_type=AnomalyType.MICROSTRUCTURE,
                    detection_method=DetectionMethod.STATISTICAL,
                    severity=min(1.0, volume_ratio / 5),
                    confidence=0.8,
                    metric_name="volume_spike",
                    observed_value=volume,
                    expected_value=avg_volume,
                    deviation=volume_ratio,
                    description=f"Volume spike: {volume_ratio:.1f}x average",
                )

            # Check for volume drought
            elif volume_ratio < 0.2:  # 20% of average volume
                return Anomaly(
                    timestamp=datetime.now(),
                    anomaly_type=AnomalyType.MICROSTRUCTURE,
                    detection_method=DetectionMethod.STATISTICAL,
                    severity=min(1.0, (1 - volume_ratio) / 0.8),
                    confidence=0.8,
                    metric_name="volume_drought",
                    observed_value=volume,
                    expected_value=avg_volume,
                    deviation=volume_ratio,
                    description=f"Volume drought: {volume_ratio:.1%} of average",
                )

        return None

    def analyze_order_imbalance(self, buy_volume: float, sell_volume: float) -> Anomaly | None:
        """
        Analyze order flow imbalance.

        Args:
            buy_volume: Buy volume
            sell_volume: Sell volume

        Returns:
            Anomaly if detected
        """
        total_volume = buy_volume + sell_volume

        if total_volume > 0:
            imbalance = (buy_volume - sell_volume) / total_volume
            self.trade_imbalance_history.append(imbalance)

            if len(self.trade_imbalance_history) >= 20:
                mean_imbalance = np.mean(self.trade_imbalance_history)
                std_imbalance = np.std(self.trade_imbalance_history)

                if std_imbalance > 0:
                    z_score = (imbalance - mean_imbalance) / std_imbalance

                    if abs(z_score) > 3:
                        return Anomaly(
                            timestamp=datetime.now(),
                            anomaly_type=AnomalyType.MICROSTRUCTURE,
                            detection_method=DetectionMethod.STATISTICAL,
                            severity=min(1.0, abs(z_score) / 5),
                            confidence=0.75,
                            metric_name="order_imbalance",
                            observed_value=imbalance,
                            expected_value=mean_imbalance,
                            deviation=z_score,
                            description=f"Order imbalance: {imbalance:.1%} (Z={z_score:.1f})",
                        )

        return None


class AnomalyDetectionSystem:
    """
    Comprehensive anomaly detection system.

    Combines multiple detection methods for robust anomaly identification.
    """

    def __init__(self, config: AnomalyDetectorConfig | None = None):
        """
        Initialize anomaly detection system.

        Args:
            config: Configuration for detectors
        """
        self.config = config or AnomalyDetectorConfig()

        # Initialize detectors
        self.isolation_forest = IsolationForestDetector(self.config)
        self.statistical_detector = StatisticalAnomalyDetector(self.config)
        self.microstructure_analyzer = MarketMicrostructureAnalyzer()

        # Anomaly history
        self.anomaly_history: list[Anomaly] = []
        self.max_history = 1000

    def train(self, historical_data: pd.DataFrame):
        """
        Train anomaly detection models.

        Args:
            historical_data: Historical data for training
        """
        # Train Isolation Forest
        self.isolation_forest.fit(historical_data)

        logger.info("Anomaly detection system trained")

    def detect(
        self,
        current_data: pd.DataFrame | None = None,
        metrics: dict[str, float] | None = None,
        market_data: dict[str, float] | None = None,
    ) -> list[Anomaly]:
        """
        Detect anomalies using all available methods.

        Args:
            current_data: Current data for Isolation Forest
            metrics: Metrics for statistical detection
            market_data: Market data for microstructure analysis

        Returns:
            List of detected anomalies
        """
        anomalies = []

        # Isolation Forest detection
        if current_data is not None and self.isolation_forest.is_fitted:
            forest_anomalies = self.isolation_forest.detect(current_data)
            anomalies.extend(forest_anomalies)

        # Statistical detection
        if metrics:
            stat_anomalies = self.statistical_detector.detect(metrics)
            anomalies.extend(stat_anomalies)

        # Microstructure analysis
        if market_data:
            if "bid" in market_data and "ask" in market_data:
                spread_anomaly = self.microstructure_analyzer.analyze_spread(
                    market_data["bid"], market_data["ask"]
                )
                if spread_anomaly:
                    anomalies.append(spread_anomaly)

            if "volume" in market_data and "avg_volume" in market_data:
                volume_anomaly = self.microstructure_analyzer.analyze_volume(
                    market_data["volume"], market_data["avg_volume"]
                )
                if volume_anomaly:
                    anomalies.append(volume_anomaly)

            if "buy_volume" in market_data and "sell_volume" in market_data:
                imbalance_anomaly = self.microstructure_analyzer.analyze_order_imbalance(
                    market_data["buy_volume"], market_data["sell_volume"]
                )
                if imbalance_anomaly:
                    anomalies.append(imbalance_anomaly)

        # Store in history
        self.anomaly_history.extend(anomalies)
        if len(self.anomaly_history) > self.max_history:
            self.anomaly_history = self.anomaly_history[-self.max_history :]

        return anomalies

    def get_summary(self) -> dict[str, Any]:
        """
        Get summary of detected anomalies.

        Returns:
            Summary statistics
        """
        if not self.anomaly_history:
            return {"total_anomalies": 0}

        # Count by type
        type_counts = {}
        for anomaly in self.anomaly_history:
            type_counts[anomaly.anomaly_type.value] = (
                type_counts.get(anomaly.anomaly_type.value, 0) + 1
            )

        # Count by method
        method_counts = {}
        for anomaly in self.anomaly_history:
            method_counts[anomaly.detection_method.value] = (
                method_counts.get(anomaly.detection_method.value, 0) + 1
            )

        # Average severity
        avg_severity = np.mean([a.severity for a in self.anomaly_history])

        # Recent anomalies (last hour)
        recent_time = datetime.now() - timedelta(hours=1)
        recent_anomalies = [a for a in self.anomaly_history if a.timestamp > recent_time]

        return {
            "total_anomalies": len(self.anomaly_history),
            "recent_anomalies": len(recent_anomalies),
            "by_type": type_counts,
            "by_method": method_counts,
            "average_severity": avg_severity,
            "high_severity": len([a for a in self.anomaly_history if a.severity > 0.7]),
        }

    def save_model(self, filepath: str):
        """Save trained models to file using secure serialization"""
        filepath = Path(filepath)

        # Save models separately using joblib for sklearn models
        if self.isolation_forest.is_fitted and self.isolation_forest.model:
            save_model(
                self.isolation_forest.model, filepath.with_suffix(".model.joblib"), "sklearn"
            )

        if self.isolation_forest.is_fitted and self.isolation_forest.scaler:
            save_model(
                self.isolation_forest.scaler, filepath.with_suffix(".scaler.joblib"), "sklearn"
            )

        # Save config as JSON
        config_data = (
            self.config.__dict__.copy() if hasattr(self.config, "__dict__") else self.config
        )
        save_json(
            {"config": config_data, "is_fitted": self.isolation_forest.is_fitted},
            filepath.with_suffix(".json"),
        )

        logger.info(f"Anomaly detection models saved to {filepath}")

    def load_model(self, filepath: str):
        """Load trained models from file using secure deserialization"""
        filepath = Path(filepath)

        # Load config
        config_data = load_json(filepath.with_suffix(".json"))
        self.config = config_data.get("config", self.config)
        is_fitted = config_data.get("is_fitted", False)

        if is_fitted:
            # Load models
            model_path = filepath.with_suffix(".model.joblib")
            if model_path.exists():
                self.isolation_forest.model = load_secure_model(model_path, model_type="sklearn")

            scaler_path = filepath.with_suffix(".scaler.joblib")
            if scaler_path.exists():
                self.isolation_forest.scaler = load_secure_model(scaler_path, model_type="sklearn")

            self.isolation_forest.is_fitted = True

        logger.info(f"Anomaly detection models loaded from {filepath}")


def demonstrate_anomaly_detection():
    """Demonstrate anomaly detection system"""
    print("Anomaly Detection System Demo")
    print("=" * 60)

    # Create sample data with anomalies
    np.random.seed(42)
    n_samples = 500

    # Generate normal data
    data = pd.DataFrame(
        {
            "returns": np.random.normal(0.001, 0.02, n_samples),
            "volume": np.random.lognormal(10, 1, n_samples),
            "volatility": np.random.gamma(2, 0.01, n_samples),
            "spread": np.random.exponential(0.001, n_samples),
        }
    )

    # Inject anomalies
    anomaly_indices = [50, 150, 250, 350, 450]
    for idx in anomaly_indices:
        data.loc[idx, "returns"] = np.random.choice([-0.08, 0.08])  # Large moves
        data.loc[idx, "volume"] *= np.random.choice([0.1, 10])  # Volume anomalies
        data.loc[idx, "spread"] *= 5  # Spread widening

    # Initialize system
    config = AnomalyDetectorConfig(contamination=0.05)
    system = AnomalyDetectionSystem(config)

    # Train on first half
    print("\nTraining on historical data...")
    train_data = data.iloc[:300]
    system.train(train_data)

    # Detect on second half
    print("\nDetecting anomalies in test data...")
    test_data = data.iloc[300:]

    detected_anomalies = []

    # Batch detection with Isolation Forest
    batch_anomalies = system.detect(current_data=test_data)
    detected_anomalies.extend(batch_anomalies)

    # Stream detection with statistical methods
    for i in range(len(test_data)):
        metrics = {
            "returns": test_data.iloc[i]["returns"],
            "volume": test_data.iloc[i]["volume"],
            "volatility": test_data.iloc[i]["volatility"],
        }

        market_data = {
            "bid": 100 * (1 - test_data.iloc[i]["spread"]),
            "ask": 100 * (1 + test_data.iloc[i]["spread"]),
            "volume": test_data.iloc[i]["volume"],
            "avg_volume": train_data["volume"].mean(),
            "buy_volume": test_data.iloc[i]["volume"] * 0.6,
            "sell_volume": test_data.iloc[i]["volume"] * 0.4,
        }

        anomalies = system.detect(metrics=metrics, market_data=market_data)
        detected_anomalies.extend(anomalies)

    # Display results
    print(f"\nDetected {len(detected_anomalies)} anomalies")

    # Show summary
    summary = system.get_summary()
    print("\nAnomaly Summary:")
    print(f"  Total: {summary['total_anomalies']}")
    print(f"  High Severity: {summary['high_severity']}")
    print(f"  Average Severity: {summary['average_severity']:.2f}")

    print("\nBy Type:")
    for atype, count in summary.get("by_type", {}).items():
        print(f"  {atype}: {count}")

    print("\nBy Method:")
    for method, count in summary.get("by_method", {}).items():
        print(f"  {method}: {count}")

    # Show sample anomalies
    if detected_anomalies:
        print("\nSample Anomalies:")
        for anomaly in detected_anomalies[:5]:
            print(f"  • {anomaly.description}")
            print(f"    Severity: {anomaly.severity:.2f}, Confidence: {anomaly.confidence:.2f}")

    print("\n✅ Anomaly Detection System operational!")


if __name__ == "__main__":
    demonstrate_anomaly_detection()
