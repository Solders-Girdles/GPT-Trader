"""
Online Learning Framework for GPT-Trader Phase 2.

This module provides sophisticated online learning capabilities:
- Incremental model training with concept drift detection
- Adaptive learning rate adjustment
- Real-time model updating and validation
- Memory-efficient streaming algorithms
- Ensemble-based online learning

Enables the trading system to continuously adapt to changing market conditions
without requiring full model retraining.
"""

from __future__ import annotations

import logging
import time
import warnings
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Optional online learning libraries
try:
    from river import drift, ensemble, linear_model, metrics, preprocessing

    HAS_RIVER = True
except ImportError:
    HAS_RIVER = False
    warnings.warn("River not available. Install with: pip install river")

try:
    from skmultiflow.data import DataStream
    from skmultiflow.drift_detection import ADWIN, PageHinkley
    from skmultiflow.meta import AdaptiveRandomForestRegressor
    from skmultiflow.trees import HoeffdingTreeRegressor

    HAS_SKMULTIFLOW = True
except ImportError:
    HAS_SKMULTIFLOW = False
    warnings.warn("scikit-multiflow not available. Install with: pip install scikit-multiflow")

from bot.utils.base import BaseConfig

logger = logging.getLogger(__name__)


@dataclass
class OnlineLearningConfig(BaseConfig):
    """Configuration for online learning framework."""

    # Learning parameters
    learning_rate: float = 0.01
    adaptive_learning_rate: bool = True
    learning_rate_decay: float = 0.99
    min_learning_rate: float = 0.001

    # Drift detection parameters
    enable_drift_detection: bool = True
    drift_detection_method: str = "adwin"  # adwin, page_hinkley, cusum
    drift_threshold: float = 0.002
    warning_threshold: float = 0.001

    # Model parameters
    online_model_type: str = "sgd"  # sgd, passive_aggressive, river_linear
    ensemble_size: int = 5
    use_ensemble: bool = True

    # Memory management
    memory_window_size: int = 1000
    batch_update_size: int = 10
    validation_window_size: int = 100

    # Performance tracking
    performance_window: int = 50
    min_samples_for_update: int = 10
    performance_threshold: float = 0.1

    # Adaptation parameters
    enable_model_selection: bool = True
    model_selection_frequency: int = 100
    enable_feature_adaptation: bool = True

    # Advanced options
    enable_uncertainty_tracking: bool = True
    enable_performance_decay: bool = True
    decay_factor: float = 0.95

    # Random state
    random_state: int = 42


@dataclass
class DriftDetectionResult:
    """Result from drift detection."""

    drift_detected: bool
    warning_detected: bool
    drift_score: float
    detection_method: str
    timestamp: float
    confidence: float = 0.0


@dataclass
class OnlineLearningResult:
    """Result from online learning update."""

    model_updated: bool
    performance_metrics: dict[str, float]
    drift_result: DriftDetectionResult | None
    learning_rate: float
    samples_processed: int
    execution_time: float


class BaseDriftDetector(ABC):
    """Base class for concept drift detection."""

    @abstractmethod
    def update(self, error: float) -> DriftDetectionResult:
        """Update detector with new error and check for drift."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset detector state."""
        pass


class ADWINDriftDetector(BaseDriftDetector):
    """ADWIN (Adaptive Windowing) drift detector."""

    def __init__(self, delta: float = 0.002) -> None:
        self.delta = delta
        self.window = deque()
        self.total = 0.0
        self.variance = 0.0
        self.width = 0
        self.mint = np.inf
        self.detection_count = 0

    def update(self, error: float) -> DriftDetectionResult:
        """Update ADWIN with new error value."""
        self.window.append(error)
        self.total += error
        self.width += 1

        # Simple drift detection based on mean change
        drift_detected = False
        warning_detected = False
        drift_score = 0.0

        if self.width >= 30:  # Minimum window size
            # Calculate recent vs historical performance
            recent_window = 10
            if len(self.window) >= recent_window * 2:
                recent_errors = list(self.window)[-recent_window:]
                historical_errors = list(self.window)[:-recent_window]

                recent_mean = np.mean(recent_errors)
                historical_mean = np.mean(historical_errors)

                # Calculate drift score as relative change
                if historical_mean != 0:
                    drift_score = abs(recent_mean - historical_mean) / abs(historical_mean)
                else:
                    drift_score = abs(recent_mean - historical_mean)

                # Detect drift based on threshold
                if drift_score > self.delta * 10:  # Drift threshold
                    drift_detected = True
                    self.detection_count += 1
                elif drift_score > self.delta * 5:  # Warning threshold
                    warning_detected = True

        # Maintain window size
        if self.width > 1000:
            removed = self.window.popleft()
            self.total -= removed
            self.width -= 1

        return DriftDetectionResult(
            drift_detected=drift_detected,
            warning_detected=warning_detected,
            drift_score=drift_score,
            detection_method="adwin",
            timestamp=time.time(),
            confidence=min(drift_score / (self.delta * 10), 1.0),
        )

    def reset(self) -> None:
        """Reset detector state."""
        self.window.clear()
        self.total = 0.0
        self.width = 0
        self.detection_count = 0


class PageHinkleyDriftDetector(BaseDriftDetector):
    """Page-Hinkley drift detector."""

    def __init__(
        self, min_instances: int = 30, delta: float = 0.005, threshold: float = 50
    ) -> None:
        self.min_instances = min_instances
        self.delta = delta  # Allowable margin
        self.threshold = threshold

        self.sum_x = 0.0
        self.x_mean = 0.0
        self.sum_ph = 0.0
        self.n = 0

    def update(self, error: float) -> DriftDetectionResult:
        """Update Page-Hinkley test with new error."""
        self.n += 1

        if self.n == 1:
            self.x_mean = error
            self.sum_x = error

        self.sum_x += error
        self.x_mean = self.sum_x / self.n

        # Page-Hinkley test
        ph_value = error - self.x_mean - self.delta
        self.sum_ph += ph_value

        drift_score = abs(self.sum_ph)
        drift_detected = False
        warning_detected = False

        if self.n >= self.min_instances:
            if drift_score > self.threshold:
                drift_detected = True
            elif drift_score > self.threshold * 0.5:
                warning_detected = True

        return DriftDetectionResult(
            drift_detected=drift_detected,
            warning_detected=warning_detected,
            drift_score=drift_score,
            detection_method="page_hinkley",
            timestamp=time.time(),
            confidence=min(drift_score / self.threshold, 1.0),
        )

    def reset(self) -> None:
        """Reset detector state."""
        self.sum_x = 0.0
        self.x_mean = 0.0
        self.sum_ph = 0.0
        self.n = 0


class BaseOnlineLearner(ABC):
    """Base class for online learning models."""

    @abstractmethod
    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Incrementally fit model with new data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass

    @abstractmethod
    def get_model_state(self) -> dict[str, Any]:
        """Get current model state for monitoring."""
        pass


class SGDOnlineLearner(BaseOnlineLearner):
    """SGD-based online learner."""

    def __init__(self, learning_rate: float = 0.01, random_state: int = 42) -> None:
        self.learning_rate = learning_rate
        self.model = SGDRegressor(
            learning_rate="constant", eta0=learning_rate, random_state=random_state
        )
        self.is_fitted = False
        self.n_features_in_ = None

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Incrementally update model."""
        if not self.is_fitted:
            self.model.partial_fit(X, y)
            self.is_fitted = True
            self.n_features_in_ = X.shape[1]
        else:
            self.model.partial_fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            return np.zeros(len(X))
        return self.model.predict(X)

    def get_model_state(self) -> dict[str, Any]:
        """Get model state."""
        return {
            "is_fitted": self.is_fitted,
            "n_features": self.n_features_in_,
            "learning_rate": self.learning_rate,
            "coef_norm": np.linalg.norm(self.model.coef_) if self.is_fitted else 0.0,
        }


class OnlineEnsemble(BaseOnlineLearner):
    """Ensemble of online learners."""

    def __init__(
        self, n_models: int = 5, base_learning_rate: float = 0.01, random_state: int = 42
    ) -> None:
        self.n_models = n_models
        self.base_learning_rate = base_learning_rate
        self.random_state = random_state

        # Create ensemble of models with different learning rates
        self.models = []
        for i in range(n_models):
            lr = base_learning_rate * (0.5 + i * 0.3)  # Vary learning rates
            model = SGDOnlineLearner(learning_rate=lr, random_state=random_state + i)
            self.models.append(model)

        self.model_weights = np.ones(n_models) / n_models
        self.model_performance = np.zeros(n_models)
        self.performance_history = [deque(maxlen=50) for _ in range(n_models)]

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Update all models in ensemble."""
        predictions = []

        # Update each model and get predictions
        for i, model in enumerate(self.models):
            model.partial_fit(X, y)
            pred = model.predict(X)
            predictions.append(pred)

            # Track performance
            if len(y) > 0:
                mse = mean_squared_error(y, pred)
                self.performance_history[i].append(mse)

                # Update recent performance
                if len(self.performance_history[i]) >= 10:
                    recent_performance = np.mean(list(self.performance_history[i])[-10:])
                    self.model_performance[i] = 1.0 / (1.0 + recent_performance)  # Higher is better

        # Update model weights based on performance
        self._update_weights()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        if not any(model.is_fitted for model in self.models):
            return np.zeros(len(X))

        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions)

        # Weighted ensemble prediction
        ensemble_pred = np.average(predictions, axis=0, weights=self.model_weights)
        return ensemble_pred

    def _update_weights(self) -> None:
        """Update model weights based on recent performance."""
        if np.sum(self.model_performance) > 0:
            # Softmax weighting
            exp_performance = np.exp(self.model_performance - np.max(self.model_performance))
            self.model_weights = exp_performance / np.sum(exp_performance)
        else:
            self.model_weights = np.ones(self.n_models) / self.n_models

    def get_model_state(self) -> dict[str, Any]:
        """Get ensemble state."""
        return {
            "n_models": self.n_models,
            "model_weights": self.model_weights.tolist(),
            "model_performance": self.model_performance.tolist(),
            "fitted_models": sum(1 for m in self.models if m.is_fitted),
        }


class OnlineLearningFramework:
    """
    Comprehensive online learning framework for trading strategies.

    Provides incremental learning, concept drift detection, and adaptive
    model selection for real-time market condition adaptation.
    """

    def __init__(self, config: OnlineLearningConfig) -> None:
        self.config = config

        # Initialize components
        self.drift_detector = self._create_drift_detector()
        self.learner = self._create_online_learner()
        self.scaler = StandardScaler()

        # State tracking
        self.is_fitted = False
        self.samples_processed = 0
        self.batch_buffer_X = []
        self.batch_buffer_y = []

        # Performance tracking
        self.performance_history = deque(maxlen=config.performance_window)
        self.drift_history = []
        self.learning_rate_history = deque(maxlen=100)
        self.current_learning_rate = config.learning_rate

        # Validation data for performance monitoring
        self.validation_X = deque(maxlen=config.validation_window_size)
        self.validation_y = deque(maxlen=config.validation_window_size)

    def _create_drift_detector(self) -> BaseDriftDetector:
        """Create drift detector based on configuration."""
        if self.config.drift_detection_method == "adwin":
            return ADWINDriftDetector(delta=self.config.drift_threshold)
        elif self.config.drift_detection_method == "page_hinkley":
            return PageHinkleyDriftDetector(delta=self.config.drift_threshold)
        else:
            return ADWINDriftDetector(delta=self.config.drift_threshold)

    def _create_online_learner(self) -> BaseOnlineLearner:
        """Create online learner based on configuration."""
        if self.config.use_ensemble:
            return OnlineEnsemble(
                n_models=self.config.ensemble_size,
                base_learning_rate=self.config.learning_rate,
                random_state=self.config.random_state,
            )
        else:
            return SGDOnlineLearner(
                learning_rate=self.config.learning_rate, random_state=self.config.random_state
            )

    def initialize(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Initialize the online learning framework with initial data."""
        logger.info("Initializing online learning framework...")

        # Convert to numpy
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y

        # Fit scaler
        self.scaler.fit(X_array)
        X_scaled = self.scaler.transform(X_array)

        # Initial model training
        self.learner.partial_fit(X_scaled, y_array)

        # Initialize validation data
        split_idx = int(len(X_array) * 0.8)
        val_X = X_scaled[split_idx:]
        val_y = y_array[split_idx:]

        for x, y_val in zip(val_X, val_y, strict=False):
            self.validation_X.append(x)
            self.validation_y.append(y_val)

        self.is_fitted = True
        self.samples_processed = len(X_array)

        logger.info(f"Online learning initialized with {len(X_array)} samples")

    def update(
        self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray
    ) -> OnlineLearningResult:
        """
        Update the online learning model with new data.

        Args:
            X: New feature data
            y: New target data

        Returns:
            OnlineLearningResult with update information
        """
        start_time = time.time()

        if not self.is_fitted:
            raise ValueError("Framework must be initialized first")

        # Convert to numpy
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y

        # Scale features
        X_scaled = self.scaler.transform(X_array)

        # Add to batch buffer
        self.batch_buffer_X.extend(X_scaled)
        self.batch_buffer_y.extend(y_array)

        model_updated = False
        drift_result = None

        # Process batch if buffer is full
        if len(self.batch_buffer_X) >= self.config.batch_update_size:
            model_updated, drift_result = self._process_batch()

        # Update validation data
        for x, y_val in zip(X_scaled, y_array, strict=False):
            self.validation_X.append(x)
            self.validation_y.append(y_val)

        # Calculate current performance
        performance_metrics = self._calculate_performance_metrics()

        self.samples_processed += len(X_array)
        execution_time = time.time() - start_time

        return OnlineLearningResult(
            model_updated=model_updated,
            performance_metrics=performance_metrics,
            drift_result=drift_result,
            learning_rate=self.current_learning_rate,
            samples_processed=len(X_array),
            execution_time=execution_time,
        )

    def _process_batch(self) -> tuple[bool, DriftDetectionResult | None]:
        """Process accumulated batch of data."""
        if not self.batch_buffer_X:
            return False, None

        X_batch = np.array(self.batch_buffer_X)
        y_batch = np.array(self.batch_buffer_y)

        # Get predictions before update for drift detection
        predictions = self.learner.predict(X_batch)
        errors = np.abs(predictions - y_batch)
        avg_error = np.mean(errors)

        # Drift detection
        drift_result = None
        if self.config.enable_drift_detection:
            drift_result = self.drift_detector.update(avg_error)

            if drift_result.drift_detected:
                logger.warning(f"Concept drift detected! Score: {drift_result.drift_score:.4f}")
                self._handle_drift_detection(drift_result)

        # Update learning rate
        if self.config.adaptive_learning_rate:
            self._update_learning_rate(avg_error)

        # Update model
        self.learner.partial_fit(X_batch, y_batch)

        # Store performance
        if len(self.validation_X) > 0:
            val_X = np.array(list(self.validation_X)[-50:])  # Use recent validation data
            val_y = np.array(list(self.validation_y)[-50:])
            val_pred = self.learner.predict(val_X)
            val_mse = mean_squared_error(val_y, val_pred)
            self.performance_history.append(val_mse)

        # Clear batch buffer
        self.batch_buffer_X.clear()
        self.batch_buffer_y.clear()

        return True, drift_result

    def _handle_drift_detection(self, drift_result: DriftDetectionResult) -> None:
        """Handle detected concept drift."""
        logger.info("Handling concept drift...")

        # Reset drift detector
        self.drift_detector.reset()

        # Increase learning rate temporarily
        if self.config.adaptive_learning_rate:
            self.current_learning_rate = min(
                self.current_learning_rate * 2.0, self.config.learning_rate * 5.0
            )

        # Store drift event
        self.drift_history.append(
            {
                "timestamp": time.time(),
                "drift_score": drift_result.drift_score,
                "samples_processed": self.samples_processed,
            }
        )

    def _update_learning_rate(self, current_error: float) -> None:
        """Adaptively update learning rate based on performance."""
        # Simple adaptive rule: increase if error is high, decrease if low
        if len(self.performance_history) >= 5:
            recent_errors = list(self.performance_history)[-5:]
            avg_recent_error = np.mean(recent_errors)

            if current_error > avg_recent_error * 1.2:
                # Error increasing, increase learning rate
                self.current_learning_rate = min(
                    self.current_learning_rate * 1.05, self.config.learning_rate * 3.0
                )
            elif current_error < avg_recent_error * 0.8:
                # Error decreasing, decrease learning rate
                self.current_learning_rate = max(
                    self.current_learning_rate * self.config.learning_rate_decay,
                    self.config.min_learning_rate,
                )

        self.learning_rate_history.append(self.current_learning_rate)

    def _calculate_performance_metrics(self) -> dict[str, float]:
        """Calculate current performance metrics."""
        metrics = {}

        if len(self.validation_X) >= 10:
            val_X = np.array(list(self.validation_X)[-50:])
            val_y = np.array(list(self.validation_y)[-50:])

            try:
                predictions = self.learner.predict(val_X)

                metrics["mse"] = mean_squared_error(val_y, predictions)
                metrics["rmse"] = np.sqrt(metrics["mse"])
                metrics["r2"] = r2_score(val_y, predictions)
                metrics["mae"] = np.mean(np.abs(val_y - predictions))

                # Prediction stability
                if len(predictions) > 1:
                    metrics["prediction_std"] = np.std(predictions)

            except Exception as e:
                logger.warning(f"Error calculating metrics: {e}")
                metrics["mse"] = 0.0

        # Model state metrics
        model_state = self.learner.get_model_state()
        metrics.update({f"model_{k}": v for k, v in model_state.items()})

        # Framework metrics
        metrics.update(
            {
                "samples_processed": self.samples_processed,
                "current_learning_rate": self.current_learning_rate,
                "drift_events": len(self.drift_history),
                "buffer_size": len(self.batch_buffer_X),
            }
        )

        return metrics

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Make predictions with current model."""
        if not self.is_fitted:
            raise ValueError("Framework must be initialized first")

        X_array = X.values if isinstance(X, pd.DataFrame) else X
        X_scaled = self.scaler.transform(X_array)

        return self.learner.predict(X_scaled)

    def get_framework_state(self) -> dict[str, Any]:
        """Get comprehensive framework state."""
        state = {
            "is_fitted": self.is_fitted,
            "samples_processed": self.samples_processed,
            "current_learning_rate": self.current_learning_rate,
            "drift_events": len(self.drift_history),
            "recent_performance": (
                list(self.performance_history)[-10:] if self.performance_history else []
            ),
            "model_state": self.learner.get_model_state(),
        }

        if self.drift_history:
            state["last_drift"] = self.drift_history[-1]

        return state

    def reset_framework(self) -> None:
        """Reset framework state for retraining."""
        logger.info("Resetting online learning framework...")

        self.drift_detector.reset()
        self.learner = self._create_online_learner()

        self.is_fitted = False
        self.samples_processed = 0
        self.batch_buffer_X.clear()
        self.batch_buffer_y.clear()

        self.performance_history.clear()
        self.drift_history.clear()
        self.learning_rate_history.clear()
        self.current_learning_rate = self.config.learning_rate

        self.validation_X.clear()
        self.validation_y.clear()


def create_default_online_learner() -> OnlineLearningFramework:
    """Create default online learning framework."""
    config = OnlineLearningConfig(
        learning_rate=0.01,
        adaptive_learning_rate=True,
        enable_drift_detection=True,
        drift_detection_method="adwin",
        use_ensemble=True,
        ensemble_size=3,
        batch_update_size=10,
        performance_window=50,
    )

    return OnlineLearningFramework(config)


# Integration with existing GPT-Trader strategy optimization
class OnlineStrategyLearner:
    """
    Online learning wrapper for trading strategy adaptation.

    Enables strategies to continuously learn and adapt to market changes
    without requiring full retraining cycles.
    """

    def __init__(
        self,
        strategy_class,
        initial_data: pd.DataFrame,
        learning_config: OnlineLearningConfig | None = None,
    ) -> None:
        self.strategy_class = strategy_class
        self.learning_config = learning_config or OnlineLearningConfig()
        self.online_framework = OnlineLearningFramework(self.learning_config)

        # Strategy-specific components
        self.feature_columns = None
        self.target_column = "future_return"
        self.prediction_history = deque(maxlen=1000)
        self.strategy_performance = deque(maxlen=252)  # 1 year of daily performance

        # Initialize with historical data
        self._initialize_with_data(initial_data)

    def _initialize_with_data(self, data: pd.DataFrame) -> None:
        """Initialize online learning with historical strategy data."""
        # Prepare features and targets for learning
        features = self._extract_features(data)
        targets = self._calculate_targets(data)

        # Initialize framework
        self.online_framework.initialize(features, targets)

        logger.info(f"Online strategy learner initialized with {len(data)} samples")

    def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features relevant to strategy performance."""
        features = pd.DataFrame(index=data.index)

        # Price-based features
        if "Close" in data.columns:
            features["returns"] = data["Close"].pct_change()
            features["volatility"] = features["returns"].rolling(20).std()
            features["momentum"] = data["Close"] / data["Close"].shift(10) - 1

        # Volume features
        if "Volume" in data.columns:
            features["volume_ratio"] = data["Volume"] / data["Volume"].rolling(20).mean()

        # Technical indicators
        if "Close" in data.columns:
            # Moving averages
            features["ma_short"] = data["Close"].rolling(5).mean() / data["Close"] - 1
            features["ma_long"] = data["Close"].rolling(20).mean() / data["Close"] - 1

            # RSI approximation
            delta = data["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features["rsi"] = 100 - (100 / (1 + rs))

        # Market regime indicators
        if len(features.columns) > 0:
            features["regime_vol"] = (
                features["volatility"] > features["volatility"].rolling(50).mean()
            )
            features["regime_trend"] = features["momentum"] > 0

        # Clean features
        features = features.fillna(0)
        self.feature_columns = list(features.columns)

        return features

    def _calculate_targets(self, data: pd.DataFrame) -> pd.Series:
        """Calculate target variable for strategy learning."""
        if "Close" not in data.columns:
            return pd.Series(0, index=data.index)

        # Forward-looking returns as target
        future_returns = data["Close"].shift(-1) / data["Close"] - 1
        return future_returns.fillna(0)

    def update_with_new_data(self, new_data: pd.DataFrame) -> OnlineLearningResult:
        """Update online learner with new market data."""
        # Extract features and targets
        features = self._extract_features(new_data)
        targets = self._calculate_targets(new_data)

        # Update online framework
        result = self.online_framework.update(features, targets)

        # Track predictions
        if len(features) > 0:
            predictions = self.online_framework.predict(features)
            for pred in predictions:
                self.prediction_history.append(pred)

        return result

    def predict_strategy_signal(self, current_data: pd.DataFrame) -> dict[str, Any]:
        """Generate strategy signal using online learning predictions."""
        features = self._extract_features(current_data)

        if len(features) == 0:
            return {"signal": 0, "confidence": 0.0, "prediction": 0.0}

        # Get prediction from online model
        prediction = self.online_framework.predict(features)

        # Convert prediction to trading signal
        signal_threshold = 0.001  # 0.1% return threshold

        if len(prediction) > 0:
            pred_value = prediction[-1]  # Most recent prediction

            if pred_value > signal_threshold:
                signal = 1  # Buy signal
            elif pred_value < -signal_threshold:
                signal = -1  # Sell signal
            else:
                signal = 0  # Hold signal

            confidence = min(abs(pred_value) / signal_threshold, 1.0)

        else:
            signal, confidence, pred_value = 0, 0.0, 0.0

        return {
            "signal": signal,
            "confidence": confidence,
            "prediction": pred_value,
            "features_used": len(features.columns),
            "samples_processed": self.online_framework.samples_processed,
        }

    def get_learning_summary(self) -> dict[str, Any]:
        """Get summary of online learning performance."""
        framework_state = self.online_framework.get_framework_state()

        summary = {
            "framework_state": framework_state,
            "feature_count": len(self.feature_columns) if self.feature_columns else 0,
            "prediction_history_length": len(self.prediction_history),
            "recent_predictions": list(self.prediction_history)[-10:],
            "strategy_performance_length": len(self.strategy_performance),
        }

        if len(self.prediction_history) >= 10:
            recent_preds = list(self.prediction_history)[-10:]
            summary["prediction_volatility"] = np.std(recent_preds)
            summary["prediction_mean"] = np.mean(recent_preds)

        return summary


# Example usage with trend breakout strategy
def create_online_trend_breakout_learner(initial_data: pd.DataFrame) -> OnlineStrategyLearner:
    """Create online learner for trend breakout strategy."""

    # This would integrate with existing TrendBreakoutStrategy
    # from bot.strategy.trend_breakout import TrendBreakoutStrategy

    config = OnlineLearningConfig(
        learning_rate=0.005,  # Conservative learning rate for trading
        adaptive_learning_rate=True,
        enable_drift_detection=True,
        drift_detection_method="adwin",
        use_ensemble=True,
        ensemble_size=3,
        batch_update_size=5,  # Small batches for frequent updates
        performance_window=100,
    )

    # Note: In actual implementation, pass real strategy class
    return OnlineStrategyLearner(
        strategy_class=None,  # TrendBreakoutStrategy,
        initial_data=initial_data,
        learning_config=config,
    )
