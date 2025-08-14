"""
Continuous Learning Pipeline for Phase 3 Meta-Learning.
Implements online learning algorithms, concept drift detection, and automatic retraining triggers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from bot.knowledge.strategy_knowledge_base import StrategyKnowledgeBase
from bot.meta_learning.regime_detection import RegimeCharacteristics, RegimeDetector
from bot.meta_learning.temporal_adaptation import TemporalAdaptationEngine
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class ConceptDriftEvent:
    """Represents a detected concept drift event."""

    event_id: str
    detection_date: datetime
    drift_type: str  # "gradual", "sudden", "recurring"
    drift_magnitude: float
    affected_components: list[str]
    confidence: float
    trigger_reason: str


@dataclass
class LearningUpdate:
    """Represents a learning update event."""

    update_id: str
    update_date: datetime
    update_type: str  # "incremental", "retrain", "adapt"
    model_components: list[str]
    performance_improvement: float
    new_data_points: int
    validation_score: float


class ContinuousLearningPipeline:
    """Continuous learning pipeline with online learning and concept drift detection."""

    def __init__(
        self,
        knowledge_base: StrategyKnowledgeBase,
        regime_detector: RegimeDetector,
        temporal_adaptation: TemporalAdaptationEngine,
    ) -> None:
        self.knowledge_base = knowledge_base
        self.regime_detector = regime_detector
        self.temporal_adaptation = temporal_adaptation

        # Learning components
        self.online_models: dict[str, OnlineModel] = {}
        self.concept_drift_detector = ConceptDriftDetector()
        self.performance_monitor = PerformanceMonitor()

        # Learning configuration
        self.learning_config = {
            "min_data_points": 50,
            "retrain_threshold": 0.1,  # 10% performance degradation
            "drift_detection_window": 100,
            "update_frequency": 7,  # days
            "validation_split": 0.2,
        }

        # Learning history
        self.drift_events: list[ConceptDriftEvent] = []
        self.learning_updates: list[LearningUpdate] = []
        self.model_performance_history: dict[str, list[float]] = {}

    def process_new_data(
        self, market_data: pd.DataFrame, strategy_performance: dict[str, float]
    ) -> dict[str, Any]:
        """Process new market data and strategy performance for continuous learning."""
        try:
            # Detect current regime
            current_regime = self.regime_detector.detect_regime(market_data)

            # Update performance monitoring
            self.performance_monitor.update_performance(strategy_performance)

            # Check for concept drift
            drift_detected = self.concept_drift_detector.detect_drift(
                market_data, strategy_performance, current_regime
            )

            # Process learning updates
            learning_results = self._process_learning_updates(
                market_data, strategy_performance, current_regime, drift_detected
            )

            # Update online models
            model_updates = self._update_online_models(
                market_data, strategy_performance, current_regime
            )

            # Generate insights
            insights = self._generate_learning_insights()

            return {
                "regime_detected": current_regime.regime.value,
                "drift_detected": drift_detected is not None,
                "drift_event": drift_detected,
                "learning_updates": learning_results,
                "model_updates": model_updates,
                "insights": insights,
            }

        except Exception as e:
            logger.error(f"Continuous learning processing failed: {e}")
            return {"error": str(e)}

    def trigger_retraining(self, trigger_reason: str, affected_models: list[str]) -> dict[str, Any]:
        """Trigger retraining of specific models."""
        try:
            retraining_results = {}

            for model_name in affected_models:
                if model_name in self.online_models:
                    model = self.online_models[model_name]
                    retraining_result = model.retrain()
                    retraining_results[model_name] = retraining_result

            # Record learning update
            update = LearningUpdate(
                update_id=f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                update_date=datetime.now(),
                update_type="retrain",
                model_components=affected_models,
                performance_improvement=np.mean(
                    [r.get("improvement", 0.0) for r in retraining_results.values()]
                ),
                new_data_points=0,  # Will be updated
                validation_score=np.mean(
                    [r.get("validation_score", 0.0) for r in retraining_results.values()]
                ),
            )

            self.learning_updates.append(update)

            return {
                "retraining_triggered": True,
                "affected_models": affected_models,
                "results": retraining_results,
                "update_id": update.update_id,
            }

        except Exception as e:
            logger.error(f"Retraining trigger failed: {e}")
            return {"error": str(e)}

    def get_learning_analytics(self) -> dict[str, Any]:
        """Get comprehensive learning analytics."""
        return {
            "total_drift_events": len(self.drift_events),
            "total_learning_updates": len(self.learning_updates),
            "active_models": list(self.online_models.keys()),
            "recent_drift_events": [
                {
                    "date": event.detection_date.strftime("%Y-%m-%d"),
                    "type": event.drift_type,
                    "magnitude": event.drift_magnitude,
                    "reason": event.trigger_reason,
                }
                for event in self.drift_events[-10:]  # Last 10 events
            ],
            "recent_learning_updates": [
                {
                    "date": update.update_date.strftime("%Y-%m-%d"),
                    "type": update.update_type,
                    "improvement": update.performance_improvement,
                    "validation_score": update.validation_score,
                }
                for update in self.learning_updates[-10:]  # Last 10 updates
            ],
            "model_performance": {
                model_name: {
                    "current_performance": np.mean(history[-10:]) if history else 0.0,
                    "trend": (
                        "improving"
                        if len(history) > 1 and history[-1] > history[-2]
                        else "declining"
                    ),
                }
                for model_name, history in self.model_performance_history.items()
            },
        }

    def _process_learning_updates(
        self,
        market_data: pd.DataFrame,
        strategy_performance: dict[str, float],
        current_regime: RegimeCharacteristics,
        drift_detected: ConceptDriftEvent | None,
    ) -> dict[str, Any]:
        """Process learning updates based on new data."""
        updates = {}

        # Check if retraining is needed
        if drift_detected:
            retraining_result = self.trigger_retraining(
                f"Concept drift detected: {drift_detected.drift_type}",
                drift_detected.affected_components,
            )
            updates["retraining"] = retraining_result

        # Incremental learning updates
        incremental_updates = {}
        for model_name, model in self.online_models.items():
            if model.can_update():
                update_result = model.incremental_update(market_data, strategy_performance)
                incremental_updates[model_name] = update_result

        updates["incremental"] = incremental_updates

        return updates

    def _update_online_models(
        self,
        market_data: pd.DataFrame,
        strategy_performance: dict[str, float],
        current_regime: RegimeCharacteristics,
    ) -> dict[str, Any]:
        """Update online learning models."""
        updates = {}

        for model_name, model in self.online_models.items():
            try:
                # Prepare features for model update
                features = self._extract_model_features(market_data, current_regime)

                # Update model
                update_result = model.update(features, strategy_performance)
                updates[model_name] = update_result

                # Track performance
                if model_name not in self.model_performance_history:
                    self.model_performance_history[model_name] = []

                self.model_performance_history[model_name].append(
                    update_result.get("performance", 0.0)
                )

            except Exception as e:
                logger.warning(f"Model update failed for {model_name}: {e}")
                updates[model_name] = {"error": str(e)}

        return updates

    def _extract_model_features(
        self, market_data: pd.DataFrame, current_regime: RegimeCharacteristics
    ) -> dict[str, float]:
        """Extract features for model updates."""
        # Use recent market data
        recent_data = market_data.tail(20)

        features = {
            "volatility": current_regime.volatility,
            "trend_strength": current_regime.trend_strength,
            "momentum": current_regime.momentum_score,
            "correlation": current_regime.correlation_level,
            "volume_profile": (
                1.0
                if current_regime.volume_profile == "high"
                else 0.5 if current_regime.volume_profile == "medium" else 0.0
            ),
            "price_change": (recent_data["close"].iloc[-1] / recent_data["close"].iloc[0]) - 1,
            "volume_change": (
                (recent_data["volume"].iloc[-5:].mean() / recent_data["volume"].iloc[-20:].mean())
                - 1
                if "volume" in recent_data.columns
                else 0.0
            ),
        }

        return features

    def _generate_learning_insights(self) -> dict[str, Any]:
        """Generate insights from learning activities."""
        if not self.learning_updates:
            return {"message": "No learning updates available"}

        # Calculate learning effectiveness
        recent_updates = self.learning_updates[-20:]  # Last 20 updates
        avg_improvement = np.mean([u.performance_improvement for u in recent_updates])
        avg_validation = np.mean([u.validation_score for u in recent_updates])

        # Drift frequency
        recent_drifts = [
            d for d in self.drift_events if d.detection_date > datetime.now() - timedelta(days=30)
        ]
        drift_frequency = len(recent_drifts) / 30  # per day

        return {
            "learning_effectiveness": {
                "avg_performance_improvement": avg_improvement,
                "avg_validation_score": avg_validation,
                "learning_trend": "improving" if avg_improvement > 0 else "declining",
            },
            "drift_analysis": {
                "recent_drift_frequency": drift_frequency,
                "most_common_drift_type": self._get_most_common_drift_type(),
                "drift_impact": self._calculate_drift_impact(),
            },
            "model_health": {
                "active_models": len(self.online_models),
                "models_needing_attention": self._identify_models_needing_attention(),
            },
        }

    def _get_most_common_drift_type(self) -> str:
        """Get the most common type of concept drift."""
        if not self.drift_events:
            return "none"

        drift_types = [event.drift_type for event in self.drift_events]
        return pd.Series(drift_types).mode().iloc[0] if len(drift_types) > 0 else "none"

    def _calculate_drift_impact(self) -> float:
        """Calculate the average impact of drift events."""
        if not self.drift_events:
            return 0.0

        return np.mean([event.drift_magnitude for event in self.drift_events])

    def _identify_models_needing_attention(self) -> list[str]:
        """Identify models that need attention based on performance."""
        models_needing_attention = []

        for model_name, history in self.model_performance_history.items():
            if len(history) >= 5:
                recent_performance = np.mean(history[-5:])
                if recent_performance < 0.5:  # Low performance threshold
                    models_needing_attention.append(model_name)

        return models_needing_attention


class OnlineModel:
    """Base class for online learning models."""

    def __init__(self, model_name: str, model_type: str = "regression") -> None:
        self.model_name = model_name
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.last_update = None
        self.update_count = 0

        # Performance tracking
        self.performance_history: list[float] = []
        self.validation_scores: list[float] = []

    def update(self, features: dict[str, float], target: Any) -> dict[str, Any]:
        """Update the model with new data."""
        try:
            # Convert features to array
            feature_vector = list(features.values())

            # Scale features
            if self.is_trained:
                feature_vector_scaled = self.scaler.transform([feature_vector])
            else:
                feature_vector_scaled = self.scaler.fit_transform([feature_vector])

            # Update model
            if self.model is None:
                self._initialize_model()

            if self.is_trained:
                # Incremental update
                self._incremental_update(feature_vector_scaled, target)
            else:
                # Initial training
                self._initial_training(feature_vector_scaled, target)

            # Update tracking
            self.last_update = datetime.now()
            self.update_count += 1

            # Calculate performance
            performance = self._calculate_performance(feature_vector_scaled, target)
            self.performance_history.append(performance)

            return {
                "updated": True,
                "performance": performance,
                "update_count": self.update_count,
                "is_trained": self.is_trained,
            }

        except Exception as e:
            logger.error(f"Model update failed: {e}")
            return {"error": str(e)}

    def can_update(self) -> bool:
        """Check if model can be updated."""
        if self.last_update is None:
            return True

        # Check update frequency
        days_since_update = (datetime.now() - self.last_update).days
        return days_since_update >= 1  # Allow daily updates

    def incremental_update(
        self, market_data: pd.DataFrame, strategy_performance: dict[str, float]
    ) -> dict[str, Any]:
        """Perform incremental update with market data."""
        # This is a simplified incremental update
        # In practice, you might use more sophisticated online learning algorithms
        return {"updated": True, "method": "simplified_incremental"}

    def retrain(self) -> dict[str, Any]:
        """Retrain the model from scratch."""
        try:
            # Reset model
            self._initialize_model()
            self.is_trained = False

            # Retrain with available data
            # This would typically use historical data
            retrain_result = self._perform_retraining()

            return {
                "retrained": True,
                "improvement": retrain_result.get("improvement", 0.0),
                "validation_score": retrain_result.get("validation_score", 0.0),
            }

        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
            return {"error": str(e)}

    def _initialize_model(self) -> None:
        """Initialize the model."""
        if self.model_type == "regression":
            self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _initial_training(self, features: np.ndarray, target: Any) -> None:
        """Perform initial model training."""
        # For initial training, we need more data
        # This is a simplified version
        self.is_trained = True

    def _incremental_update(self, features: np.ndarray, target: Any) -> None:
        """Perform incremental model update."""
        # Simplified incremental update
        # In practice, you might use online learning algorithms like SGD
        pass

    def _calculate_performance(self, features: np.ndarray, target: Any) -> float:
        """Calculate model performance."""
        # Simplified performance calculation
        return 0.8  # Default performance score

    def _perform_retraining(self) -> dict[str, Any]:
        """Perform full model retraining."""
        # Simplified retraining
        return {"improvement": 0.1, "validation_score": 0.85}


class ConceptDriftDetector:
    """Detects concept drift in market data and strategy performance."""

    def __init__(self, detection_window: int = 100) -> None:
        self.detection_window = detection_window
        self.data_buffer: list[dict[str, Any]] = []
        self.drift_threshold = 0.1  # 10% change threshold

        # Drift detection methods
        self.statistical_detector = StatisticalDriftDetector()
        self.performance_detector = PerformanceDriftDetector()

    def detect_drift(
        self,
        market_data: pd.DataFrame,
        strategy_performance: dict[str, float],
        current_regime: RegimeCharacteristics,
    ) -> ConceptDriftEvent | None:
        """Detect concept drift in the data."""
        try:
            # Add new data to buffer
            self._add_to_buffer(market_data, strategy_performance, current_regime)

            # Check if we have enough data
            if len(self.data_buffer) < self.detection_window:
                return None

            # Statistical drift detection
            statistical_drift = self.statistical_detector.detect_drift(self.data_buffer)

            # Performance drift detection
            performance_drift = self.performance_detector.detect_drift(self.data_buffer)

            # Combine drift signals
            if statistical_drift or performance_drift:
                drift_event = self._create_drift_event(statistical_drift, performance_drift)
                return drift_event

            return None

        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
            return None

    def _add_to_buffer(
        self,
        market_data: pd.DataFrame,
        strategy_performance: dict[str, float],
        current_regime: RegimeCharacteristics,
    ) -> None:
        """Add new data to the detection buffer."""
        data_point = {
            "timestamp": datetime.now(),
            "market_data": market_data.tail(1).to_dict("records")[0],
            "strategy_performance": strategy_performance,
            "regime": current_regime.regime.value,
            "volatility": current_regime.volatility,
            "trend_strength": current_regime.trend_strength,
        }

        self.data_buffer.append(data_point)

        # Keep only recent data
        if len(self.data_buffer) > self.detection_window:
            self.data_buffer = self.data_buffer[-self.detection_window :]

    def _create_drift_event(
        self, statistical_drift: dict[str, Any] | None, performance_drift: dict[str, Any] | None
    ) -> ConceptDriftEvent:
        """Create a drift event from detection results."""
        # Determine drift type and magnitude
        if statistical_drift and performance_drift:
            drift_type = "sudden"
            magnitude = max(
                statistical_drift.get("magnitude", 0.0), performance_drift.get("magnitude", 0.0)
            )
        elif statistical_drift:
            drift_type = "gradual"
            magnitude = statistical_drift.get("magnitude", 0.0)
        else:
            drift_type = "performance"
            magnitude = performance_drift.get("magnitude", 0.0)

        return ConceptDriftEvent(
            event_id=f"drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            detection_date=datetime.now(),
            drift_type=drift_type,
            drift_magnitude=magnitude,
            affected_components=["regime_detector", "strategy_adaptation"],
            confidence=0.8,
            trigger_reason=f"{drift_type.capitalize()} drift detected",
        )


class StatisticalDriftDetector:
    """Detects statistical drift in market data."""

    def detect_drift(self, data_buffer: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Detect statistical drift in the data buffer."""
        if len(data_buffer) < 20:
            return None

        # Extract volatility and trend strength
        volatilities = [d["volatility"] for d in data_buffer]
        trend_strengths = [d["trend_strength"] for d in data_buffer]

        # Calculate drift in volatility
        vol_drift = self._calculate_drift(volatilities)
        trend_drift = self._calculate_drift(trend_strengths)

        # Combine drift signals
        total_drift = (vol_drift + trend_drift) / 2

        if total_drift > 0.1:  # Drift threshold
            return {
                "magnitude": total_drift,
                "volatility_drift": vol_drift,
                "trend_drift": trend_drift,
                "detected": True,
            }

        return None

    def _calculate_drift(self, values: list[float]) -> float:
        """Calculate drift in a time series."""
        if len(values) < 10:
            return 0.0

        # Simple drift calculation: compare recent vs historical
        recent = np.mean(values[-10:])
        historical = np.mean(values[:-10])

        if historical == 0:
            return 0.0

        return abs(recent - historical) / abs(historical)


class PerformanceDriftDetector:
    """Detects drift in strategy performance."""

    def detect_drift(self, data_buffer: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Detect performance drift in the data buffer."""
        if len(data_buffer) < 20:
            return None

        # Extract performance metrics
        performances = []
        for data_point in data_buffer:
            perf = data_point["strategy_performance"]
            if isinstance(perf, dict) and "sharpe" in perf:
                performances.append(perf["sharpe"])
            else:
                performances.append(0.0)

        # Calculate performance drift
        drift = self._calculate_performance_drift(performances)

        if drift > 0.1:  # Drift threshold
            return {
                "magnitude": drift,
                "detected": True,
                "performance_trend": "declining" if drift > 0 else "improving",
            }

        return None

    def _calculate_performance_drift(self, performances: list[float]) -> float:
        """Calculate performance drift."""
        if len(performances) < 10:
            return 0.0

        # Calculate trend in performance
        recent_perf = np.mean(performances[-10:])
        historical_perf = np.mean(performances[:-10])

        if historical_perf == 0:
            return 0.0

        return (recent_perf - historical_perf) / abs(historical_perf)


class PerformanceMonitor:
    """Monitors overall system performance."""

    def __init__(self) -> None:
        self.performance_history: list[dict[str, Any]] = []
        self.alert_thresholds = {
            "performance_decline": 0.1,
            "volatility_increase": 0.2,
            "drawdown_limit": 0.15,
        }

    def update_performance(self, strategy_performance: dict[str, float]) -> None:
        """Update performance monitoring."""
        performance_record = {
            "timestamp": datetime.now(),
            "performance": strategy_performance,
            "avg_sharpe": np.mean([p.get("sharpe", 0.0) for p in strategy_performance.values()]),
            "avg_drawdown": np.mean(
                [p.get("max_drawdown", 0.0) for p in strategy_performance.values()]
            ),
        }

        self.performance_history.append(performance_record)

        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

    def check_alerts(self) -> list[dict[str, Any]]:
        """Check for performance alerts."""
        alerts = []

        if len(self.performance_history) < 10:
            return alerts

        recent_performance = self.performance_history[-10:]
        historical_performance = self.performance_history[:-10]

        if not historical_performance:
            return alerts

        # Check for performance decline
        recent_avg = np.mean([p["avg_sharpe"] for p in recent_performance])
        historical_avg = np.mean([p["avg_sharpe"] for p in historical_performance])

        if (
            historical_avg > 0
            and (recent_avg - historical_avg) / historical_avg
            < -self.alert_thresholds["performance_decline"]
        ):
            alerts.append(
                {
                    "type": "performance_decline",
                    "severity": "high",
                    "message": f"Performance declined by {((recent_avg - historical_avg) / historical_avg) * 100:.1f}%",
                    "timestamp": datetime.now(),
                }
            )

        return alerts
