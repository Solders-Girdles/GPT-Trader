"""
Shadow Mode Prediction System
Phase 3, Week 2: MON-016
Run models in parallel without affecting production
"""

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ShadowStatus(Enum):
    """Shadow mode status"""

    INACTIVE = "inactive"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"


class ComparisonMode(Enum):
    """How to compare shadow predictions"""

    EXACT = "exact"  # Predictions must match exactly
    THRESHOLD = "threshold"  # Within threshold distance
    DIRECTIONAL = "directional"  # Same direction (up/down)
    STATISTICAL = "statistical"  # Statistical equivalence


@dataclass
class ShadowConfig:
    """Configuration for shadow mode"""

    # Basic settings
    model_id: str
    shadow_model_id: str

    # Duration settings
    min_predictions: int = 1000
    max_predictions: int = 100000
    duration_hours: int = 24

    # Comparison settings
    comparison_mode: ComparisonMode = ComparisonMode.THRESHOLD
    agreement_threshold: float = 0.95  # 95% agreement
    value_threshold: float = 0.01  # 1% difference threshold

    # Performance settings
    log_predictions: bool = True
    log_interval: int = 100  # Log every N predictions
    batch_size: int = 1  # Process predictions in batches

    # Monitoring settings
    track_latency: bool = True
    track_memory: bool = True
    track_errors: bool = True

    # Safety settings
    stop_on_error: bool = False
    max_error_rate: float = 0.05  # 5% error rate
    timeout_seconds: float = 5.0  # Per prediction timeout


@dataclass
class ShadowPrediction:
    """Single shadow prediction result"""

    timestamp: datetime
    prediction_id: str

    # Inputs
    features: dict[str, Any]

    # Production prediction
    prod_prediction: Any
    prod_confidence: float | None = None
    prod_latency_ms: float | None = None

    # Shadow prediction
    shadow_prediction: Any = None
    shadow_confidence: float | None = None
    shadow_latency_ms: float | None = None

    # Comparison
    agreement: bool | None = None
    difference: float | None = None

    # Metadata
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ShadowResults:
    """Aggregated shadow mode results"""

    model_id: str
    shadow_model_id: str
    status: ShadowStatus

    # Timing
    start_time: datetime
    end_time: datetime | None = None
    duration_hours: float = 0

    # Counts
    total_predictions: int = 0
    successful_predictions: int = 0
    failed_predictions: int = 0

    # Agreement metrics
    agreement_rate: float = 0
    exact_matches: int = 0
    directional_matches: int = 0

    # Performance comparison
    avg_prod_latency_ms: float = 0
    avg_shadow_latency_ms: float = 0
    latency_ratio: float = 1.0  # shadow/prod

    # Accuracy comparison (if ground truth available)
    prod_accuracy: float | None = None
    shadow_accuracy: float | None = None

    # Error analysis
    error_rate: float = 0
    error_types: dict[str, int] = field(default_factory=dict)

    # Confidence analysis
    avg_prod_confidence: float = 0
    avg_shadow_confidence: float = 0
    confidence_correlation: float | None = None

    # Detailed metrics
    hourly_metrics: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "model_id": self.model_id,
            "shadow_model_id": self.shadow_model_id,
            "status": self.status.value,
            "total_predictions": self.total_predictions,
            "agreement_rate": self.agreement_rate,
            "avg_shadow_latency_ms": self.avg_shadow_latency_ms,
            "latency_ratio": self.latency_ratio,
            "error_rate": self.error_rate,
        }


class ShadowMode:
    """
    Shadow mode prediction system for safe model testing.

    Features:
    - Parallel prediction execution
    - No impact on production
    - Comprehensive comparison metrics
    - Latency and resource tracking
    - Automatic result aggregation
    """

    def __init__(
        self,
        production_model: Callable,
        shadow_model: Callable,
        config: ShadowConfig | None = None,
    ):
        """
        Initialize shadow mode.

        Args:
            production_model: Production model predict function
            shadow_model: Shadow model predict function
            config: Shadow mode configuration
        """
        self.production_model = production_model
        self.shadow_model = shadow_model
        self.config = config or ShadowConfig(model_id="production", shadow_model_id="shadow")

        # State tracking
        self.status = ShadowStatus.INACTIVE
        self.predictions: deque = deque(maxlen=10000)  # Keep last 10k
        self.results = ShadowResults(
            model_id=self.config.model_id,
            shadow_model_id=self.config.shadow_model_id,
            status=ShadowStatus.INACTIVE,
            start_time=datetime.now(),
        )

        # Performance tracking
        self.latency_buffer = deque(maxlen=1000)
        self.error_buffer = deque(maxlen=100)

        # Threading
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._stop_flag = threading.Event()

        # Callbacks
        self.on_prediction_callbacks: list[Callable] = []
        self.on_completion_callbacks: list[Callable] = []

    def start(self) -> None:
        """Start shadow mode"""
        if self.status == ShadowStatus.RUNNING:
            logger.warning("Shadow mode already running")
            return

        logger.info(f"Starting shadow mode for {self.config.shadow_model_id}")
        self.status = ShadowStatus.STARTING
        self.results.start_time = datetime.now()
        self._stop_flag.clear()
        self.status = ShadowStatus.RUNNING

    def stop(self) -> ShadowResults:
        """
        Stop shadow mode and return results.

        Returns:
            Aggregated shadow results
        """
        if self.status != ShadowStatus.RUNNING:
            logger.warning("Shadow mode not running")
            return self.results

        logger.info("Stopping shadow mode")
        self.status = ShadowStatus.STOPPING
        self._stop_flag.set()

        # Wait for pending predictions
        self.executor.shutdown(wait=True, timeout=10)

        # Finalize results
        self.results.end_time = datetime.now()
        self.results.duration_hours = (
            self.results.end_time - self.results.start_time
        ).total_seconds() / 3600
        self.status = ShadowStatus.COMPLETED
        self.results.status = ShadowStatus.COMPLETED

        # Calculate final metrics
        self._calculate_final_metrics()

        # Trigger callbacks
        for callback in self.on_completion_callbacks:
            try:
                callback(self.results)
            except Exception as e:
                logger.error(f"Callback error: {e}")

        return self.results

    def predict(self, features: dict[str, Any], prediction_id: str | None = None) -> Any:
        """
        Make prediction with shadow mode.

        Args:
            features: Input features
            prediction_id: Optional prediction ID

        Returns:
            Production model prediction (shadow runs async)
        """
        if self.status != ShadowStatus.RUNNING:
            # Not in shadow mode, just return production prediction
            return self.production_model(features)

        # Generate prediction ID
        if prediction_id is None:
            prediction_id = f"pred_{datetime.now().timestamp()}"

        # Create prediction record
        shadow_pred = ShadowPrediction(
            timestamp=datetime.now(), prediction_id=prediction_id, features=features
        )

        # Make production prediction
        prod_start = time.time()
        try:
            prod_result = self.production_model(features)
            shadow_pred.prod_prediction = prod_result
            shadow_pred.prod_latency_ms = (time.time() - prod_start) * 1000

            # Extract confidence if available
            if isinstance(prod_result, tuple) and len(prod_result) > 1:
                shadow_pred.prod_prediction = prod_result[0]
                shadow_pred.prod_confidence = prod_result[1]
        except Exception as e:
            logger.error(f"Production prediction error: {e}")
            raise

        # Run shadow prediction asynchronously
        if not self._stop_flag.is_set():
            future = self.executor.submit(self._shadow_predict, shadow_pred, features)
            # Don't wait for shadow result

        # Return production prediction immediately
        return prod_result

    def _shadow_predict(self, shadow_pred: ShadowPrediction, features: dict[str, Any]) -> None:
        """
        Execute shadow prediction (runs async).

        Args:
            shadow_pred: Prediction record
            features: Input features
        """
        shadow_start = time.time()

        try:
            # Make shadow prediction with timeout
            shadow_result = self.shadow_model(features)
            shadow_pred.shadow_prediction = shadow_result
            shadow_pred.shadow_latency_ms = (time.time() - shadow_start) * 1000

            # Extract confidence if available
            if isinstance(shadow_result, tuple) and len(shadow_result) > 1:
                shadow_pred.shadow_prediction = shadow_result[0]
                shadow_pred.shadow_confidence = shadow_result[1]

            # Compare predictions
            shadow_pred.agreement, shadow_pred.difference = self._compare_predictions(
                shadow_pred.prod_prediction, shadow_pred.shadow_prediction
            )

        except Exception as e:
            shadow_pred.error = str(e)
            self.error_buffer.append(e)
            logger.debug(f"Shadow prediction error: {e}")

            # Stop if error rate too high
            if self.config.stop_on_error:
                error_rate = len(self.error_buffer) / max(len(self.predictions), 1)
                if error_rate > self.config.max_error_rate:
                    logger.error(f"Shadow error rate {error_rate:.2%} exceeds threshold")
                    self.stop()

        # Store prediction
        self.predictions.append(shadow_pred)
        self.results.total_predictions += 1

        if shadow_pred.error:
            self.results.failed_predictions += 1
        else:
            self.results.successful_predictions += 1

        # Update running metrics
        self._update_metrics(shadow_pred)

        # Log progress
        if (
            self.config.log_predictions
            and self.results.total_predictions % self.config.log_interval == 0
        ):
            self._log_progress()

        # Trigger callbacks
        for callback in self.on_prediction_callbacks:
            try:
                callback(shadow_pred)
            except Exception as e:
                logger.error(f"Callback error: {e}")

        # Check completion criteria
        if self.results.total_predictions >= self.config.min_predictions:
            hours_elapsed = (datetime.now() - self.results.start_time).total_seconds() / 3600
            if (
                self.results.total_predictions >= self.config.max_predictions
                or hours_elapsed >= self.config.duration_hours
            ):
                logger.info("Shadow mode completion criteria met")
                self.stop()

    def _compare_predictions(self, prod_pred: Any, shadow_pred: Any) -> tuple[bool, float]:
        """
        Compare production and shadow predictions.

        Args:
            prod_pred: Production prediction
            shadow_pred: Shadow prediction

        Returns:
            Tuple of (agreement, difference)
        """
        if shadow_pred is None:
            return False, float("inf")

        # Handle different comparison modes
        if self.config.comparison_mode == ComparisonMode.EXACT:
            agreement = prod_pred == shadow_pred
            difference = 0 if agreement else 1

        elif self.config.comparison_mode == ComparisonMode.THRESHOLD:
            # Numerical comparison
            try:
                difference = abs(float(prod_pred) - float(shadow_pred))
                agreement = difference <= self.config.value_threshold
            except (TypeError, ValueError):
                # Fall back to exact comparison for non-numeric
                agreement = prod_pred == shadow_pred
                difference = 0 if agreement else 1

        elif self.config.comparison_mode == ComparisonMode.DIRECTIONAL:
            # Check if predictions have same sign/direction
            try:
                prod_sign = np.sign(float(prod_pred))
                shadow_sign = np.sign(float(shadow_pred))
                agreement = prod_sign == shadow_sign
                difference = abs(prod_sign - shadow_sign)
            except (TypeError, ValueError):
                agreement = prod_pred == shadow_pred
                difference = 0 if agreement else 1

        else:  # STATISTICAL
            # Would need more context for statistical comparison
            agreement = prod_pred == shadow_pred
            difference = 0 if agreement else 1

        return agreement, difference

    def _update_metrics(self, prediction: ShadowPrediction) -> None:
        """Update running metrics"""
        # Update agreement rate
        if prediction.agreement is not None:
            recent_agreements = [
                p.agreement for p in list(self.predictions)[-100:] if p.agreement is not None
            ]
            if recent_agreements:
                self.results.agreement_rate = sum(recent_agreements) / len(recent_agreements)

        # Update latency metrics
        if prediction.prod_latency_ms:
            prod_latencies = [
                p.prod_latency_ms for p in list(self.predictions)[-100:] if p.prod_latency_ms
            ]
            self.results.avg_prod_latency_ms = np.mean(prod_latencies)

        if prediction.shadow_latency_ms:
            shadow_latencies = [
                p.shadow_latency_ms for p in list(self.predictions)[-100:] if p.shadow_latency_ms
            ]
            self.results.avg_shadow_latency_ms = np.mean(shadow_latencies)

            if self.results.avg_prod_latency_ms > 0:
                self.results.latency_ratio = (
                    self.results.avg_shadow_latency_ms / self.results.avg_prod_latency_ms
                )

        # Update error rate
        self.results.error_rate = self.results.failed_predictions / max(
            self.results.total_predictions, 1
        )

    def _calculate_final_metrics(self) -> None:
        """Calculate final aggregated metrics"""
        if not self.predictions:
            return

        # Agreement metrics
        agreements = [p.agreement for p in self.predictions if p.agreement is not None]
        if agreements:
            self.results.agreement_rate = sum(agreements) / len(agreements)
            self.results.exact_matches = sum(agreements)

        # Latency metrics
        prod_latencies = [p.prod_latency_ms for p in self.predictions if p.prod_latency_ms]
        shadow_latencies = [p.shadow_latency_ms for p in self.predictions if p.shadow_latency_ms]

        if prod_latencies:
            self.results.avg_prod_latency_ms = np.mean(prod_latencies)
        if shadow_latencies:
            self.results.avg_shadow_latency_ms = np.mean(shadow_latencies)

        # Confidence metrics
        prod_confidences = [p.prod_confidence for p in self.predictions if p.prod_confidence]
        shadow_confidences = [p.shadow_confidence for p in self.predictions if p.shadow_confidence]

        if prod_confidences:
            self.results.avg_prod_confidence = np.mean(prod_confidences)
        if shadow_confidences:
            self.results.avg_shadow_confidence = np.mean(shadow_confidences)

        # Correlation between confidences
        if (
            prod_confidences
            and shadow_confidences
            and len(prod_confidences) == len(shadow_confidences)
        ):
            self.results.confidence_correlation = np.corrcoef(
                prod_confidences[: len(shadow_confidences)],
                shadow_confidences[: len(prod_confidences)],
            )[0, 1]

        # Error analysis
        errors = [p.error for p in self.predictions if p.error]
        for error in errors:
            error_type = type(error).__name__ if not isinstance(error, str) else "Error"
            self.results.error_types[error_type] = self.results.error_types.get(error_type, 0) + 1

    def _log_progress(self) -> None:
        """Log shadow mode progress"""
        logger.info(
            f"Shadow progress: {self.results.total_predictions} predictions, "
            f"Agreement: {self.results.agreement_rate:.2%}, "
            f"Latency ratio: {self.results.latency_ratio:.2f}x, "
            f"Errors: {self.results.error_rate:.2%}"
        )

    def get_report(self) -> dict[str, Any]:
        """
        Generate comprehensive shadow mode report.

        Returns:
            Report dictionary
        """
        report = {
            "summary": self.results.to_dict(),
            "status": self.status.value,
            "predictions_analyzed": self.results.total_predictions,
            "duration_hours": self.results.duration_hours,
            "agreement": {
                "overall_rate": self.results.agreement_rate,
                "exact_matches": self.results.exact_matches,
                "threshold": self.config.agreement_threshold,
                "meets_threshold": self.results.agreement_rate >= self.config.agreement_threshold,
            },
            "performance": {
                "avg_prod_latency_ms": self.results.avg_prod_latency_ms,
                "avg_shadow_latency_ms": self.results.avg_shadow_latency_ms,
                "latency_ratio": self.results.latency_ratio,
                "shadow_slower": self.results.latency_ratio > 1.0,
            },
            "reliability": {
                "success_rate": 1 - self.results.error_rate,
                "error_rate": self.results.error_rate,
                "error_types": self.results.error_types,
                "total_errors": self.results.failed_predictions,
            },
            "confidence": {
                "avg_prod_confidence": self.results.avg_prod_confidence,
                "avg_shadow_confidence": self.results.avg_shadow_confidence,
                "correlation": self.results.confidence_correlation,
            },
            "recommendation": self._generate_recommendation(),
        }

        return report

    def _generate_recommendation(self) -> str:
        """Generate recommendation based on shadow results"""
        if self.results.total_predictions < self.config.min_predictions:
            return "Insufficient data for recommendation"

        issues = []

        # Check agreement
        if self.results.agreement_rate < self.config.agreement_threshold:
            issues.append(f"Low agreement rate ({self.results.agreement_rate:.2%})")

        # Check latency
        if self.results.latency_ratio > 2.0:
            issues.append(f"High latency ({self.results.latency_ratio:.1f}x slower)")

        # Check errors
        if self.results.error_rate > self.config.max_error_rate:
            issues.append(f"High error rate ({self.results.error_rate:.2%})")

        if not issues:
            return "Shadow model ready for promotion"
        else:
            return f"Shadow model needs improvement: {', '.join(issues)}"

    def add_ground_truth(self, prediction_id: str, ground_truth: Any) -> None:
        """
        Add ground truth for accuracy calculation.

        Args:
            prediction_id: Prediction ID
            ground_truth: Actual value
        """
        # Find prediction
        for pred in self.predictions:
            if pred.prediction_id == prediction_id:
                pred.metadata["ground_truth"] = ground_truth

                # Calculate accuracy if possible
                if pred.prod_prediction is not None:
                    pred.metadata["prod_correct"] = pred.prod_prediction == ground_truth
                if pred.shadow_prediction is not None:
                    pred.metadata["shadow_correct"] = pred.shadow_prediction == ground_truth

                break

        # Update accuracy metrics
        self._update_accuracy_metrics()

    def _update_accuracy_metrics(self) -> None:
        """Update accuracy metrics based on ground truth"""
        prod_correct = []
        shadow_correct = []

        for pred in self.predictions:
            if "prod_correct" in pred.metadata:
                prod_correct.append(pred.metadata["prod_correct"])
            if "shadow_correct" in pred.metadata:
                shadow_correct.append(pred.metadata["shadow_correct"])

        if prod_correct:
            self.results.prod_accuracy = sum(prod_correct) / len(prod_correct)
        if shadow_correct:
            self.results.shadow_accuracy = sum(shadow_correct) / len(shadow_correct)


def demonstrate_shadow_mode():
    """Demonstrate shadow mode functionality"""
    print("Shadow Mode Demonstration")
    print("=" * 60)

    # Mock model functions
    def production_model(features):
        """Mock production model"""
        # Simulate some processing
        time.sleep(0.001)  # 1ms latency
        prediction = features.get("value", 0) * 1.1
        confidence = 0.85
        return prediction, confidence

    def shadow_model(features):
        """Mock shadow model"""
        # Simulate slightly different model
        time.sleep(0.002)  # 2ms latency (slower)
        prediction = features.get("value", 0) * 1.12  # Slightly different
        confidence = 0.87
        return prediction, confidence

    # Create shadow mode
    config = ShadowConfig(
        model_id="prod_v1",
        shadow_model_id="shadow_v2",
        min_predictions=10,
        comparison_mode=ComparisonMode.THRESHOLD,
        value_threshold=0.05,  # 5% difference acceptable
        log_interval=5,
    )

    shadow = ShadowMode(production_model, shadow_model, config)

    # Start shadow mode
    shadow.start()
    print(f"Shadow mode started: {shadow.status.value}")
    print()

    # Make predictions
    print("Making predictions in shadow mode...")
    for i in range(15):
        features = {"value": i * 10}
        result = shadow.predict(features, f"pred_{i}")
        if i % 5 == 0:
            print(f"  Prediction {i}: {result}")

    # Wait a bit for async shadow predictions
    time.sleep(0.5)

    # Stop and get results
    print("\nStopping shadow mode...")
    results = shadow.stop()

    # Generate report
    report = shadow.get_report()

    print("\nShadow Mode Report:")
    print(f"  Total predictions: {report['predictions_analyzed']}")
    print(f"  Agreement rate: {report['agreement']['overall_rate']:.2%}")
    print(f"  Avg production latency: {report['performance']['avg_prod_latency_ms']:.2f}ms")
    print(f"  Avg shadow latency: {report['performance']['avg_shadow_latency_ms']:.2f}ms")
    print(f"  Latency ratio: {report['performance']['latency_ratio']:.2f}x")
    print(f"  Error rate: {report['reliability']['error_rate']:.2%}")
    print(f"  Recommendation: {report['recommendation']}")


if __name__ == "__main__":
    demonstrate_shadow_mode()
