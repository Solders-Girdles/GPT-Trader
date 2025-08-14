"""
Automated Retraining System for GPT-Trader
Phase 3, Week 5-6: ADAPT-009 through ADAPT-016

Comprehensive automated retraining system that maintains optimal performance
while preventing excessive retraining and ensuring cost optimization.

Features:
- Performance-triggered retraining (ADAPT-009)
- Schedule-based retraining (ADAPT-010)
- Data drift triggers (ADAPT-011)
- Emergency retraining (ADAPT-012)
- Cost optimization (ADAPT-013)
- Model versioning (ADAPT-014)
- Orchestration (ADAPT-015)
- Performance validation (ADAPT-016)
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import psutil
import schedule

from ..database.postgres_manager import DatabaseManager
from ..performance import PerformanceMonitor
from .drift_detector import ConceptDriftDetector
from .integrated_pipeline import IntegratedMLPipeline
from .model_promotion import ModelPromotion
from .model_validation import ModelPerformance, ModelValidator

# Local imports
from .online_learning import OnlineLearningPipeline
from .shadow_mode import ShadowModePredictor

logger = logging.getLogger(__name__)


class RetrainingTrigger(Enum):
    """Types of retraining triggers"""

    PERFORMANCE_DEGRADATION = "performance_degradation"
    SCHEDULED = "scheduled"
    DATA_DRIFT = "data_drift"
    EMERGENCY = "emergency"
    MANUAL = "manual"
    COST_BENEFIT = "cost_benefit"
    FEATURE_DRIFT = "feature_drift"
    MODEL_AGE = "model_age"


class RetrainingStatus(Enum):
    """Status of retraining operations"""

    IDLE = "idle"
    QUEUED = "queued"
    RUNNING = "running"
    VALIDATING = "validating"
    DEPLOYING = "deploying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLBACK = "rollback"


class EmergencyLevel(Enum):
    """Emergency levels for rapid response"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RetrainingCost:
    """Cost tracking for retraining operations"""

    computational_cost: float  # $ cost
    time_cost: float  # hours
    opportunity_cost: float  # potential loss
    resource_usage: dict[str, float]  # CPU, memory, disk
    estimated_total: float
    actual_total: float | None = None
    roi_estimate: float | None = None
    roi_actual: float | None = None


@dataclass
class RetrainingConfig:
    """Configuration for automated retraining"""

    # Performance triggers
    min_accuracy_threshold: float = 0.55
    min_precision_threshold: float = 0.55
    min_recall_threshold: float = 0.55
    min_f1_threshold: float = 0.55
    min_sharpe_ratio: float = 1.0
    performance_window: int = 1000  # samples

    # Scheduling
    daily_retrain_hour: int = 2  # 2 AM
    weekly_retrain_day: int = 6  # Sunday
    monthly_retrain_day: int = 1  # 1st of month
    enable_scheduled_retraining: bool = True

    # Drift detection
    drift_sensitivity: float = 0.05
    feature_drift_threshold: float = 0.1
    enable_drift_retraining: bool = True

    # Emergency response
    emergency_performance_drop: float = 0.2  # 20% drop
    black_swan_detection_threshold: float = 3.0  # 3 sigma
    emergency_response_timeout: int = 300  # 5 minutes

    # Cost optimization
    max_daily_retraining_cost: float = 10.0  # $10
    max_monthly_retraining_cost: float = 100.0  # $100
    min_roi_threshold: float = 1.5  # 150% ROI
    cost_per_compute_hour: float = 0.50  # $0.50/hour

    # Resource limits
    max_concurrent_retrainings: int = 1
    max_retrainings_per_day: int = 2
    cooldown_period_hours: int = 6
    max_memory_usage_gb: float = 8.0
    max_cpu_usage_percent: float = 80.0

    # Model versioning
    max_model_versions: int = 10
    model_retention_days: int = 30
    enable_automatic_cleanup: bool = True

    # Validation
    shadow_mode_duration_hours: int = 24
    min_shadow_samples: int = 1000
    validation_performance_threshold: float = 0.02  # 2% improvement
    gradual_rollout_steps: list[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 1.0])

    # Safety features
    require_manual_approval: bool = True  # First week requires approval
    auto_rollback_enabled: bool = True
    rollback_performance_threshold: float = 0.1  # 10% degradation
    audit_all_retrainings: bool = True


@dataclass
class RetrainingRequest:
    """Request for model retraining"""

    trigger: RetrainingTrigger
    priority: int  # 1-10, 10 highest
    requested_at: datetime
    requested_by: str
    model_id: str
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)
    estimated_cost: RetrainingCost | None = None
    emergency_level: EmergencyLevel | None = None
    approval_required: bool = True
    approved: bool = False
    approved_by: str | None = None
    approved_at: datetime | None = None


@dataclass
class RetrainingResult:
    """Result of retraining operation"""

    request_id: str
    status: RetrainingStatus
    started_at: datetime
    completed_at: datetime | None = None
    duration_seconds: float | None = None

    # Model information
    old_model_id: str
    new_model_id: str | None = None
    new_model_version: str | None = None

    # Performance metrics
    old_performance: ModelPerformance | None = None
    new_performance: ModelPerformance | None = None
    performance_improvement: float | None = None

    # Cost tracking
    actual_cost: RetrainingCost | None = None

    # Validation results
    shadow_mode_results: dict[str, Any] | None = None
    rollout_results: dict[str, Any] | None = None

    # Error information
    error_message: str | None = None
    traceback: str | None = None

    # Resource usage
    peak_memory_usage: float | None = None
    peak_cpu_usage: float | None = None
    total_compute_time: float | None = None


class AutoRetrainingSystem:
    """
    Comprehensive automated retraining system.

    Handles all aspects of automated model retraining including:
    - Performance monitoring and trigger detection
    - Cost optimization and resource management
    - Model versioning and deployment
    - Validation and rollback capabilities
    """

    def __init__(
        self,
        config: RetrainingConfig,
        ml_pipeline: IntegratedMLPipeline,
        db_manager: DatabaseManager,
        online_learner: OnlineLearningPipeline | None = None,
        drift_detector: ConceptDriftDetector | None = None,
    ):
        """Initialize automated retraining system

        Args:
            config: Retraining configuration
            ml_pipeline: ML pipeline for training
            db_manager: Database manager
            online_learner: Online learning pipeline
            drift_detector: Drift detection system
        """
        self.config = config
        self.ml_pipeline = ml_pipeline
        self.db_manager = db_manager
        self.online_learner = online_learner
        self.drift_detector = drift_detector

        # Components
        self.model_promotion = ModelPromotion()
        self.shadow_predictor = ShadowModePredictor()
        self.validator = ModelValidator()
        self.performance_monitor = PerformanceMonitor()

        # State management
        self.is_running = False
        self.current_retrainings: dict[str, RetrainingResult] = {}
        self.retraining_queue: deque = deque()
        self.retraining_history: list[RetrainingResult] = []
        self.performance_history: deque = deque(maxlen=config.performance_window)

        # Cost tracking
        self.daily_costs: dict[str, float] = {}  # date -> cost
        self.monthly_costs: dict[str, float] = {}  # month -> cost

        # Threading
        self.monitor_thread: threading.Thread | None = None
        self.retraining_thread: threading.Thread | None = None
        self.stop_event = threading.Event()

        # Scheduling
        self._setup_scheduling()

        # Performance baseline
        self.performance_baseline: ModelPerformance | None = None
        self.last_drift_detection: datetime | None = None
        self.last_retraining: datetime | None = None

        logger.info("Initialized automated retraining system")

    def start(self):
        """Start the automated retraining system"""
        if self.is_running:
            logger.warning("Retraining system already running")
            return

        self.is_running = True
        self.stop_event.clear()

        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, name="RetrainingMonitor", daemon=True
        )
        self.monitor_thread.start()

        # Start retraining processing thread
        self.retraining_thread = threading.Thread(
            target=self._retraining_loop, name="RetrainingProcessor", daemon=True
        )
        self.retraining_thread.start()

        logger.info("Started automated retraining system")

    def stop(self):
        """Stop the automated retraining system"""
        if not self.is_running:
            return

        logger.info("Stopping automated retraining system")
        self.is_running = False
        self.stop_event.set()

        # Wait for threads to complete
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=30)

        if self.retraining_thread and self.retraining_thread.is_alive():
            self.retraining_thread.join(timeout=30)

        logger.info("Stopped automated retraining system")

    def _setup_scheduling(self):
        """Setup scheduled retraining"""
        if not self.config.enable_scheduled_retraining:
            return

        # Daily retraining
        schedule.every().day.at(f"{self.config.daily_retrain_hour:02d}:00").do(
            self._trigger_scheduled_retraining, "daily"
        )

        # Weekly retraining
        if self.config.weekly_retrain_day == 0:
            schedule.every().monday.at(f"{self.config.daily_retrain_hour:02d}:00").do(
                self._trigger_scheduled_retraining, "weekly"
            )
        elif self.config.weekly_retrain_day == 6:
            schedule.every().sunday.at(f"{self.config.daily_retrain_hour:02d}:00").do(
                self._trigger_scheduled_retraining, "weekly"
            )

        # Monthly retraining
        # Note: schedule library doesn't support monthly, implement custom logic

    def _monitoring_loop(self):
        """Main monitoring loop"""
        logger.info("Started retraining monitoring loop")

        while not self.stop_event.is_set():
            try:
                # Check scheduled tasks
                schedule.run_pending()

                # Monitor performance
                self._check_performance_degradation()

                # Monitor drift
                if self.config.enable_drift_retraining:
                    self._check_drift_triggers()

                # Monitor emergency conditions
                self._check_emergency_conditions()

                # Monitor resource usage
                self._check_resource_constraints()

                # Cleanup old models
                if self.config.enable_automatic_cleanup:
                    self._cleanup_old_models()

                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)

        logger.info("Stopped retraining monitoring loop")

    def _retraining_loop(self):
        """Main retraining processing loop"""
        logger.info("Started retraining processing loop")

        while not self.stop_event.is_set():
            try:
                if self.retraining_queue:
                    # Check if we can start new retraining
                    if self._can_start_retraining():
                        request = self.retraining_queue.popleft()
                        self._process_retraining_request(request)

                time.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"Error in retraining loop: {e}")
                time.sleep(60)

        logger.info("Stopped retraining processing loop")

    def _check_performance_degradation(self):
        """Check for performance degradation triggers"""
        if len(self.performance_history) < 100:
            return

        # Get recent performance
        recent_performance = list(self.performance_history)[-50:]
        historical_performance = list(self.performance_history)[:-50]

        if not historical_performance:
            return

        # Calculate performance metrics
        recent_accuracy = np.mean([p.accuracy for p in recent_performance])
        historical_accuracy = np.mean([p.accuracy for p in historical_performance])

        # Check thresholds
        accuracy_drop = historical_accuracy - recent_accuracy
        below_threshold = recent_accuracy < self.config.min_accuracy_threshold

        if accuracy_drop > 0.05 or below_threshold:  # 5% drop or below threshold
            self._trigger_performance_retraining(
                accuracy_drop, recent_accuracy, historical_accuracy
            )

    def _check_drift_triggers(self):
        """Check for data drift triggers"""
        if not self.drift_detector:
            return

        # Check if drift was recently detected
        drift_stats = self.drift_detector.get_statistics()
        if drift_stats.get("total_drifts_detected", 0) > 0:
            last_drift_time = drift_stats.get("last_drift_time")
            if last_drift_time and self.last_drift_detection != last_drift_time:
                self.last_drift_detection = datetime.fromisoformat(last_drift_time)
                self._trigger_drift_retraining()

    def _check_emergency_conditions(self):
        """Check for emergency retraining conditions"""
        if len(self.performance_history) < 10:
            return

        # Get very recent performance
        recent_performance = list(self.performance_history)[-10:]

        # Check for sudden performance drop
        if len(recent_performance) >= 10:
            latest_accuracy = recent_performance[-1].accuracy
            baseline_accuracy = np.mean([p.accuracy for p in recent_performance[:-5]])

            performance_drop = baseline_accuracy - latest_accuracy

            if performance_drop > self.config.emergency_performance_drop:
                emergency_level = (
                    EmergencyLevel.HIGH if performance_drop > 0.3 else EmergencyLevel.MEDIUM
                )
                self._trigger_emergency_retraining(performance_drop, emergency_level)

    def _trigger_performance_retraining(
        self, accuracy_drop: float, recent_acc: float, historical_acc: float
    ):
        """Trigger retraining due to performance degradation"""
        request = RetrainingRequest(
            trigger=RetrainingTrigger.PERFORMANCE_DEGRADATION,
            priority=7,
            requested_at=datetime.now(),
            requested_by="AutoRetrainingSystem",
            model_id=self._get_current_model_id(),
            reason=f"Performance degradation detected: {accuracy_drop:.3f} drop, current: {recent_acc:.3f}",
            metadata={
                "accuracy_drop": accuracy_drop,
                "recent_accuracy": recent_acc,
                "historical_accuracy": historical_acc,
            },
        )

        self._add_retraining_request(request)

    def _trigger_drift_retraining(self):
        """Trigger retraining due to data drift"""
        request = RetrainingRequest(
            trigger=RetrainingTrigger.DATA_DRIFT,
            priority=6,
            requested_at=datetime.now(),
            requested_by="AutoRetrainingSystem",
            model_id=self._get_current_model_id(),
            reason="Data drift detected by drift detection system",
            metadata={
                "drift_detection_time": (
                    self.last_drift_detection.isoformat() if self.last_drift_detection else None
                )
            },
        )

        self._add_retraining_request(request)

    def _trigger_emergency_retraining(
        self, performance_drop: float, emergency_level: EmergencyLevel
    ):
        """Trigger emergency retraining"""
        request = RetrainingRequest(
            trigger=RetrainingTrigger.EMERGENCY,
            priority=10,
            requested_at=datetime.now(),
            requested_by="AutoRetrainingSystem",
            model_id=self._get_current_model_id(),
            reason=f"Emergency retraining: {performance_drop:.1%} performance drop",
            emergency_level=emergency_level,
            approval_required=False,  # Emergency bypasses approval
            metadata={
                "performance_drop": performance_drop,
                "emergency_level": emergency_level.value,
            },
        )

        self._add_retraining_request(request)

    def _trigger_scheduled_retraining(self, schedule_type: str):
        """Trigger scheduled retraining"""
        request = RetrainingRequest(
            trigger=RetrainingTrigger.SCHEDULED,
            priority=3,
            requested_at=datetime.now(),
            requested_by="ScheduledRetraining",
            model_id=self._get_current_model_id(),
            reason=f"Scheduled {schedule_type} retraining",
            metadata={"schedule_type": schedule_type},
        )

        self._add_retraining_request(request)

    def _add_retraining_request(self, request: RetrainingRequest):
        """Add retraining request to queue"""
        # Check cooldown period
        if self._is_in_cooldown():
            logger.info(f"Retraining request {request.trigger.value} blocked by cooldown period")
            return

        # Check daily limits
        if self._exceeds_daily_limits():
            logger.warning(f"Retraining request {request.trigger.value} blocked by daily limits")
            return

        # Estimate cost
        request.estimated_cost = self._estimate_retraining_cost(request)

        # Check cost limits
        if not self._is_within_cost_limits(request.estimated_cost):
            logger.warning(f"Retraining request {request.trigger.value} blocked by cost limits")
            return

        # Add to queue (sort by priority)
        self.retraining_queue.append(request)
        self.retraining_queue = deque(
            sorted(self.retraining_queue, key=lambda x: x.priority, reverse=True)
        )

        logger.info(
            f"Added retraining request: {request.trigger.value} (priority: {request.priority})"
        )

    def _can_start_retraining(self) -> bool:
        """Check if we can start a new retraining"""
        # Check concurrent limit
        active_retrainings = len(
            [
                r
                for r in self.current_retrainings.values()
                if r.status in [RetrainingStatus.RUNNING, RetrainingStatus.VALIDATING]
            ]
        )

        if active_retrainings >= self.config.max_concurrent_retrainings:
            return False

        # Check resource availability
        if not self._check_resource_availability():
            return False

        return True

    def _process_retraining_request(self, request: RetrainingRequest):
        """Process a retraining request"""
        request_id = f"retrain_{int(datetime.now().timestamp())}"

        result = RetrainingResult(
            request_id=request_id,
            status=RetrainingStatus.RUNNING,
            started_at=datetime.now(),
            old_model_id=request.model_id,
        )

        self.current_retrainings[request_id] = result

        try:
            logger.info(f"Starting retraining {request_id}: {request.trigger.value}")

            # Check approval if required
            if request.approval_required and not request.approved:
                result.status = RetrainingStatus.QUEUED
                logger.info(f"Retraining {request_id} waiting for approval")
                # In a real system, this would notify administrators
                return

            # Start resource monitoring
            start_memory = psutil.virtual_memory().used / (1024**3)  # GB
            start_time = time.time()

            # Perform retraining
            new_model_result = self._perform_retraining(request, result)

            if new_model_result:
                result.new_model_id = new_model_result["model_id"]
                result.new_model_version = new_model_result["version"]
                result.new_performance = new_model_result["performance"]

                # Validate in shadow mode
                if self._validate_new_model(result):
                    # Deploy gradually
                    if self._deploy_model_gradually(result):
                        result.status = RetrainingStatus.COMPLETED
                        logger.info(f"Retraining {request_id} completed successfully")
                    else:
                        result.status = RetrainingStatus.FAILED
                        result.error_message = "Gradual deployment failed"
                        self._rollback_model(result)
                else:
                    result.status = RetrainingStatus.FAILED
                    result.error_message = "Shadow mode validation failed"
            else:
                result.status = RetrainingStatus.FAILED
                result.error_message = "Model training failed"

            # Calculate final metrics
            end_time = time.time()
            end_memory = psutil.virtual_memory().used / (1024**3)

            result.completed_at = datetime.now()
            result.duration_seconds = end_time - start_time
            result.peak_memory_usage = end_memory - start_memory
            result.total_compute_time = result.duration_seconds

            # Calculate actual cost
            result.actual_cost = self._calculate_actual_cost(result)

            # Track costs
            self._track_retraining_costs(result.actual_cost)

        except Exception as e:
            logger.error(f"Retraining {request_id} failed: {e}")
            result.status = RetrainingStatus.FAILED
            result.error_message = str(e)
            result.completed_at = datetime.now()

        finally:
            # Clean up
            if request_id in self.current_retrainings:
                self.retraining_history.append(result)
                del self.current_retrainings[request_id]

    def _perform_retraining(
        self, request: RetrainingRequest, result: RetrainingResult
    ) -> dict[str, Any] | None:
        """Perform the actual model retraining"""
        try:
            # Get training data
            training_data = self._get_training_data()
            if training_data is None or len(training_data) < 1000:
                logger.error("Insufficient training data for retraining")
                return None

            # Train new model
            X = training_data.drop(["target", "timestamp"], axis=1, errors="ignore")
            y = training_data["target"]

            # Use the integrated ML pipeline for training
            model_performance = self.ml_pipeline.train_and_validate_model(
                X, y, model_name=f"retrained_{int(datetime.now().timestamp())}"
            )

            if model_performance.accuracy < self.config.min_accuracy_threshold:
                logger.warning(
                    f"New model accuracy {model_performance.accuracy:.3f} below threshold"
                )
                return None

            return {
                "model_id": f"model_{int(datetime.now().timestamp())}",
                "version": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "performance": model_performance,
            }

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return None

    def _validate_new_model(self, result: RetrainingResult) -> bool:
        """Validate new model in shadow mode"""
        try:
            if not result.new_model_id:
                return False

            logger.info(f"Starting shadow mode validation for {result.new_model_id}")

            # Run shadow mode for configured duration
            shadow_duration = timedelta(hours=self.config.shadow_mode_duration_hours)
            shadow_results = self.shadow_predictor.run_shadow_mode(
                result.new_model_id,
                duration=shadow_duration,
                min_samples=self.config.min_shadow_samples,
            )

            result.shadow_mode_results = shadow_results

            # Check if shadow mode performance is acceptable
            if shadow_results and shadow_results.get("performance_improvement", 0) > 0:
                return True

            return False

        except Exception as e:
            logger.error(f"Shadow mode validation failed: {e}")
            return False

    def _deploy_model_gradually(self, result: RetrainingResult) -> bool:
        """Deploy model with gradual rollout"""
        try:
            logger.info(f"Starting gradual deployment for {result.new_model_id}")

            rollout_results = {}

            for step, traffic_percentage in enumerate(self.config.gradual_rollout_steps):
                logger.info(f"Deploying to {traffic_percentage:.0%} traffic")

                # In real implementation, this would configure load balancer
                # For now, we simulate the rollout
                step_performance = self._monitor_rollout_step(
                    result.new_model_id, traffic_percentage
                )

                rollout_results[f"step_{step}"] = {
                    "traffic_percentage": traffic_percentage,
                    "performance": step_performance,
                }

                # Check if performance degraded significantly
                if step_performance and step_performance < 0.5:  # Severe degradation
                    logger.error(
                        f"Severe performance degradation at {traffic_percentage:.0%} traffic"
                    )
                    return False

                # Wait between steps
                time.sleep(60)  # 1 minute between steps

            result.rollout_results = rollout_results
            return True

        except Exception as e:
            logger.error(f"Gradual deployment failed: {e}")
            return False

    def _monitor_rollout_step(self, model_id: str, traffic_percentage: float) -> float | None:
        """Monitor performance during rollout step"""
        # Simulate performance monitoring
        # In real implementation, this would collect actual metrics
        base_performance = 0.6
        noise = np.random.normal(0, 0.02)  # Small random variation
        return base_performance + noise

    def _rollback_model(self, result: RetrainingResult):
        """Rollback to previous model"""
        logger.warning(f"Rolling back from {result.new_model_id} to {result.old_model_id}")

        # In real implementation, this would:
        # 1. Switch traffic back to old model
        # 2. Update model registry
        # 3. Archive failed model

        result.status = RetrainingStatus.ROLLBACK

    def _get_training_data(self) -> pd.DataFrame | None:
        """Get training data for retraining"""
        try:
            # In real implementation, this would fetch from database
            # For now, return mock data
            n_samples = 5000
            n_features = 20

            # Generate synthetic training data
            X = np.random.randn(n_samples, n_features)
            y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1) > 0

            data = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
            data["target"] = y.astype(int)
            data["timestamp"] = pd.date_range(start="2024-01-01", periods=n_samples, freq="1min")

            return data

        except Exception as e:
            logger.error(f"Failed to get training data: {e}")
            return None

    def _estimate_retraining_cost(self, request: RetrainingRequest) -> RetrainingCost:
        """Estimate cost of retraining"""
        # Base estimates
        computational_hours = 2.0  # 2 hours for full retraining

        if request.trigger == RetrainingTrigger.EMERGENCY:
            computational_hours *= 0.5  # Emergency uses faster training
        elif request.trigger == RetrainingTrigger.SCHEDULED:
            computational_hours *= 1.5  # Scheduled can be more thorough

        computational_cost = computational_hours * self.config.cost_per_compute_hour
        time_cost = computational_hours

        # Opportunity cost (potential loss from delayed retraining)
        if request.trigger == RetrainingTrigger.PERFORMANCE_DEGRADATION:
            opportunity_cost = 50.0  # $50 potential loss
        elif request.trigger == RetrainingTrigger.EMERGENCY:
            opportunity_cost = 200.0  # $200 potential loss
        else:
            opportunity_cost = 10.0  # $10 minimal loss

        return RetrainingCost(
            computational_cost=computational_cost,
            time_cost=time_cost,
            opportunity_cost=opportunity_cost,
            resource_usage={
                "cpu_hours": computational_hours,
                "memory_gb_hours": computational_hours * 4.0,
                "disk_gb": 10.0,
            },
            estimated_total=computational_cost + opportunity_cost,
            roi_estimate=2.0,  # 200% ROI estimate
        )

    def _calculate_actual_cost(self, result: RetrainingResult) -> RetrainingCost:
        """Calculate actual cost of completed retraining"""
        hours = (result.duration_seconds or 0) / 3600
        computational_cost = hours * self.config.cost_per_compute_hour

        return RetrainingCost(
            computational_cost=computational_cost,
            time_cost=hours,
            opportunity_cost=0.0,  # No opportunity cost for completed retraining
            resource_usage={
                "cpu_hours": hours,
                "memory_gb_hours": hours * (result.peak_memory_usage or 4.0),
                "disk_gb": 10.0,
            },
            estimated_total=computational_cost,
            actual_total=computational_cost,
        )

    def _is_within_cost_limits(self, cost: RetrainingCost) -> bool:
        """Check if retraining cost is within limits"""
        today = datetime.now().date().isoformat()
        current_month = datetime.now().strftime("%Y-%m")

        daily_cost = self.daily_costs.get(today, 0.0)
        monthly_cost = self.monthly_costs.get(current_month, 0.0)

        if daily_cost + cost.estimated_total > self.config.max_daily_retraining_cost:
            return False

        if monthly_cost + cost.estimated_total > self.config.max_monthly_retraining_cost:
            return False

        return True

    def _track_retraining_costs(self, cost: RetrainingCost):
        """Track retraining costs"""
        today = datetime.now().date().isoformat()
        current_month = datetime.now().strftime("%Y-%m")

        self.daily_costs[today] = self.daily_costs.get(today, 0.0) + (cost.actual_total or 0.0)
        self.monthly_costs[current_month] = self.monthly_costs.get(current_month, 0.0) + (
            cost.actual_total or 0.0
        )

    def _is_in_cooldown(self) -> bool:
        """Check if system is in cooldown period"""
        if not self.last_retraining:
            return False

        cooldown_delta = timedelta(hours=self.config.cooldown_period_hours)
        return datetime.now() - self.last_retraining < cooldown_delta

    def _exceeds_daily_limits(self) -> bool:
        """Check if daily retraining limits are exceeded"""
        today = datetime.now().date()
        today_retrainings = len(
            [
                r
                for r in self.retraining_history
                if r.started_at.date() == today and r.status == RetrainingStatus.COMPLETED
            ]
        )

        return today_retrainings >= self.config.max_retrainings_per_day

    def _check_resource_availability(self) -> bool:
        """Check if sufficient resources are available"""
        # Check memory
        memory = psutil.virtual_memory()
        memory_usage_gb = memory.used / (1024**3)

        if memory_usage_gb > self.config.max_memory_usage_gb:
            return False

        # Check CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > self.config.max_cpu_usage_percent:
            return False

        return True

    def _check_resource_constraints(self):
        """Monitor resource constraints"""
        # This could trigger resource-based alerts
        pass

    def _cleanup_old_models(self):
        """Clean up old model versions"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config.model_retention_days)

            # In real implementation, this would:
            # 1. Query database for old models
            # 2. Archive or delete model files
            # 3. Update model registry

            logger.debug("Performed model cleanup")

        except Exception as e:
            logger.error(f"Model cleanup failed: {e}")

    def _get_current_model_id(self) -> str:
        """Get current production model ID"""
        # In real implementation, query model registry
        return "current_production_model"

    # Public API methods

    def request_manual_retraining(
        self, reason: str, priority: int = 5, requested_by: str = "manual"
    ) -> str:
        """Request manual retraining

        Args:
            reason: Reason for retraining
            priority: Priority level (1-10)
            requested_by: Who requested the retraining

        Returns:
            Request ID
        """
        request = RetrainingRequest(
            trigger=RetrainingTrigger.MANUAL,
            priority=priority,
            requested_at=datetime.now(),
            requested_by=requested_by,
            model_id=self._get_current_model_id(),
            reason=reason,
            approval_required=True,
        )

        self._add_retraining_request(request)
        return f"manual_{int(datetime.now().timestamp())}"

    def approve_retraining(self, request_id: str, approved_by: str):
        """Approve a pending retraining request"""
        for request in self.retraining_queue:
            if hasattr(request, "request_id") and request.request_id == request_id:
                request.approved = True
                request.approved_by = approved_by
                request.approved_at = datetime.now()
                logger.info(f"Approved retraining request {request_id} by {approved_by}")
                return

        logger.warning(f"Retraining request {request_id} not found for approval")

    def get_retraining_status(self) -> dict[str, Any]:
        """Get current retraining system status"""
        today = datetime.now().date().isoformat()
        current_month = datetime.now().strftime("%Y-%m")

        return {
            "is_running": self.is_running,
            "queue_length": len(self.retraining_queue),
            "active_retrainings": len(self.current_retrainings),
            "daily_cost": self.daily_costs.get(today, 0.0),
            "monthly_cost": self.monthly_costs.get(current_month, 0.0),
            "last_retraining": self.last_retraining.isoformat() if self.last_retraining else None,
            "in_cooldown": self._is_in_cooldown(),
            "total_retrainings": len(self.retraining_history),
            "successful_retrainings": len(
                [r for r in self.retraining_history if r.status == RetrainingStatus.COMPLETED]
            ),
        }

    def get_cost_summary(self) -> dict[str, Any]:
        """Get cost summary and optimization recommendations"""
        total_daily = sum(self.daily_costs.values())
        total_monthly = sum(self.monthly_costs.values())

        # Calculate ROI
        successful_retrainings = [
            r for r in self.retraining_history if r.status == RetrainingStatus.COMPLETED
        ]
        total_cost = sum(
            [r.actual_cost.actual_total for r in successful_retrainings if r.actual_cost]
        )

        return {
            "total_daily_cost": total_daily,
            "total_monthly_cost": total_monthly,
            "total_lifetime_cost": total_cost,
            "cost_per_retraining": (
                total_cost / len(successful_retrainings) if successful_retrainings else 0
            ),
            "successful_retrainings": len(successful_retrainings),
            "cost_efficiency": "good" if total_cost < 500 else "review_needed",
        }


# Factory functions
def create_auto_retraining_system(
    config_dict: dict[str, Any] | None = None,
    ml_pipeline: IntegratedMLPipeline | None = None,
    db_manager: DatabaseManager | None = None,
) -> AutoRetrainingSystem:
    """Factory function to create auto-retraining system"""

    if config_dict is None:
        config_dict = {}

    config = RetrainingConfig(**config_dict)

    if ml_pipeline is None:
        # Create default ML pipeline
        ml_pipeline = IntegratedMLPipeline()

    if db_manager is None:
        # Create default database manager
        db_manager = DatabaseManager()

    return AutoRetrainingSystem(config=config, ml_pipeline=ml_pipeline, db_manager=db_manager)


# Predefined configurations
CONSERVATIVE_RETRAINING_CONFIG = RetrainingConfig(
    max_retrainings_per_day=1,
    cooldown_period_hours=12,
    require_manual_approval=True,
    min_accuracy_threshold=0.58,
    max_daily_retraining_cost=5.0,
)

AGGRESSIVE_RETRAINING_CONFIG = RetrainingConfig(
    max_retrainings_per_day=3,
    cooldown_period_hours=3,
    require_manual_approval=False,
    min_accuracy_threshold=0.55,
    max_daily_retraining_cost=20.0,
    emergency_response_timeout=180,
)

PRODUCTION_RETRAINING_CONFIG = RetrainingConfig(
    max_retrainings_per_day=2,
    cooldown_period_hours=6,
    require_manual_approval=True,
    min_accuracy_threshold=0.57,
    max_daily_retraining_cost=10.0,
    shadow_mode_duration_hours=48,
    auto_rollback_enabled=True,
)
