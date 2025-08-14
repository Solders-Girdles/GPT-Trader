"""
Automatic Model Promotion and Rollback System
Phase 3, Week 2: MON-013, MON-014
Automated model lifecycle management with promotion and rollback capabilities
"""

import json
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ModelStage(Enum):
    """Model lifecycle stages"""

    DEVELOPMENT = "development"  # In development/training
    STAGING = "staging"  # Ready for testing
    SHADOW = "shadow"  # Running in shadow mode
    CANDIDATE = "candidate"  # Candidate for production
    PRODUCTION = "production"  # Active in production
    ARCHIVED = "archived"  # Retired model
    ROLLBACK = "rollback"  # Available for rollback


class PromotionCriteria(Enum):
    """Criteria for model promotion"""

    PERFORMANCE = "performance"  # Based on performance metrics
    TIME_BASED = "time_based"  # After certain duration
    MANUAL = "manual"  # Manual promotion
    AUTOMATIC = "automatic"  # Fully automatic
    THRESHOLD = "threshold"  # Performance threshold met
    UNANIMOUS = "unanimous"  # All criteria must pass


@dataclass
class ModelVersion:
    """Model version information"""

    model_id: str
    version: str
    stage: ModelStage
    created_at: datetime
    promoted_at: datetime | None = None

    # Performance metrics
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    auc_roc: float | None = None

    # Trading metrics
    sharpe_ratio: float | None = None
    max_drawdown: float | None = None
    win_rate: float | None = None
    profit_factor: float | None = None

    # Metadata
    training_data_start: datetime | None = None
    training_data_end: datetime | None = None
    n_features: int | None = None
    n_samples_train: int | None = None

    # Shadow mode results
    shadow_predictions: int = 0
    shadow_accuracy: float | None = None
    shadow_confidence: float | None = None

    # A/B test results
    ab_test_id: str | None = None
    ab_test_p_value: float | None = None
    ab_test_winner: bool | None = None

    # Model artifacts
    model_path: str | None = None
    config_path: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "model_id": self.model_id,
            "version": self.version,
            "stage": self.stage.value,
            "created_at": self.created_at.isoformat(),
            "promoted_at": self.promoted_at.isoformat() if self.promoted_at else None,
            "accuracy": self.accuracy,
            "sharpe_ratio": self.sharpe_ratio,
            "shadow_predictions": self.shadow_predictions,
            "shadow_accuracy": self.shadow_accuracy,
            "ab_test_winner": self.ab_test_winner,
        }


@dataclass
class PromotionConfig:
    """Configuration for automatic promotion"""

    # Performance thresholds
    min_accuracy: float = 0.55
    min_sharpe_ratio: float = 1.0
    max_drawdown: float = 0.20

    # Shadow mode requirements
    min_shadow_predictions: int = 1000
    min_shadow_accuracy: float = 0.54
    shadow_duration_hours: int = 24

    # A/B testing requirements
    require_ab_test: bool = True
    ab_test_confidence: float = 0.95
    ab_test_min_samples: int = 100

    # Promotion criteria
    criteria: PromotionCriteria = PromotionCriteria.AUTOMATIC
    unanimous_vote: bool = False  # All criteria must pass

    # Safety settings
    enable_auto_rollback: bool = True
    rollback_threshold: float = 0.05  # 5% performance drop
    rollback_window_hours: int = 6
    max_rollback_attempts: int = 3

    # Gradual rollout
    enable_gradual_rollout: bool = True
    initial_traffic_percent: float = 10.0
    traffic_increment: float = 10.0
    increment_interval_hours: int = 1


class ModelPromotion:
    """
    Automatic model promotion and rollback system.

    Features:
    - Automatic promotion based on criteria
    - Shadow mode validation
    - A/B test integration
    - Gradual rollout
    - Automatic rollback on degradation
    - Version history tracking
    """

    def __init__(
        self, config: PromotionConfig | None = None, model_registry_path: str = "models/registry"
    ):
        """
        Initialize promotion system.

        Args:
            config: Promotion configuration
            model_registry_path: Path to model registry
        """
        self.config = config or PromotionConfig()
        self.registry_path = Path(model_registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

        # Model tracking
        self.models: dict[str, ModelVersion] = {}
        self.production_model: ModelVersion | None = None
        self.rollback_history: deque = deque(maxlen=10)

        # Traffic routing for gradual rollout
        self.traffic_split: dict[str, float] = {}

        # Performance tracking
        self.performance_history: dict[str, list[dict]] = {}

        # Load existing registry
        self._load_registry()

    def register_model(self, model_version: ModelVersion) -> str:
        """
        Register a new model version.

        Args:
            model_version: Model version to register

        Returns:
            Registered model ID
        """
        model_id = model_version.model_id
        self.models[model_id] = model_version

        # Initialize performance tracking
        if model_id not in self.performance_history:
            self.performance_history[model_id] = []

        # Save registry
        self._save_registry()

        logger.info(f"Registered model {model_id} in stage {model_version.stage.value}")
        return model_id

    def check_promotion_eligibility(self, model_id: str) -> tuple[bool, list[str]]:
        """
        Check if model is eligible for promotion.

        Args:
            model_id: Model to check

        Returns:
            Tuple of (is_eligible, reasons)
        """
        if model_id not in self.models:
            return False, ["Model not found"]

        model = self.models[model_id]
        reasons = []
        checks_passed = []

        # Check performance metrics
        if self.config.min_accuracy and model.accuracy:
            if model.accuracy >= self.config.min_accuracy:
                checks_passed.append(f"Accuracy {model.accuracy:.3f} >= {self.config.min_accuracy}")
            else:
                reasons.append(f"Accuracy {model.accuracy:.3f} < {self.config.min_accuracy}")

        if self.config.min_sharpe_ratio and model.sharpe_ratio:
            if model.sharpe_ratio >= self.config.min_sharpe_ratio:
                checks_passed.append(
                    f"Sharpe {model.sharpe_ratio:.2f} >= {self.config.min_sharpe_ratio}"
                )
            else:
                reasons.append(f"Sharpe {model.sharpe_ratio:.2f} < {self.config.min_sharpe_ratio}")

        if self.config.max_drawdown and model.max_drawdown:
            if model.max_drawdown <= self.config.max_drawdown:
                checks_passed.append(
                    f"Drawdown {model.max_drawdown:.2%} <= {self.config.max_drawdown:.2%}"
                )
            else:
                reasons.append(
                    f"Drawdown {model.max_drawdown:.2%} > {self.config.max_drawdown:.2%}"
                )

        # Check shadow mode requirements
        if model.stage == ModelStage.SHADOW:
            if model.shadow_predictions >= self.config.min_shadow_predictions:
                checks_passed.append(
                    f"Shadow predictions {model.shadow_predictions} >= {self.config.min_shadow_predictions}"
                )
            else:
                reasons.append(
                    f"Insufficient shadow predictions: {model.shadow_predictions} < {self.config.min_shadow_predictions}"
                )

            if model.shadow_accuracy and model.shadow_accuracy >= self.config.min_shadow_accuracy:
                checks_passed.append(
                    f"Shadow accuracy {model.shadow_accuracy:.3f} >= {self.config.min_shadow_accuracy}"
                )
            else:
                reasons.append("Shadow accuracy below threshold")

        # Check A/B test results
        if self.config.require_ab_test:
            if model.ab_test_winner:
                checks_passed.append("Won A/B test")
            elif model.ab_test_p_value and model.ab_test_p_value < (
                1 - self.config.ab_test_confidence
            ):
                checks_passed.append(f"A/B test significant (p={model.ab_test_p_value:.4f})")
            else:
                reasons.append("Did not win A/B test or test not significant")

        # Determine eligibility based on criteria
        if self.config.criteria == PromotionCriteria.UNANIMOUS:
            is_eligible = len(reasons) == 0
        else:
            is_eligible = len(checks_passed) > len(reasons)

        return is_eligible, reasons if not is_eligible else checks_passed

    def promote_model(
        self, model_id: str, target_stage: ModelStage | None = None, force: bool = False
    ) -> bool:
        """
        Promote model to next stage.

        Args:
            model_id: Model to promote
            target_stage: Target stage (auto-determine if None)
            force: Force promotion regardless of criteria

        Returns:
            True if promotion successful
        """
        if model_id not in self.models:
            logger.error(f"Model {model_id} not found")
            return False

        model = self.models[model_id]
        current_stage = model.stage

        # Check eligibility unless forced
        if not force:
            is_eligible, reasons = self.check_promotion_eligibility(model_id)
            if not is_eligible:
                logger.warning(f"Model {model_id} not eligible for promotion: {reasons}")
                return False

        # Determine target stage
        if target_stage is None:
            target_stage = self._get_next_stage(current_stage)

        # Handle production promotion specially
        if target_stage == ModelStage.PRODUCTION:
            return self._promote_to_production(model_id)

        # Update model stage
        model.stage = target_stage
        model.promoted_at = datetime.now()

        # Save registry
        self._save_registry()

        logger.info(f"Promoted model {model_id} from {current_stage.value} to {target_stage.value}")
        return True

    def _promote_to_production(self, model_id: str) -> bool:
        """
        Promote model to production with safety checks.

        Args:
            model_id: Model to promote

        Returns:
            True if successful
        """
        new_model = self.models[model_id]

        # Archive current production model
        if self.production_model:
            old_model = self.production_model
            old_model.stage = ModelStage.ARCHIVED

            # Keep for rollback
            self.rollback_history.append(
                {
                    "model_id": old_model.model_id,
                    "timestamp": datetime.now(),
                    "metrics": {
                        "accuracy": old_model.accuracy,
                        "sharpe_ratio": old_model.sharpe_ratio,
                    },
                }
            )

        # Promote new model
        new_model.stage = ModelStage.PRODUCTION
        new_model.promoted_at = datetime.now()
        self.production_model = new_model

        # Setup gradual rollout if enabled
        if self.config.enable_gradual_rollout:
            self.traffic_split = {
                new_model.model_id: self.config.initial_traffic_percent / 100,
                "previous": 1 - (self.config.initial_traffic_percent / 100),
            }
            logger.info(
                f"Starting gradual rollout at {self.config.initial_traffic_percent}% traffic"
            )

        # Save registry
        self._save_registry()

        logger.info(f"Model {model_id} promoted to production")
        return True

    def rollback_model(self, target_model_id: str | None = None) -> bool:
        """
        Rollback to previous model.

        Args:
            target_model_id: Specific model to rollback to (latest if None)

        Returns:
            True if rollback successful
        """
        if not self.rollback_history:
            logger.error("No models available for rollback")
            return False

        # Get target model
        if target_model_id:
            rollback_entry = next(
                (entry for entry in self.rollback_history if entry["model_id"] == target_model_id),
                None,
            )
            if not rollback_entry:
                logger.error(f"Model {target_model_id} not found in rollback history")
                return False
        else:
            rollback_entry = self.rollback_history[-1]

        rollback_model_id = rollback_entry["model_id"]

        if rollback_model_id not in self.models:
            logger.error(f"Rollback model {rollback_model_id} not found")
            return False

        # Demote current production model
        if self.production_model:
            self.production_model.stage = ModelStage.ROLLBACK

        # Promote rollback model
        rollback_model = self.models[rollback_model_id]
        rollback_model.stage = ModelStage.PRODUCTION
        rollback_model.promoted_at = datetime.now()
        self.production_model = rollback_model

        # Reset traffic split
        self.traffic_split = {rollback_model_id: 1.0}

        # Save registry
        self._save_registry()

        logger.warning(f"Rolled back to model {rollback_model_id}")
        return True

    def check_for_degradation(self, model_id: str, current_metrics: dict[str, float]) -> bool:
        """
        Check if model performance has degraded.

        Args:
            model_id: Model to check
            current_metrics: Current performance metrics

        Returns:
            True if degradation detected
        """
        if model_id not in self.models:
            return False

        model = self.models[model_id]

        # Compare with baseline metrics
        degradation_detected = False

        if "accuracy" in current_metrics and model.accuracy:
            accuracy_drop = model.accuracy - current_metrics["accuracy"]
            if accuracy_drop > self.config.rollback_threshold:
                logger.warning(f"Accuracy degradation: {accuracy_drop:.3f}")
                degradation_detected = True

        if "sharpe_ratio" in current_metrics and model.sharpe_ratio:
            sharpe_drop = model.sharpe_ratio - current_metrics["sharpe_ratio"]
            if sharpe_drop > 0.5:  # Significant Sharpe drop
                logger.warning(f"Sharpe ratio degradation: {sharpe_drop:.2f}")
                degradation_detected = True

        # Track performance history
        self.performance_history[model_id].append(
            {
                "timestamp": datetime.now(),
                "metrics": current_metrics,
                "degradation": degradation_detected,
            }
        )

        # Auto-rollback if enabled and degradation detected
        if degradation_detected and self.config.enable_auto_rollback:
            if model.stage == ModelStage.PRODUCTION:
                logger.critical(f"Auto-rollback triggered for model {model_id}")
                self.rollback_model()

        return degradation_detected

    def update_traffic_split(self) -> dict[str, float]:
        """
        Update traffic split for gradual rollout.

        Returns:
            Updated traffic split
        """
        if not self.config.enable_gradual_rollout:
            return self.traffic_split

        if not self.production_model:
            return {}

        current_split = self.traffic_split.get(self.production_model.model_id, 0)

        # Check if we should increase traffic
        if current_split < 1.0:
            # Check promotion time
            hours_since_promotion = 0
            if self.production_model.promoted_at:
                hours_since_promotion = (
                    datetime.now() - self.production_model.promoted_at
                ).total_seconds() / 3600

            # Increase traffic if interval passed
            if hours_since_promotion >= self.config.increment_interval_hours:
                new_split = min(1.0, current_split + self.config.traffic_increment / 100)
                self.traffic_split[self.production_model.model_id] = new_split
                self.traffic_split["previous"] = 1 - new_split

                logger.info(f"Updated traffic split to {new_split:.0%} for production model")

        return self.traffic_split

    def get_model_for_prediction(self) -> str:
        """
        Get model ID for making predictions (considers traffic split).

        Returns:
            Model ID to use
        """
        if not self.production_model:
            # Return best candidate
            candidates = [
                m
                for m in self.models.values()
                if m.stage in [ModelStage.CANDIDATE, ModelStage.STAGING]
            ]
            if candidates:
                # Sort by accuracy
                candidates.sort(key=lambda x: x.accuracy or 0, reverse=True)
                return candidates[0].model_id
            return None

        # If gradual rollout enabled, use traffic split
        if self.config.enable_gradual_rollout and self.traffic_split:
            rand = np.random.random()
            cumsum = 0
            for model_id, split in self.traffic_split.items():
                cumsum += split
                if rand < cumsum:
                    return model_id if model_id != "previous" else self._get_previous_model()

        return self.production_model.model_id

    def _get_previous_model(self) -> str | None:
        """Get previous production model ID"""
        if self.rollback_history:
            return self.rollback_history[-1]["model_id"]
        return None

    def _get_next_stage(self, current_stage: ModelStage) -> ModelStage:
        """Determine next stage in promotion pipeline"""
        progression = {
            ModelStage.DEVELOPMENT: ModelStage.STAGING,
            ModelStage.STAGING: ModelStage.SHADOW,
            ModelStage.SHADOW: ModelStage.CANDIDATE,
            ModelStage.CANDIDATE: ModelStage.PRODUCTION,
            ModelStage.PRODUCTION: ModelStage.ARCHIVED,
            ModelStage.ARCHIVED: ModelStage.ARCHIVED,
            ModelStage.ROLLBACK: ModelStage.STAGING,
        }
        return progression.get(current_stage, ModelStage.STAGING)

    def _save_registry(self):
        """Save model registry to disk"""
        registry_file = self.registry_path / "registry.json"
        registry_data = {
            "models": {model_id: model.to_dict() for model_id, model in self.models.items()},
            "production_model": self.production_model.model_id if self.production_model else None,
            "traffic_split": self.traffic_split,
            "rollback_history": list(self.rollback_history),
        }

        with open(registry_file, "w") as f:
            json.dump(registry_data, f, indent=2, default=str)

    def _load_registry(self):
        """Load model registry from disk"""
        registry_file = self.registry_path / "registry.json"
        if not registry_file.exists():
            return

        try:
            with open(registry_file) as f:
                registry_data = json.load(f)

            # Reconstruct models
            for model_id, model_data in registry_data.get("models", {}).items():
                # Convert back to ModelVersion
                # This is simplified - in production would need proper deserialization
                pass

            # Load traffic split
            self.traffic_split = registry_data.get("traffic_split", {})

            # Load rollback history
            self.rollback_history = deque(registry_data.get("rollback_history", []), maxlen=10)

        except Exception as e:
            logger.error(f"Failed to load registry: {e}")

    def get_promotion_report(self) -> dict[str, Any]:
        """
        Generate promotion status report.

        Returns:
            Report dictionary
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "production_model": self.production_model.model_id if self.production_model else None,
            "models_by_stage": {},
            "traffic_split": self.traffic_split,
            "rollback_available": len(self.rollback_history) > 0,
        }

        # Group models by stage
        for stage in ModelStage:
            models_in_stage = [m.model_id for m in self.models.values() if m.stage == stage]
            if models_in_stage:
                report["models_by_stage"][stage.value] = models_in_stage

        # Check promotion eligibility for candidates
        candidates = [
            m for m in self.models.values() if m.stage in [ModelStage.CANDIDATE, ModelStage.SHADOW]
        ]

        report["promotion_candidates"] = []
        for candidate in candidates:
            is_eligible, reasons = self.check_promotion_eligibility(candidate.model_id)
            report["promotion_candidates"].append(
                {
                    "model_id": candidate.model_id,
                    "stage": candidate.stage.value,
                    "eligible": is_eligible,
                    "reasons": reasons,
                }
            )

        return report


def demonstrate_model_promotion():
    """Demonstrate model promotion system"""
    print("Model Promotion System Demo")
    print("=" * 60)

    # Create promotion system
    config = PromotionConfig(
        min_accuracy=0.55,
        min_sharpe_ratio=1.0,
        enable_gradual_rollout=True,
        enable_auto_rollback=True,
    )
    promotion = ModelPromotion(config)

    # Register development model
    model_v1 = ModelVersion(
        model_id="model_v1",
        version="1.0.0",
        stage=ModelStage.DEVELOPMENT,
        created_at=datetime.now(),
        accuracy=0.52,
        sharpe_ratio=0.8,
    )
    promotion.register_model(model_v1)

    # Check eligibility (should fail)
    is_eligible, reasons = promotion.check_promotion_eligibility("model_v1")
    print(f"Model v1 eligible: {is_eligible}")
    print(f"Reasons: {reasons}")
    print()

    # Register better model
    model_v2 = ModelVersion(
        model_id="model_v2",
        version="2.0.0",
        stage=ModelStage.SHADOW,
        created_at=datetime.now(),
        accuracy=0.58,
        sharpe_ratio=1.2,
        shadow_predictions=1500,
        shadow_accuracy=0.57,
        ab_test_winner=True,
    )
    promotion.register_model(model_v2)

    # Check eligibility (should pass)
    is_eligible, reasons = promotion.check_promotion_eligibility("model_v2")
    print(f"Model v2 eligible: {is_eligible}")
    print(f"Reasons: {reasons}")
    print()

    # Promote to production
    success = promotion.promote_model("model_v2", ModelStage.PRODUCTION)
    print(f"Promotion successful: {success}")
    print(f"Traffic split: {promotion.traffic_split}")
    print()

    # Simulate degradation
    print("Simulating performance degradation...")
    degraded = promotion.check_for_degradation("model_v2", {"accuracy": 0.50, "sharpe_ratio": 0.9})
    print(f"Degradation detected: {degraded}")
    print()

    # Get promotion report
    report = promotion.get_promotion_report()
    print("Promotion Report:")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    demonstrate_model_promotion()
