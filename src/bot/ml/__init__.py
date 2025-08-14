"""
Machine Learning components for GPT-Trader
Phase 2.5 - Enhanced ML Pipeline
Phase 4 - Deep Learning Components
"""

# Phase 3 - Week 5-6: Automated Retraining System (ADAPT-009 to ADAPT-016)
from .auto_retraining import (
    AGGRESSIVE_RETRAINING_CONFIG,
    CONSERVATIVE_RETRAINING_CONFIG,
    PRODUCTION_RETRAINING_CONFIG,
    AutoRetrainingSystem,
    EmergencyLevel,
    RetrainingConfig,
    RetrainingCost,
    RetrainingRequest,
    RetrainingResult,
    RetrainingStatus,
    create_auto_retraining_system,
)
from .auto_retraining import (
    RetrainingTrigger as AutoRetrainingTrigger,
)
from .base import FeatureEngineer, MLModel, ModelRegistry

# Phase 4 - Deep Learning Components (DL-001 to DL-004)
from .deep_learning import (
    AttentionConfig,
    AttentionMechanism,
    AttentionType,
    LSTMArchitecture,
    LSTMConfig,
    LSTMDataPipeline,
    LSTMTrainingFramework,
    ScalingMethod,
    SequenceConfig,
    TaskType,
    TrainingConfig,
    TrainingResults,
    create_integrated_lstm_pipeline,
)

# Phase 2.5 components
from .feature_engineering_v2 import FeatureConfig, OptimizedFeatureEngineer
from .feature_selector import AdvancedFeatureSelector, FeatureSelectionConfig, SelectionMethod
from .integrated_pipeline import IntegratedMLPipeline
from .model_degradation_monitor import (
    DegradationMetrics,
    DegradationStatus,
    DegradationType,
    ModelDegradationMonitor,
    RetrainingTrigger,
    create_degradation_monitor,
)
from .model_validation import ModelPerformance, ModelValidator, ValidationConfig
from .model_versioning import (
    ModelFormat,
    ModelMetadata,
    ModelStage,
    ModelVersioning,
    VersionComparisonResult,
    VersionType,
    create_model_versioning,
)
from .performance_tracker import ModelHealth, ModelStatus, PerformanceMetrics, PerformanceTracker
from .retraining_scheduler import (
    DAILY_RETRAINING_SCHEDULE,
    PERFORMANCE_ADAPTIVE_SCHEDULE,
    WEEKLY_RETRAINING_SCHEDULE,
    ExecutionResult,
    RetrainingScheduler,
    ScheduleConfig,
    ScheduledTask,
    ScheduleType,
    TaskPriority,
    TaskStatus,
    create_adaptive_schedule,
    create_cron_schedule,
    create_interval_schedule,
)
from .validation_reporter import ValidationReporter, create_validation_reporter

# Phase 2.5 - Day 7 Walk-Forward Validation
from .walk_forward_validator import (
    FoldResult,
    WalkForwardConfig,
    WalkForwardResults,
    WalkForwardValidator,
    create_walk_forward_validator,
)

__all__ = [
    # Original components
    "MLModel",
    "FeatureEngineer",
    "ModelRegistry",
    # Phase 2.5 - Feature Engineering
    "OptimizedFeatureEngineer",
    "FeatureConfig",
    # Phase 2.5 - Feature Selection
    "AdvancedFeatureSelector",
    "FeatureSelectionConfig",
    "SelectionMethod",
    # Phase 2.5 - Model Validation
    "ModelValidator",
    "ValidationConfig",
    "ModelPerformance",
    # Phase 2.5 - Performance Tracking
    "PerformanceTracker",
    "PerformanceMetrics",
    "ModelHealth",
    "ModelStatus",
    # Phase 2.5 - Integrated Pipeline
    "IntegratedMLPipeline",
    # Phase 2.5 - Walk-Forward Validation
    "WalkForwardValidator",
    "WalkForwardConfig",
    "WalkForwardResults",
    "FoldResult",
    "create_walk_forward_validator",
    # Phase 2.5 - Model Degradation
    "ModelDegradationMonitor",
    "DegradationMetrics",
    "DegradationStatus",
    "DegradationType",
    "RetrainingTrigger",
    "create_degradation_monitor",
    # Phase 2.5 - Validation Reporting
    "ValidationReporter",
    "create_validation_reporter",
    # Phase 3 - Automated Retraining System
    "AutoRetrainingSystem",
    "RetrainingConfig",
    "RetrainingRequest",
    "RetrainingResult",
    "AutoRetrainingTrigger",
    "RetrainingStatus",
    "EmergencyLevel",
    "RetrainingCost",
    "create_auto_retraining_system",
    "CONSERVATIVE_RETRAINING_CONFIG",
    "AGGRESSIVE_RETRAINING_CONFIG",
    "PRODUCTION_RETRAINING_CONFIG",
    # Retraining Scheduler
    "RetrainingScheduler",
    "ScheduleConfig",
    "ScheduleType",
    "TaskPriority",
    "TaskStatus",
    "ScheduledTask",
    "ExecutionResult",
    "create_cron_schedule",
    "create_interval_schedule",
    "create_adaptive_schedule",
    "DAILY_RETRAINING_SCHEDULE",
    "WEEKLY_RETRAINING_SCHEDULE",
    "PERFORMANCE_ADAPTIVE_SCHEDULE",
    # Model Versioning
    "ModelVersioning",
    "ModelMetadata",
    "ModelStage",
    "VersionType",
    "ModelFormat",
    "VersionComparisonResult",
    "create_model_versioning",
    # Phase 4 - Deep Learning Components
    "LSTMArchitecture",
    "LSTMConfig",
    "TaskType",
    "LSTMDataPipeline",
    "SequenceConfig",
    "ScalingMethod",
    "LSTMTrainingFramework",
    "TrainingConfig",
    "TrainingResults",
    "AttentionMechanism",
    "AttentionConfig",
    "AttentionType",
    "create_integrated_lstm_pipeline",
]
