"""
Recovery system data models and types.

Defines enums, dataclasses, and configuration models for the recovery subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

__all__ = [
    "RecoveryMode",
    "FailureType",
    "RecoveryStatus",
    "FailureEvent",
    "RecoveryOperation",
    "RecoveryConfig",
]


class RecoveryMode(Enum):
    """Recovery operation modes"""

    AUTOMATIC = "automatic"
    MANUAL = "manual"
    EMERGENCY = "emergency"
    MAINTENANCE = "maintenance"


class FailureType(Enum):
    """Types of system failures"""

    REDIS_DOWN = "redis_down"
    POSTGRES_DOWN = "postgres_down"
    S3_UNAVAILABLE = "s3_unavailable"
    NETWORK_PARTITION = "network_partition"
    DATA_CORRUPTION = "data_corruption"
    TRADING_ENGINE_CRASH = "trading_engine_crash"
    ML_MODEL_FAILURE = "ml_model_failure"
    API_GATEWAY_DOWN = "api_gateway_down"
    MEMORY_OVERFLOW = "memory_overflow"
    DISK_FULL = "disk_full"


class RecoveryStatus(Enum):
    """Recovery operation status"""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class FailureEvent:
    """Failure event information"""

    failure_type: FailureType
    timestamp: datetime
    severity: str  # critical, high, medium, low
    affected_components: list[str]
    error_message: str
    stack_trace: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryOperation:
    """Recovery operation details"""

    operation_id: str
    failure_event: FailureEvent
    recovery_mode: RecoveryMode
    status: RecoveryStatus
    started_at: datetime
    completed_at: datetime | None = None
    recovery_time_seconds: float | None = None
    data_loss_estimate: str | None = None
    actions_taken: list[str] = field(default_factory=list)
    validation_results: dict[str, bool] = field(default_factory=dict)


@dataclass
class RecoveryConfig:
    """Configuration for recovery operations"""

    rto_minutes: int = 5  # Recovery Time Objective
    rpo_minutes: int = 1  # Recovery Point Objective
    max_retry_attempts: int = 3
    retry_delay_seconds: int = 30
    automatic_recovery_enabled: bool = True
    failure_detection_interval_seconds: int = 10
    validation_timeout_seconds: int = 60
    escalation_threshold_minutes: int = 3
    alert_channels: list[str] = field(default_factory=lambda: ["log", "email"])
