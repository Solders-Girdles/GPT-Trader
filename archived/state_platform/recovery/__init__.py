"""
Recovery subsystem for Bot V2 Trading System.

Provides failure detection, recovery orchestration, and validation with
RTO <5 minutes and RPO <1 minute targets.

Structure:
- orchestrator.py: Main RecoveryOrchestrator facade (backward-compatible as RecoveryHandler)
- detection.py: Failure detection and health checks
- validation.py: Post-recovery validation
- alerting.py: Alert dispatch and escalation
- handlers/: Recovery handler implementations by domain
  - storage.py: Redis, PostgreSQL, S3
  - trading.py: Trading engine, ML models
  - system.py: Memory, disk, network, API
- models.py: Data models and enums
"""

from bot_v2.state.recovery.models import (
    FailureEvent,
    FailureType,
    RecoveryConfig,
    RecoveryMode,
    RecoveryOperation,
    RecoveryStatus,
)
from bot_v2.state.recovery.orchestrator import (
    RecoveryHandler,
    RecoveryOrchestrator,
    detect_and_recover,
)

__all__ = [
    # Models
    "RecoveryMode",
    "FailureType",
    "RecoveryStatus",
    "FailureEvent",
    "RecoveryOperation",
    "RecoveryConfig",
    # Orchestrator (main API)
    "RecoveryOrchestrator",
    "RecoveryHandler",  # Backward compatibility alias
    "detect_and_recover",
]
