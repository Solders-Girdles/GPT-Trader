"""
Checkpoint subsystem for Bot V2 Trading System.

Provides atomic checkpoint operations with consistency guarantees,
version management, and rollback capabilities for system state.

Structure:
- handler.py: Main CheckpointHandler facade
- models.py: Data models (Checkpoint, CheckpointConfig, enums)
- capture.py: State capture logic
- storage.py: Atomic storage operations
- restoration.py: Restore and rollback operations
- verification.py: Integrity verification
"""

from bot_v2.state.checkpoint.handler import (
    CheckpointHandler,
    create_checkpoint,
    restore_latest_checkpoint,
)
from bot_v2.state.checkpoint.models import (
    Checkpoint,
    CheckpointConfig,
    CheckpointStatus,
    CheckpointType,
)

__all__ = [
    # Models
    "Checkpoint",
    "CheckpointStatus",
    "CheckpointType",
    "CheckpointConfig",
    # Handler
    "CheckpointHandler",
    # Convenience functions
    "create_checkpoint",
    "restore_latest_checkpoint",
]
