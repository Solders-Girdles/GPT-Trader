"""
State Management System for Bot V2 Trading System

Provides comprehensive state management with multi-tier storage,
atomic checkpoints, automated recovery, and encrypted backups.
"""

from bot_v2.state.backup_manager import (
    BackupConfig,
    BackupManager,
    BackupMetadata,
    BackupStatus,
    BackupType,
    StorageTier,
    create_backup,
    restore_latest,
)
from bot_v2.state.checkpoint import (
    Checkpoint,
    CheckpointConfig,
    CheckpointHandler,
    CheckpointStatus,
    CheckpointType,
    create_checkpoint,
    restore_latest_checkpoint,
)
from bot_v2.state.recovery import (
    FailureEvent,
    FailureType,
    RecoveryConfig,
    RecoveryHandler,
    RecoveryMode,
    RecoveryOperation,
    RecoveryStatus,
    detect_and_recover,
)
from bot_v2.state.state_manager import (
    StateCategory,
    StateConfig,
    StateManager,
    StateMetadata,
    delete_state,
    get_state,
    get_state_manager,
    set_state,
)

__all__ = [
    # State Manager
    "StateManager",
    "StateCategory",
    "StateMetadata",
    "StateConfig",
    "get_state_manager",
    "get_state",
    "set_state",
    "delete_state",
    # Checkpoint Handler
    "CheckpointHandler",
    "Checkpoint",
    "CheckpointStatus",
    "CheckpointType",
    "CheckpointConfig",
    "create_checkpoint",
    "restore_latest_checkpoint",
    # Recovery Handler
    "RecoveryHandler",
    "RecoveryMode",
    "FailureType",
    "RecoveryStatus",
    "FailureEvent",
    "RecoveryOperation",
    "RecoveryConfig",
    "detect_and_recover",
    # Backup Manager
    "BackupManager",
    "BackupType",
    "BackupStatus",
    "StorageTier",
    "BackupMetadata",
    "BackupConfig",
    "create_backup",
    "restore_latest",
]

# Version
__version__ = "1.0.0"
