"""
State Management System for Bot V2 Trading System

Provides comprehensive state management with multi-tier storage,
atomic checkpoints, automated recovery, and encrypted backups.
"""

from .state_manager import (
    StateManager,
    StateCategory,
    StateMetadata,
    StateConfig,
    get_state_manager,
    get_state,
    set_state,
    delete_state
)

from .checkpoint_handler import (
    CheckpointHandler,
    Checkpoint,
    CheckpointStatus,
    CheckpointType,
    CheckpointConfig,
    create_checkpoint,
    restore_latest_checkpoint
)

from .recovery_handler import (
    RecoveryHandler,
    RecoveryMode,
    FailureType,
    RecoveryStatus,
    FailureEvent,
    RecoveryOperation,
    RecoveryConfig,
    detect_and_recover
)

from .backup_manager import (
    BackupManager,
    BackupType,
    BackupStatus,
    StorageTier,
    BackupMetadata,
    BackupConfig,
    create_backup,
    restore_latest
)

__all__ = [
    # State Manager
    'StateManager',
    'StateCategory',
    'StateMetadata',
    'StateConfig',
    'get_state_manager',
    'get_state',
    'set_state',
    'delete_state',
    
    # Checkpoint Handler
    'CheckpointHandler',
    'Checkpoint',
    'CheckpointStatus',
    'CheckpointType',
    'CheckpointConfig',
    'create_checkpoint',
    'restore_latest_checkpoint',
    
    # Recovery Handler
    'RecoveryHandler',
    'RecoveryMode',
    'FailureType',
    'RecoveryStatus',
    'FailureEvent',
    'RecoveryOperation',
    'RecoveryConfig',
    'detect_and_recover',
    
    # Backup Manager
    'BackupManager',
    'BackupType',
    'BackupStatus',
    'StorageTier',
    'BackupMetadata',
    'BackupConfig',
    'create_backup',
    'restore_latest'
]

# Version
__version__ = '1.0.0'