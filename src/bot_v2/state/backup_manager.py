"""Compatibility layer for the backup manager implementation.

This module keeps the public import surface ``bot_v2.state.backup_manager``
intact while delegating to the single source implementation under
``bot_v2.state.backup.operations``.
"""

from bot_v2.state.backup.operations import (
    BackupConfig,
    BackupManager,
    BackupMetadata,
    BackupStatus,
    BackupType,
    StorageTier,
    create_backup,
    restore_latest,
)

__all__ = [
    "BackupManager",
    "BackupConfig",
    "BackupMetadata",
    "BackupStatus",
    "BackupType",
    "StorageTier",
    "create_backup",
    "restore_latest",
]
