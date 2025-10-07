"""Main checkpoint handler facade coordinating all checkpoint operations"""

import gzip
import json
import logging
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from bot_v2.state.checkpoint.capture import StateCapture
from bot_v2.state.checkpoint.models import (
    Checkpoint,
    CheckpointConfig,
    CheckpointStatus,
    CheckpointType,
)
from bot_v2.state.checkpoint.restoration import CheckpointRestoration
from bot_v2.state.checkpoint.storage import CheckpointStorage
from bot_v2.state.checkpoint.verification import CheckpointVerification

logger = logging.getLogger(__name__)


class CheckpointHandler:
    """
    Orchestrates checkpoint operations with atomic guarantees.

    Coordinates state capture, storage, restoration, and verification.
    """

    def __init__(self, state_manager: Any, config: CheckpointConfig | None = None) -> None:
        self.state_manager = state_manager
        self.config = config or CheckpointConfig()

        # Initialize components
        self.storage = CheckpointStorage(self.config)
        self.capture = StateCapture(state_manager)
        self.verification = CheckpointVerification(state_manager, self.storage)
        self.restoration = CheckpointRestoration(state_manager, self.storage, self.verification)

        # State tracking
        self._checkpoint_lock = threading.Lock()
        self._checkpoint_history: list[Checkpoint] = []
        self._current_version = 0

        # Load existing checkpoints
        self._load_checkpoint_history()

        logger.info(
            f"CheckpointHandler initialized with {len(self._checkpoint_history)} checkpoints"
        )

    async def create_checkpoint(
        self, checkpoint_type: CheckpointType = CheckpointType.AUTOMATIC
    ) -> Checkpoint | None:
        """
        Create atomic checkpoint of current system state.

        Args:
            checkpoint_type: Type of checkpoint to create

        Returns:
            Created checkpoint or None on failure
        """
        with self._checkpoint_lock:
            try:
                logger.info(f"Creating {checkpoint_type.value} checkpoint")

                # Pause trading if configured
                if self.config.pause_trading_during_checkpoint:
                    await self.state_manager.set_state("system:checkpoint_in_progress", True)

                # Capture system state
                state_snapshot = await self.capture.capture_system_state()

                if not state_snapshot:
                    logger.error("Failed to capture system state")
                    return None

                # Create checkpoint object
                checkpoint = Checkpoint(
                    checkpoint_id=self._generate_checkpoint_id(checkpoint_type),
                    timestamp=datetime.utcnow(),
                    state_snapshot=state_snapshot,
                    version=self._get_next_version(),
                    consistency_hash=self.storage.calculate_consistency_hash(state_snapshot),
                    size_bytes=len(json.dumps(state_snapshot, default=str)),
                    status=CheckpointStatus.CREATING,
                    checkpoint_type=checkpoint_type,
                )

                # Serialize and compress
                serialized = json.dumps(checkpoint.state_snapshot, default=str).encode("utf-8")
                if self.config.compression_enabled:
                    data = gzip.compress(serialized, compresslevel=6)
                else:
                    data = serialized

                # Store atomically
                success = await self.storage.store_checkpoint_atomic(checkpoint, data)

                if success:
                    checkpoint.status = CheckpointStatus.VALID
                    self._checkpoint_history.append(checkpoint)
                    self._cleanup_old_checkpoints()

                    logger.info(
                        f"Checkpoint {checkpoint.checkpoint_id} created successfully "
                        f"({checkpoint.size_bytes / 1024:.1f}KB)"
                    )
                    return checkpoint
                else:
                    logger.error("Failed to store checkpoint")
                    return None

            except Exception as e:
                logger.error(f"Checkpoint creation failed: {e}")
                return None

            finally:
                # Resume trading
                if self.config.pause_trading_during_checkpoint:
                    await self.state_manager.set_state("system:checkpoint_in_progress", False)

    async def restore_from_checkpoint(self, checkpoint: Checkpoint) -> bool:
        """Restore system state from checkpoint"""
        return await self.restoration.restore_from_checkpoint(checkpoint)

    async def rollback_to_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Rollback system to specific checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to rollback to

        Returns:
            Success status
        """
        checkpoint = self.get_checkpoint(checkpoint_id)
        if not checkpoint:
            logger.error(f"Checkpoint {checkpoint_id} not found")
            return False

        # Create emergency checkpoint before rollback
        emergency_cp = await self.create_checkpoint(checkpoint_type=CheckpointType.EMERGENCY)

        if emergency_cp:
            logger.info(
                f"Created emergency checkpoint {emergency_cp.checkpoint_id} before rollback"
            )

        # Perform rollback
        success = await self.restore_from_checkpoint(checkpoint)

        if success:
            logger.info(f"Successfully rolled back to checkpoint {checkpoint_id}")
        else:
            logger.error(f"Rollback to checkpoint {checkpoint_id} failed")

            # Try to restore emergency checkpoint
            if emergency_cp:
                logger.info("Attempting to restore emergency checkpoint")
                await self.restore_from_checkpoint(emergency_cp)

        return success

    def get_checkpoint(self, checkpoint_id: str) -> Checkpoint | None:
        """Get checkpoint by ID"""
        for checkpoint in self._checkpoint_history:
            if checkpoint.checkpoint_id == checkpoint_id:
                return checkpoint

        # Try loading from disk
        return self.storage.load_checkpoint_from_disk(checkpoint_id)

    def get_latest_checkpoint(self) -> Checkpoint | None:
        """Get most recent valid checkpoint"""
        valid_checkpoints = [
            cp for cp in self._checkpoint_history if cp.status == CheckpointStatus.VALID
        ]

        if valid_checkpoints:
            return max(valid_checkpoints, key=lambda cp: cp.timestamp)

        return None

    async def find_valid_checkpoint(self, before: datetime | None = None) -> Checkpoint | None:
        """Find last valid checkpoint before given time"""
        valid_checkpoints = [
            cp for cp in self._checkpoint_history if cp.status == CheckpointStatus.VALID
        ]

        if before:
            valid_checkpoints = [cp for cp in valid_checkpoints if cp.timestamp < before]

        if not valid_checkpoints:
            return None

        # Verify integrity of most recent checkpoint
        for checkpoint in sorted(valid_checkpoints, key=lambda cp: cp.timestamp, reverse=True):
            if await self.storage.verify_checkpoint_integrity(checkpoint):
                return checkpoint

        return None

    def _generate_checkpoint_id(self, checkpoint_type: CheckpointType) -> str:
        """Generate unique checkpoint ID"""
        return f"CP_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{checkpoint_type.value}_{uuid.uuid4().hex[:8]}"

    def _get_next_version(self) -> int:
        """Get next version number"""
        self._current_version += 1
        return self._current_version

    def _load_checkpoint_history(self) -> None:
        """Load checkpoint history from disk"""
        try:
            checkpoint_path = Path(self.config.checkpoint_dir)

            if not checkpoint_path.exists():
                return

            # Load all checkpoint metadata files
            for meta_file in checkpoint_path.glob("*.meta"):
                checkpoint_id = meta_file.stem
                checkpoint = self.storage.load_checkpoint_from_disk(checkpoint_id)

                if checkpoint:
                    self._checkpoint_history.append(checkpoint)
                    self._current_version = max(self._current_version, checkpoint.version)

            logger.info(f"Loaded {len(self._checkpoint_history)} checkpoints from disk")

        except Exception as e:
            logger.error(f"Failed to load checkpoint history: {e}")

    def _cleanup_old_checkpoints(self) -> None:
        """Clean up old checkpoints"""
        if len(self._checkpoint_history) <= self.config.max_checkpoints:
            return

        # Sort by timestamp (oldest first)
        sorted_checkpoints = sorted(self._checkpoint_history, key=lambda cp: cp.timestamp)

        # Keep most recent checkpoints
        to_delete = sorted_checkpoints[: -self.config.max_checkpoints]

        for checkpoint in to_delete:
            # Don't delete emergency or manual checkpoints
            if checkpoint.checkpoint_type in [CheckpointType.EMERGENCY, CheckpointType.MANUAL]:
                continue

            # Delete old automatic checkpoints
            cutoff_date = datetime.utcnow() - timedelta(days=self.config.retention_days)
            if checkpoint.timestamp < cutoff_date:
                self.storage.delete_checkpoint_files(checkpoint.checkpoint_id)
                self._checkpoint_history.remove(checkpoint)
                logger.debug(f"Deleted old checkpoint {checkpoint.checkpoint_id}")

    def get_checkpoint_stats(self) -> dict[str, Any]:
        """Get checkpoint statistics"""
        if not self._checkpoint_history:
            return {"total_checkpoints": 0, "total_size_mb": 0}

        total_size = sum(cp.size_bytes for cp in self._checkpoint_history)
        valid_count = sum(
            1 for cp in self._checkpoint_history if cp.status == CheckpointStatus.VALID
        )

        return {
            "total_checkpoints": len(self._checkpoint_history),
            "valid_checkpoints": valid_count,
            "total_size_mb": total_size / (1024 * 1024),
            "latest_checkpoint": (
                self.get_latest_checkpoint().timestamp if self.get_latest_checkpoint() else None
            ),
            "average_size_kb": total_size / len(self._checkpoint_history) / 1024,
        }


# Convenience functions
async def create_checkpoint(
    state_manager: Any, checkpoint_type: CheckpointType = CheckpointType.MANUAL
) -> Checkpoint | None:
    """Create a checkpoint"""
    handler = CheckpointHandler(state_manager)
    return await handler.create_checkpoint(checkpoint_type)


async def restore_latest_checkpoint(state_manager: Any) -> bool:
    """Restore from latest checkpoint"""
    handler = CheckpointHandler(state_manager)
    latest = handler.get_latest_checkpoint()
    if latest:
        return await handler.restore_from_checkpoint(latest)
    return False
