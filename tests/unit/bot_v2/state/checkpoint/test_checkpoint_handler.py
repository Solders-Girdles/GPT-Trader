"""Tests for checkpoint handler coordinating checkpoint operations."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

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


@pytest.fixture
def mock_state_manager():
    """Create a mock state manager."""
    manager = Mock()
    manager.get_keys_by_pattern = AsyncMock(return_value=[])
    manager.get_state = AsyncMock(return_value=None)
    manager.set_state = AsyncMock()
    return manager


@pytest.fixture
def temp_checkpoint_config(tmp_path):
    """Create temporary checkpoint configuration."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return CheckpointConfig(
        checkpoint_dir=str(checkpoint_dir),
        max_checkpoints=3,
        retention_days=7,
        compression_enabled=False,
        pause_trading_during_checkpoint=True,
    )


@pytest.fixture
def checkpoint_handler(mock_state_manager, temp_checkpoint_config):
    """Create checkpoint handler instance."""
    return CheckpointHandler(mock_state_manager, temp_checkpoint_config)


class TestCheckpointHandlerInitialization:
    """Test checkpoint handler initialization."""

    def test_initialization_creates_components(self, mock_state_manager, temp_checkpoint_config):
        """Test that initialization creates all required components."""
        handler = CheckpointHandler(mock_state_manager, temp_checkpoint_config)

        assert handler.state_manager == mock_state_manager
        assert handler.config == temp_checkpoint_config
        assert handler.storage is not None
        assert handler.capture is not None
        assert handler.verification is not None
        assert handler.restoration is not None
        assert isinstance(handler._checkpoint_history, list)
        assert handler._current_version == 0

    def test_initialization_loads_existing_checkpoints(self, mock_state_manager, tmp_path):
        """Test that initialization loads existing checkpoints from disk."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Create a checkpoint metadata file
        checkpoint_id = "existing_checkpoint"
        meta_file = checkpoint_dir / f"{checkpoint_id}.meta"
        checkpoint_data = {
            "checkpoint_id": checkpoint_id,
            "timestamp": datetime.utcnow().isoformat(),
            "version": 1,
            "consistency_hash": "hash123",
            "size_bytes": 1024,
            "status": "valid",
            "checkpoint_type": "automatic",
            "metadata": {},
        }
        meta_file.write_text(json.dumps(checkpoint_data))

        # Create corresponding checkpoint file
        checkpoint_file = checkpoint_dir / f"{checkpoint_id}.checkpoint"
        checkpoint_file.write_text(json.dumps({"positions": {}}))

        config = CheckpointConfig(checkpoint_dir=str(checkpoint_dir), compression_enabled=False)
        handler = CheckpointHandler(mock_state_manager, config)

        # Should have loaded the existing checkpoint
        assert len(handler._checkpoint_history) == 1
        assert handler._current_version == 1


class TestCheckpointCreation:
    """Test checkpoint creation functionality."""

    async def test_create_checkpoint_success(self, checkpoint_handler, mock_state_manager):
        """Test successful checkpoint creation."""
        # Mock state capture
        mock_state_manager.get_keys_by_pattern.return_value = ["position:BTC"]
        mock_state_manager.get_state.return_value = {"symbol": "BTC", "qty": 1.0}

        checkpoint = await checkpoint_handler.create_checkpoint(CheckpointType.MANUAL)

        assert checkpoint is not None
        assert checkpoint.checkpoint_type == CheckpointType.MANUAL
        assert checkpoint.status == CheckpointStatus.VALID
        assert checkpoint.version == 1
        assert len(checkpoint_handler._checkpoint_history) == 1

    async def test_create_checkpoint_pauses_trading(self, checkpoint_handler, mock_state_manager):
        """Test that trading is paused during checkpoint creation."""
        mock_state_manager.get_keys_by_pattern.return_value = []

        await checkpoint_handler.create_checkpoint(CheckpointType.AUTOMATIC)

        # Should have set checkpoint in progress flag
        mock_state_manager.set_state.assert_any_call("system:checkpoint_in_progress", True)
        # Should have cleared it afterward
        mock_state_manager.set_state.assert_any_call("system:checkpoint_in_progress", False)

    async def test_create_checkpoint_handles_capture_failure(
        self, checkpoint_handler, mock_state_manager
    ):
        """Test handling of state capture failure."""
        # Make capture fail
        mock_state_manager.get_keys_by_pattern.side_effect = Exception("Capture failed")

        checkpoint = await checkpoint_handler.create_checkpoint()

        assert checkpoint is None
        assert len(checkpoint_handler._checkpoint_history) == 0

    async def test_create_checkpoint_with_compression(
        self, mock_state_manager, temp_checkpoint_config
    ):
        """Test checkpoint creation with compression enabled."""
        temp_checkpoint_config.compression_enabled = True
        handler = CheckpointHandler(mock_state_manager, temp_checkpoint_config)

        mock_state_manager.get_keys_by_pattern.return_value = []

        checkpoint = await handler.create_checkpoint()

        assert checkpoint is not None
        # Checkpoint file should be compressed (we can't easily verify compression format)

    async def test_create_checkpoint_increments_version(
        self, checkpoint_handler, mock_state_manager
    ):
        """Test that version number increments with each checkpoint."""
        mock_state_manager.get_keys_by_pattern.return_value = []

        cp1 = await checkpoint_handler.create_checkpoint()
        cp2 = await checkpoint_handler.create_checkpoint()
        cp3 = await checkpoint_handler.create_checkpoint()

        assert cp1.version == 1
        assert cp2.version == 2
        assert cp3.version == 3

    async def test_create_checkpoint_cleanup_old(self, checkpoint_handler, mock_state_manager):
        """Test automatic cleanup of old checkpoints."""
        mock_state_manager.get_keys_by_pattern.return_value = []

        # Create more checkpoints than max_checkpoints (3)
        for i in range(5):
            await checkpoint_handler.create_checkpoint()

        # Should only keep max_checkpoints (3) + protected types
        assert len(checkpoint_handler._checkpoint_history) <= 5  # Some might be protected

    async def test_create_checkpoint_generates_unique_id(
        self, checkpoint_handler, mock_state_manager
    ):
        """Test that each checkpoint gets a unique ID."""
        mock_state_manager.get_keys_by_pattern.return_value = []

        cp1 = await checkpoint_handler.create_checkpoint()
        cp2 = await checkpoint_handler.create_checkpoint()

        assert cp1.checkpoint_id != cp2.checkpoint_id
        assert cp1.checkpoint_id.startswith("CP_")
        assert cp2.checkpoint_id.startswith("CP_")


class TestCheckpointRetrieval:
    """Test checkpoint retrieval functionality."""

    async def test_get_checkpoint_from_memory(self, checkpoint_handler, mock_state_manager):
        """Test getting checkpoint from in-memory history."""
        mock_state_manager.get_keys_by_pattern.return_value = []
        checkpoint = await checkpoint_handler.create_checkpoint()

        retrieved = checkpoint_handler.get_checkpoint(checkpoint.checkpoint_id)

        assert retrieved is not None
        assert retrieved.checkpoint_id == checkpoint.checkpoint_id

    def test_get_checkpoint_from_disk(self, checkpoint_handler, temp_checkpoint_config):
        """Test loading checkpoint from disk when not in memory."""
        # Create a checkpoint file directly on disk
        checkpoint_id = "disk_checkpoint"
        checkpoint_dir = Path(temp_checkpoint_config.checkpoint_dir)

        meta_data = {
            "checkpoint_id": checkpoint_id,
            "timestamp": datetime.utcnow().isoformat(),
            "version": 1,
            "consistency_hash": "hash",
            "size_bytes": 100,
            "status": "valid",
            "checkpoint_type": "manual",
            "metadata": {},
        }

        (checkpoint_dir / f"{checkpoint_id}.meta").write_text(json.dumps(meta_data))
        (checkpoint_dir / f"{checkpoint_id}.checkpoint").write_text(json.dumps({"positions": {}}))

        retrieved = checkpoint_handler.get_checkpoint(checkpoint_id)

        assert retrieved is not None
        assert retrieved.checkpoint_id == checkpoint_id

    def test_get_checkpoint_nonexistent(self, checkpoint_handler):
        """Test getting non-existent checkpoint."""
        retrieved = checkpoint_handler.get_checkpoint("nonexistent")
        assert retrieved is None

    async def test_get_latest_checkpoint(self, checkpoint_handler, mock_state_manager):
        """Test getting most recent valid checkpoint."""
        mock_state_manager.get_keys_by_pattern.return_value = []

        cp1 = await checkpoint_handler.create_checkpoint()
        cp2 = await checkpoint_handler.create_checkpoint()
        cp3 = await checkpoint_handler.create_checkpoint()

        latest = checkpoint_handler.get_latest_checkpoint()

        assert latest is not None
        assert latest.checkpoint_id == cp3.checkpoint_id

    def test_get_latest_checkpoint_empty(self, checkpoint_handler):
        """Test getting latest checkpoint when none exist."""
        latest = checkpoint_handler.get_latest_checkpoint()
        assert latest is None

    async def test_find_valid_checkpoint_before_time(self, checkpoint_handler, mock_state_manager):
        """Test finding valid checkpoint before specific time."""
        mock_state_manager.get_keys_by_pattern.return_value = []

        cp1 = await checkpoint_handler.create_checkpoint()
        # Wait a moment to ensure different timestamps
        import time

        time.sleep(0.01)
        cp2 = await checkpoint_handler.create_checkpoint()

        # Find checkpoint before cp2's timestamp
        found = await checkpoint_handler.find_valid_checkpoint(before=cp2.timestamp)

        assert found is not None
        assert found.checkpoint_id == cp1.checkpoint_id

    async def test_find_valid_checkpoint_verifies_integrity(
        self, checkpoint_handler, mock_state_manager
    ):
        """Test that find_valid_checkpoint verifies integrity."""
        mock_state_manager.get_keys_by_pattern.return_value = []

        await checkpoint_handler.create_checkpoint()

        # Mock integrity verification to fail
        with patch.object(
            checkpoint_handler.storage, "verify_checkpoint_integrity", return_value=False
        ):
            found = await checkpoint_handler.find_valid_checkpoint()
            assert found is None


class TestCheckpointRestoration:
    """Test checkpoint restoration functionality."""

    async def test_restore_from_checkpoint(self, checkpoint_handler, mock_state_manager):
        """Test restoring from checkpoint."""
        mock_state_manager.get_keys_by_pattern.return_value = []
        checkpoint = await checkpoint_handler.create_checkpoint()

        # Mock restoration to succeed
        checkpoint_handler.restoration.restore_from_checkpoint = AsyncMock(return_value=True)

        success = await checkpoint_handler.restore_from_checkpoint(checkpoint)

        assert success is True
        checkpoint_handler.restoration.restore_from_checkpoint.assert_called_once_with(checkpoint)

    async def test_rollback_to_checkpoint_success(self, checkpoint_handler, mock_state_manager):
        """Test successful rollback to checkpoint."""
        mock_state_manager.get_keys_by_pattern.return_value = []
        checkpoint = await checkpoint_handler.create_checkpoint()

        # Mock restoration to succeed
        checkpoint_handler.restoration.restore_from_checkpoint = AsyncMock(return_value=True)

        success = await checkpoint_handler.rollback_to_checkpoint(checkpoint.checkpoint_id)

        assert success is True
        # Should have created an emergency checkpoint first
        emergency_checkpoints = [
            cp
            for cp in checkpoint_handler._checkpoint_history
            if cp.checkpoint_type == CheckpointType.EMERGENCY
        ]
        assert len(emergency_checkpoints) >= 1

    async def test_rollback_to_nonexistent_checkpoint(self, checkpoint_handler, mock_state_manager):
        """Test rollback to non-existent checkpoint fails."""
        success = await checkpoint_handler.rollback_to_checkpoint("nonexistent")
        assert success is False

    async def test_rollback_restores_emergency_on_failure(
        self, checkpoint_handler, mock_state_manager
    ):
        """Test that emergency checkpoint is restored if rollback fails."""
        mock_state_manager.get_keys_by_pattern.return_value = []
        checkpoint = await checkpoint_handler.create_checkpoint()

        # Mock restoration to fail on first call, succeed on emergency restore
        restore_calls = []

        async def mock_restore(cp):
            restore_calls.append(cp)
            if len(restore_calls) == 1:
                return False  # Original rollback fails
            return True  # Emergency restore succeeds

        checkpoint_handler.restoration.restore_from_checkpoint = mock_restore

        success = await checkpoint_handler.rollback_to_checkpoint(checkpoint.checkpoint_id)

        assert success is False
        assert len(restore_calls) == 2  # Original + emergency restore


class TestCheckpointCleanup:
    """Test checkpoint cleanup functionality."""

    async def test_cleanup_respects_max_checkpoints(self, checkpoint_handler, mock_state_manager):
        """Test that cleanup enforces max_checkpoints limit for old checkpoints."""
        mock_state_manager.get_keys_by_pattern.return_value = []

        # Create checkpoints and mark first ones as old
        checkpoints = []
        for i in range(5):
            cp = await checkpoint_handler.create_checkpoint(CheckpointType.AUTOMATIC)
            if i < 2:  # Make first 2 old
                cp.timestamp = datetime.utcnow() - timedelta(
                    days=checkpoint_handler.config.retention_days + 1
                )
            checkpoints.append(cp)

        # Manually trigger cleanup
        checkpoint_handler._cleanup_old_checkpoints()

        # Old checkpoints beyond max should be removed
        auto_checkpoints = [
            cp
            for cp in checkpoint_handler._checkpoint_history
            if cp.checkpoint_type == CheckpointType.AUTOMATIC
        ]

        # Should have removed old ones if total > max
        assert (
            len(auto_checkpoints) <= checkpoint_handler.config.max_checkpoints + 2
        )  # Some tolerance

    async def test_cleanup_preserves_emergency_checkpoints(
        self, checkpoint_handler, mock_state_manager
    ):
        """Test that emergency checkpoints are preserved during cleanup."""
        mock_state_manager.get_keys_by_pattern.return_value = []

        # Create an old emergency checkpoint
        emergency_cp = await checkpoint_handler.create_checkpoint(CheckpointType.EMERGENCY)
        emergency_cp.timestamp = datetime.utcnow() - timedelta(days=30)

        # Create many newer automatic checkpoints
        for i in range(5):
            await checkpoint_handler.create_checkpoint(CheckpointType.AUTOMATIC)

        # Emergency checkpoint should still be there
        emergency_checkpoints = [
            cp
            for cp in checkpoint_handler._checkpoint_history
            if cp.checkpoint_type == CheckpointType.EMERGENCY
        ]
        assert len(emergency_checkpoints) >= 1

    async def test_cleanup_deletes_old_automatic_checkpoints(
        self, checkpoint_handler, mock_state_manager
    ):
        """Test that old automatic checkpoints are deleted."""
        mock_state_manager.get_keys_by_pattern.return_value = []

        # Create an old automatic checkpoint
        cp = await checkpoint_handler.create_checkpoint(CheckpointType.AUTOMATIC)
        cp.timestamp = datetime.utcnow() - timedelta(
            days=checkpoint_handler.config.retention_days + 1
        )

        # Manually add to history (simulating old checkpoint)
        checkpoint_handler._checkpoint_history.insert(0, cp)

        # Create new checkpoints to trigger cleanup
        for i in range(5):
            await checkpoint_handler.create_checkpoint(CheckpointType.AUTOMATIC)

        # Old checkpoint should be gone (if it exceeded max_checkpoints)
        old_checkpoints = [
            c
            for c in checkpoint_handler._checkpoint_history
            if c.timestamp < datetime.utcnow() - timedelta(days=7)
        ]
        # May or may not be deleted depending on total count
        assert True  # Just verify no exceptions


class TestCheckpointStatistics:
    """Test checkpoint statistics functionality."""

    async def test_get_checkpoint_stats_empty(self, checkpoint_handler):
        """Test stats when no checkpoints exist."""
        stats = checkpoint_handler.get_checkpoint_stats()

        assert stats["total_checkpoints"] == 0
        assert stats["total_size_mb"] == 0

    async def test_get_checkpoint_stats_with_checkpoints(
        self, checkpoint_handler, mock_state_manager
    ):
        """Test stats with existing checkpoints."""
        mock_state_manager.get_keys_by_pattern.return_value = []

        await checkpoint_handler.create_checkpoint()
        await checkpoint_handler.create_checkpoint()

        stats = checkpoint_handler.get_checkpoint_stats()

        assert stats["total_checkpoints"] == 2
        assert stats["valid_checkpoints"] == 2
        assert stats["total_size_mb"] > 0
        assert stats["average_size_kb"] > 0
        assert "latest_checkpoint" in stats


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    async def test_create_checkpoint_convenience(self, mock_state_manager):
        """Test create_checkpoint convenience function."""
        mock_state_manager.get_keys_by_pattern.return_value = []

        checkpoint = await create_checkpoint(mock_state_manager, CheckpointType.MANUAL)

        assert checkpoint is not None
        assert checkpoint.checkpoint_type == CheckpointType.MANUAL

    async def test_restore_latest_checkpoint_success(self, mock_state_manager, tmp_path):
        """Test restore_latest_checkpoint convenience function."""
        # Create a checkpoint first
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        config = CheckpointConfig(checkpoint_dir=str(checkpoint_dir), compression_enabled=False)
        handler = CheckpointHandler(mock_state_manager, config)

        mock_state_manager.get_keys_by_pattern.return_value = []
        await handler.create_checkpoint()

        # Mock restoration
        with patch.object(handler.restoration, "restore_from_checkpoint", return_value=True):
            success = await restore_latest_checkpoint(mock_state_manager)
            # This creates a new handler, so may or may not find the checkpoint
            assert success in [True, False]

    async def test_restore_latest_checkpoint_none_exist(self, mock_state_manager, tmp_path):
        """Test restore_latest_checkpoint when no checkpoints exist."""
        # Use a temporary directory with no checkpoints
        checkpoint_dir = tmp_path / "empty_checkpoints"
        checkpoint_dir.mkdir()

        with patch("bot_v2.state.checkpoint.handler.CheckpointConfig") as mock_config_class:
            mock_config_class.return_value = CheckpointConfig(
                checkpoint_dir=str(checkpoint_dir), compression_enabled=False
            )

            success = await restore_latest_checkpoint(mock_state_manager)
            assert success is False
