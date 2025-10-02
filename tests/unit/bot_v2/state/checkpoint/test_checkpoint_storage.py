"""Tests for checkpoint storage module."""

import json
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from bot_v2.state.checkpoint.models import Checkpoint, CheckpointConfig
from bot_v2.state.checkpoint.storage import CheckpointStorage


class TestCheckpointStorage:
    """Test suite for CheckpointStorage class."""

    @pytest.fixture
    def temp_checkpoint_dir(self, tmp_path):
        """Create temporary checkpoint directory."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        return checkpoint_dir

    @pytest.fixture
    def checkpoint_config(self, temp_checkpoint_dir):
        """Create checkpoint configuration."""
        return CheckpointConfig(
            checkpoint_dir=str(temp_checkpoint_dir),
            compression_enabled=False,
            max_checkpoints=5
        )

    @pytest.fixture
    def checkpoint_storage(self, checkpoint_config):
        """Create CheckpointStorage instance."""
        return CheckpointStorage(checkpoint_config)

    @pytest.fixture
    def sample_checkpoint(self):
        """Create sample checkpoint."""
        return Checkpoint(
            checkpoint_id="test_checkpoint_001",
            version=1,
            size_bytes=1024,
            timestamp=datetime.fromisoformat("2024-01-01T00:00:00"),
            state_snapshot={"positions": {}, "orders": {}},
            consistency_hash="abc123",
            metadata={"test": "data"}
        )

    def test_initialization(self, checkpoint_storage, temp_checkpoint_dir):
        """Test CheckpointStorage initialization."""
        assert checkpoint_storage.checkpoint_path.exists()
        assert checkpoint_storage.checkpoint_path == Path(temp_checkpoint_dir)

    def test_creates_checkpoint_dir_if_not_exists(self, tmp_path):
        """Test that checkpoint directory is created if it doesn't exist."""
        new_dir = tmp_path / "new_checkpoints"
        assert not new_dir.exists()

        config = CheckpointConfig(checkpoint_dir=str(new_dir))
        storage = CheckpointStorage(config)

        assert new_dir.exists()

    @pytest.mark.asyncio
    async def test_store_checkpoint_atomic_success(
        self, checkpoint_storage, sample_checkpoint
    ):
        """Test successful atomic checkpoint storage."""
        data = json.dumps(sample_checkpoint.state_snapshot).encode("utf-8")

        result = await checkpoint_storage.store_checkpoint_atomic(sample_checkpoint, data)

        assert result is True

        # Check files were created
        checkpoint_file = checkpoint_storage.checkpoint_path / f"{sample_checkpoint.checkpoint_id}.checkpoint"
        meta_file = checkpoint_storage.checkpoint_path / f"{sample_checkpoint.checkpoint_id}.meta"

        assert checkpoint_file.exists()
        assert meta_file.exists()

    @pytest.mark.asyncio
    async def test_store_checkpoint_metadata_excludes_snapshot(
        self, checkpoint_storage, sample_checkpoint
    ):
        """Test that metadata file doesn't duplicate state snapshot."""
        data = json.dumps(sample_checkpoint.state_snapshot).encode("utf-8")

        await checkpoint_storage.store_checkpoint_atomic(sample_checkpoint, data)

        meta_file = checkpoint_storage.checkpoint_path / f"{sample_checkpoint.checkpoint_id}.meta"
        with open(meta_file, "r") as f:
            metadata = json.load(f)

        # State snapshot should not be in metadata
        assert "state_snapshot" not in metadata
        assert "checkpoint_id" in metadata
        assert "timestamp" in metadata

    def test_load_checkpoint_from_disk_success(
        self, checkpoint_storage, sample_checkpoint
    ):
        """Test successful checkpoint loading."""
        # First store a checkpoint
        data = json.dumps(sample_checkpoint.state_snapshot).encode("utf-8")
        import asyncio
        asyncio.run(checkpoint_storage.store_checkpoint_atomic(sample_checkpoint, data))

        # Load it back
        loaded = checkpoint_storage.load_checkpoint_from_disk(sample_checkpoint.checkpoint_id)

        assert loaded is not None
        assert loaded.checkpoint_id == sample_checkpoint.checkpoint_id
        assert loaded.state_snapshot == sample_checkpoint.state_snapshot

    def test_load_checkpoint_missing_files(self, checkpoint_storage):
        """Test loading non-existent checkpoint."""
        loaded = checkpoint_storage.load_checkpoint_from_disk("nonexistent")

        assert loaded is None

    def test_calculate_consistency_hash(self, checkpoint_storage):
        """Test consistency hash calculation."""
        state = {"positions": {"BTC": 1.0}, "orders": []}

        hash1 = checkpoint_storage.calculate_consistency_hash(state)
        hash2 = checkpoint_storage.calculate_consistency_hash(state)

        # Same data should produce same hash
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) > 0

    def test_delete_checkpoint_files_success(self, checkpoint_storage, sample_checkpoint):
        """Test successful checkpoint deletion."""
        # Create files
        checkpoint_file = checkpoint_storage.checkpoint_path / f"{sample_checkpoint.checkpoint_id}.checkpoint"
        meta_file = checkpoint_storage.checkpoint_path / f"{sample_checkpoint.checkpoint_id}.meta"

        checkpoint_file.write_text("{}")
        meta_file.write_text("{}")

        # Delete
        result = checkpoint_storage.delete_checkpoint_files(sample_checkpoint.checkpoint_id)

        assert result is True
        assert not checkpoint_file.exists()
        assert not meta_file.exists()

    @pytest.mark.asyncio
    async def test_store_checkpoint_atomic_failure_cleanup(
        self, checkpoint_storage, sample_checkpoint
    ):
        """Test that failed checkpoint storage cleans up partial files."""
        # Mock storage to fail during write
        with patch.object(
            checkpoint_storage.storage, "write_atomic", side_effect=Exception("Write failed")
        ):
            data = b"test data"
            result = await checkpoint_storage.store_checkpoint_atomic(sample_checkpoint, data)

            assert result is False

            # Files should not exist after cleanup
            checkpoint_file = (
                checkpoint_storage.checkpoint_path
                / f"{sample_checkpoint.checkpoint_id}.checkpoint"
            )
            meta_file = (
                checkpoint_storage.checkpoint_path / f"{sample_checkpoint.checkpoint_id}.meta"
            )
            assert not checkpoint_file.exists()
            assert not meta_file.exists()

    def test_load_checkpoint_with_compression(self, checkpoint_storage, sample_checkpoint):
        """Test loading compressed checkpoint."""
        import gzip

        # Enable compression in config
        checkpoint_storage.config.compression_enabled = True

        # Store compressed checkpoint
        data = json.dumps(sample_checkpoint.state_snapshot).encode("utf-8")
        compressed_data = gzip.compress(data)

        checkpoint_file = (
            checkpoint_storage.checkpoint_path / f"{sample_checkpoint.checkpoint_id}.checkpoint"
        )
        meta_file = (
            checkpoint_storage.checkpoint_path / f"{sample_checkpoint.checkpoint_id}.meta"
        )

        checkpoint_file.write_bytes(compressed_data)

        metadata = sample_checkpoint.to_dict()
        del metadata["state_snapshot"]
        meta_file.write_text(json.dumps(metadata, default=str))

        # Load it back
        loaded = checkpoint_storage.load_checkpoint_from_disk(sample_checkpoint.checkpoint_id)

        assert loaded is not None
        assert loaded.state_snapshot == sample_checkpoint.state_snapshot

    def test_load_checkpoint_error_handling(self, checkpoint_storage, sample_checkpoint):
        """Test error handling during checkpoint load."""
        # Create corrupted metadata file
        meta_file = (
            checkpoint_storage.checkpoint_path / f"{sample_checkpoint.checkpoint_id}.meta"
        )
        meta_file.write_text("invalid json{")

        loaded = checkpoint_storage.load_checkpoint_from_disk(sample_checkpoint.checkpoint_id)

        assert loaded is None

    @pytest.mark.asyncio
    async def test_verify_checkpoint_integrity_success(
        self, checkpoint_storage, sample_checkpoint
    ):
        """Test successful checkpoint integrity verification."""
        # Calculate correct hash for the snapshot
        sample_checkpoint.consistency_hash = checkpoint_storage.calculate_consistency_hash(
            sample_checkpoint.state_snapshot
        )

        # Store checkpoint
        data = json.dumps(sample_checkpoint.state_snapshot).encode("utf-8")
        await checkpoint_storage.store_checkpoint_atomic(sample_checkpoint, data)

        # Verify integrity
        result = await checkpoint_storage.verify_checkpoint_integrity(sample_checkpoint)

        assert result is True

    @pytest.mark.asyncio
    async def test_verify_checkpoint_integrity_missing_file(
        self, checkpoint_storage, sample_checkpoint
    ):
        """Test integrity verification when file is missing."""
        result = await checkpoint_storage.verify_checkpoint_integrity(sample_checkpoint)

        assert result is False

    @pytest.mark.asyncio
    async def test_verify_checkpoint_integrity_hash_mismatch(
        self, checkpoint_storage, sample_checkpoint
    ):
        """Test integrity verification when hash doesn't match."""
        # Store checkpoint
        data = json.dumps(sample_checkpoint.state_snapshot).encode("utf-8")
        await checkpoint_storage.store_checkpoint_atomic(sample_checkpoint, data)

        # Modify checkpoint's hash to create mismatch
        sample_checkpoint.consistency_hash = "wrong_hash"

        result = await checkpoint_storage.verify_checkpoint_integrity(sample_checkpoint)

        assert result is False

    @pytest.mark.asyncio
    async def test_verify_checkpoint_integrity_with_compression(
        self, checkpoint_storage, sample_checkpoint
    ):
        """Test integrity verification with compressed checkpoint."""
        import gzip

        checkpoint_storage.config.compression_enabled = True

        # Calculate correct hash
        sample_checkpoint.consistency_hash = checkpoint_storage.calculate_consistency_hash(
            sample_checkpoint.state_snapshot
        )

        # Store compressed checkpoint
        data = json.dumps(sample_checkpoint.state_snapshot).encode("utf-8")
        compressed_data = gzip.compress(data)

        checkpoint_file = (
            checkpoint_storage.checkpoint_path / f"{sample_checkpoint.checkpoint_id}.checkpoint"
        )
        checkpoint_file.write_bytes(compressed_data)

        result = await checkpoint_storage.verify_checkpoint_integrity(sample_checkpoint)

        assert result is True

    @pytest.mark.asyncio
    async def test_verify_checkpoint_integrity_corrupted_data(
        self, checkpoint_storage, sample_checkpoint
    ):
        """Test integrity verification with corrupted data."""
        # Create corrupted checkpoint file
        checkpoint_file = (
            checkpoint_storage.checkpoint_path / f"{sample_checkpoint.checkpoint_id}.checkpoint"
        )
        checkpoint_file.write_text("corrupted data{")

        result = await checkpoint_storage.verify_checkpoint_integrity(sample_checkpoint)

        assert result is False

    def test_delete_checkpoint_files_error_handling(
        self, checkpoint_storage, sample_checkpoint
    ):
        """Test error handling during checkpoint deletion."""
        # Mock storage to fail
        with patch.object(
            checkpoint_storage.storage, "delete", side_effect=Exception("Delete failed")
        ):
            result = checkpoint_storage.delete_checkpoint_files(sample_checkpoint.checkpoint_id)

            assert result is False
