"""Tests for checkpoint verification module."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, Mock

from bot_v2.state.checkpoint.models import Checkpoint
from bot_v2.state.checkpoint.verification import CheckpointVerification


class TestCheckpointVerification:
    """Test suite for CheckpointVerification class."""

    @pytest.fixture
    def mock_state_manager(self):
        """Create mock state manager."""
        manager = Mock()
        manager.get_state = AsyncMock()
        manager.get_keys_by_pattern = AsyncMock()
        return manager

    @pytest.fixture
    def mock_storage(self):
        """Create mock storage."""
        storage = Mock()
        storage.calculate_consistency_hash = Mock()
        return storage

    @pytest.fixture
    def checkpoint_verification(self, mock_state_manager, mock_storage):
        """Create CheckpointVerification instance."""
        return CheckpointVerification(mock_state_manager, mock_storage)

    @pytest.fixture
    def sample_checkpoint(self):
        """Create sample checkpoint."""
        return Checkpoint(
            checkpoint_id="test_checkpoint_001",
            version=1,
            size_bytes=1024,
            timestamp=datetime.fromisoformat("2024-01-01T00:00:00"),
            state_snapshot={
                "timestamp": "2024-01-01T00:00:00",
                "positions": {
                    "position:BTC": {"symbol": "BTC", "qty": 1.0},
                    "position:ETH": {"symbol": "ETH", "qty": 2.0}
                },
                "portfolio": {"total_value": 100000}
            },
            consistency_hash="abc123",
            metadata={"test": "data"}
        )

    def test_initialization(self, checkpoint_verification, mock_state_manager, mock_storage):
        """Test CheckpointVerification initialization."""
        assert checkpoint_verification.state_manager == mock_state_manager
        assert checkpoint_verification.storage == mock_storage

    @pytest.mark.asyncio
    async def test_verify_checkpoint_success(
        self, checkpoint_verification, sample_checkpoint, mock_storage
    ):
        """Test successful checkpoint verification."""
        mock_storage.calculate_consistency_hash.return_value = "abc123"

        result = await checkpoint_verification.verify_checkpoint(sample_checkpoint)

        assert result is True
        mock_storage.calculate_consistency_hash.assert_called_once_with(sample_checkpoint.state_snapshot)

    @pytest.mark.asyncio
    async def test_verify_checkpoint_hash_mismatch(
        self, checkpoint_verification, sample_checkpoint, mock_storage, caplog
    ):
        """Test checkpoint verification fails on hash mismatch."""
        mock_storage.calculate_consistency_hash.return_value = "different_hash"

        result = await checkpoint_verification.verify_checkpoint(sample_checkpoint)

        assert result is False
        assert "hash mismatch" in caplog.text
        assert sample_checkpoint.checkpoint_id in caplog.text

    @pytest.mark.asyncio
    async def test_verify_checkpoint_missing_timestamp(
        self, checkpoint_verification, mock_storage, caplog
    ):
        """Test checkpoint verification fails when timestamp missing."""
        checkpoint_no_timestamp = Checkpoint(
            checkpoint_id="test_no_timestamp",
            version=1,
            size_bytes=512,
            timestamp=datetime.fromisoformat("2024-01-01T00:00:00"),
            state_snapshot={
                "positions": {},
                "portfolio": {}
            },
            consistency_hash="xyz789",
            metadata={}
        )

        mock_storage.calculate_consistency_hash.return_value = "xyz789"

        result = await checkpoint_verification.verify_checkpoint(checkpoint_no_timestamp)

        assert result is False
        assert "missing timestamp" in caplog.text

    @pytest.mark.asyncio
    async def test_verify_checkpoint_handles_errors(
        self, checkpoint_verification, sample_checkpoint, mock_storage, caplog
    ):
        """Test error handling during checkpoint verification."""
        mock_storage.calculate_consistency_hash.side_effect = Exception("Hash calculation error")

        result = await checkpoint_verification.verify_checkpoint(sample_checkpoint)

        assert result is False
        assert "Checkpoint verification failed" in caplog.text

    @pytest.mark.asyncio
    async def test_verify_restoration_success(
        self, checkpoint_verification, sample_checkpoint, mock_state_manager
    ):
        """Test successful restoration verification."""
        mock_state_manager.get_state.return_value = {"total_value": 100000}
        mock_state_manager.get_keys_by_pattern.return_value = ["position:BTC", "position:ETH"]

        result = await checkpoint_verification.verify_restoration(sample_checkpoint)

        assert result is True
        mock_state_manager.get_state.assert_called_once_with("portfolio_current")
        mock_state_manager.get_keys_by_pattern.assert_called_once_with("position:*")

    @pytest.mark.asyncio
    async def test_verify_restoration_missing_portfolio(
        self, checkpoint_verification, sample_checkpoint, mock_state_manager, caplog
    ):
        """Test restoration verification fails when portfolio not restored."""
        mock_state_manager.get_state.return_value = None

        result = await checkpoint_verification.verify_restoration(sample_checkpoint)

        assert result is False
        assert "Portfolio data not restored" in caplog.text

    @pytest.mark.asyncio
    async def test_verify_restoration_position_count_mismatch(
        self, checkpoint_verification, sample_checkpoint, mock_state_manager, caplog
    ):
        """Test restoration verification fails on position count mismatch."""
        mock_state_manager.get_state.return_value = {"total_value": 100000}
        # Only 1 position restored, but checkpoint has 2
        mock_state_manager.get_keys_by_pattern.return_value = ["position:BTC"]

        result = await checkpoint_verification.verify_restoration(sample_checkpoint)

        assert result is False
        assert "Position count mismatch" in caplog.text
        assert "expected 2" in caplog.text
        assert "got 1" in caplog.text

    @pytest.mark.asyncio
    async def test_verify_restoration_more_positions_than_expected(
        self, checkpoint_verification, sample_checkpoint, mock_state_manager, caplog
    ):
        """Test restoration verification fails when more positions restored than expected."""
        mock_state_manager.get_state.return_value = {"total_value": 100000}
        # 3 positions restored, but checkpoint has 2
        mock_state_manager.get_keys_by_pattern.return_value = [
            "position:BTC", "position:ETH", "position:SOL"
        ]

        result = await checkpoint_verification.verify_restoration(sample_checkpoint)

        assert result is False
        assert "Position count mismatch" in caplog.text
        assert "expected 2" in caplog.text
        assert "got 3" in caplog.text

    @pytest.mark.asyncio
    async def test_verify_restoration_empty_positions(
        self, checkpoint_verification, mock_state_manager
    ):
        """Test restoration verification with no positions."""
        checkpoint_no_positions = Checkpoint(
            checkpoint_id="test_no_positions",
            version=1,
            size_bytes=256,
            timestamp=datetime.fromisoformat("2024-01-01T00:00:00"),
            state_snapshot={
                "timestamp": "2024-01-01T00:00:00",
                "positions": {},
                "portfolio": {"total_value": 50000}
            },
            consistency_hash="xyz",
            metadata={}
        )

        mock_state_manager.get_state.return_value = {"total_value": 50000}
        mock_state_manager.get_keys_by_pattern.return_value = []

        result = await checkpoint_verification.verify_restoration(checkpoint_no_positions)

        assert result is True

    @pytest.mark.asyncio
    async def test_verify_restoration_handles_errors(
        self, checkpoint_verification, sample_checkpoint, mock_state_manager, caplog
    ):
        """Test error handling during restoration verification."""
        mock_state_manager.get_state.side_effect = Exception("State error")

        result = await checkpoint_verification.verify_restoration(sample_checkpoint)

        assert result is False
        assert "Restoration verification failed" in caplog.text

    @pytest.mark.asyncio
    async def test_verify_restoration_missing_positions_key(
        self, checkpoint_verification, mock_state_manager
    ):
        """Test restoration verification when positions key is missing from snapshot."""
        checkpoint_no_pos_key = Checkpoint(
            checkpoint_id="test_no_pos_key",
            version=1,
            size_bytes=128,
            timestamp=datetime.fromisoformat("2024-01-01T00:00:00"),
            state_snapshot={
                "timestamp": "2024-01-01T00:00:00",
                "portfolio": {"total_value": 50000}
            },
            consistency_hash="xyz",
            metadata={}
        )

        mock_state_manager.get_state.return_value = {"total_value": 50000}
        mock_state_manager.get_keys_by_pattern.return_value = []

        result = await checkpoint_verification.verify_restoration(checkpoint_no_pos_key)

        # Should succeed with 0 expected positions
        assert result is True

    @pytest.mark.asyncio
    async def test_verify_checkpoint_logs_checkpoint_id(
        self, checkpoint_verification, sample_checkpoint, mock_storage, caplog
    ):
        """Test that checkpoint ID is logged in verification messages."""
        import logging

        mock_storage.calculate_consistency_hash.return_value = "wrong_hash"

        with caplog.at_level(logging.ERROR):
            await checkpoint_verification.verify_checkpoint(sample_checkpoint)

        assert sample_checkpoint.checkpoint_id in caplog.text
