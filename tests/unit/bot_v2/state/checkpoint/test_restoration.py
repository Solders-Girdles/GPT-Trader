"""Tests for checkpoint restoration module."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, Mock

from bot_v2.state.checkpoint.models import Checkpoint
from bot_v2.state.checkpoint.restoration import CheckpointRestoration


class TestCheckpointRestoration:
    """Test suite for CheckpointRestoration class."""

    @pytest.fixture
    def mock_state_manager(self):
        """Create mock state manager."""
        manager = Mock()
        manager.get_keys_by_pattern = AsyncMock()
        manager.delete_state = AsyncMock()
        manager.set_state = AsyncMock()

        # Batch methods that count items
        async def batch_delete_mock(keys):
            return len(keys)

        async def batch_set_mock(items, ttl_seconds=None):
            return len(items)

        manager.batch_delete_state = AsyncMock(side_effect=batch_delete_mock)
        manager.batch_set_state = AsyncMock(side_effect=batch_set_mock)
        return manager

    @pytest.fixture
    def mock_storage(self):
        """Create mock storage."""
        storage = Mock()
        storage.verify_checkpoint_integrity = AsyncMock()
        return storage

    @pytest.fixture
    def mock_verification(self):
        """Create mock verification."""
        verification = Mock()
        verification.verify_restoration = AsyncMock()
        return verification

    @pytest.fixture
    def checkpoint_restoration(self, mock_state_manager, mock_storage, mock_verification):
        """Create CheckpointRestoration instance."""
        return CheckpointRestoration(mock_state_manager, mock_storage, mock_verification)

    @pytest.fixture
    def sample_checkpoint(self):
        """Create sample checkpoint."""
        return Checkpoint(
            checkpoint_id="test_checkpoint_001",
            version=1,
            size_bytes=1024,
            timestamp=datetime.fromisoformat("2024-01-01T00:00:00"),
            state_snapshot={
                "positions": {"position:BTC": {"symbol": "BTC", "qty": 1.0}},
                "orders": {"order:1": {"order_id": "1", "status": "open"}},
                "portfolio": {"total_value": 100000},
                "ml_models": {"ml_model:trend": {"type": "LSTM"}},
                "configuration": {"config:risk": {"max_position": 1000}},
                "performance_metrics": {"sharpe_ratio": 1.5},
            },
            consistency_hash="abc123",
            metadata={"test": "data"},
        )

    def test_initialization(
        self, checkpoint_restoration, mock_state_manager, mock_storage, mock_verification
    ):
        """Test CheckpointRestoration initialization."""
        assert checkpoint_restoration.state_manager == mock_state_manager
        assert checkpoint_restoration.storage == mock_storage
        assert checkpoint_restoration.verification == mock_verification

    @pytest.mark.asyncio
    async def test_restore_from_checkpoint_success(
        self,
        checkpoint_restoration,
        sample_checkpoint,
        mock_storage,
        mock_verification,
        mock_state_manager,
    ):
        """Test successful checkpoint restoration."""
        # Setup mocks
        mock_storage.verify_checkpoint_integrity.return_value = True
        mock_verification.verify_restoration.return_value = True
        mock_state_manager.get_keys_by_pattern.return_value = []
        mock_state_manager.set_state.return_value = True

        result = await checkpoint_restoration.restore_from_checkpoint(sample_checkpoint)

        assert result is True
        mock_storage.verify_checkpoint_integrity.assert_called_once_with(sample_checkpoint)
        mock_verification.verify_restoration.assert_called_once_with(sample_checkpoint)

    @pytest.mark.asyncio
    async def test_restore_integrity_check_failure(
        self, checkpoint_restoration, sample_checkpoint, mock_storage
    ):
        """Test restoration fails when integrity check fails."""
        mock_storage.verify_checkpoint_integrity.return_value = False

        result = await checkpoint_restoration.restore_from_checkpoint(sample_checkpoint)

        assert result is False
        mock_storage.verify_checkpoint_integrity.assert_called_once()

    @pytest.mark.asyncio
    async def test_restore_verification_failure(
        self,
        checkpoint_restoration,
        sample_checkpoint,
        mock_storage,
        mock_verification,
        mock_state_manager,
    ):
        """Test restoration handles verification failure."""
        mock_storage.verify_checkpoint_integrity.return_value = True
        mock_verification.verify_restoration.return_value = False
        mock_state_manager.get_keys_by_pattern.return_value = []
        mock_state_manager.set_state.return_value = True

        result = await checkpoint_restoration.restore_from_checkpoint(sample_checkpoint)

        # Should fail even though state was restored
        assert result is False

    @pytest.mark.asyncio
    async def test_pause_and_resume_called(
        self,
        checkpoint_restoration,
        sample_checkpoint,
        mock_storage,
        mock_verification,
        mock_state_manager,
    ):
        """Test that pause and resume operations are called."""
        mock_storage.verify_checkpoint_integrity.return_value = True
        mock_verification.verify_restoration.return_value = True
        mock_state_manager.get_keys_by_pattern.return_value = []
        mock_state_manager.set_state.return_value = True

        await checkpoint_restoration.restore_from_checkpoint(sample_checkpoint)

        # Check pause was called
        pause_calls = [
            call
            for call in mock_state_manager.set_state.call_args_list
            if len(call[0]) > 0 and call[0][0] == "system:trading_paused"
        ]
        assert len(pause_calls) >= 2  # Once for pause, once for resume

    @pytest.mark.asyncio
    async def test_clear_current_state(self, checkpoint_restoration, mock_state_manager):
        """Test clearing current state before restoration."""
        position_keys = ["position:BTC", "position:ETH"]
        order_keys = ["order:1", "order:2"]

        async def get_keys_side_effect(pattern):
            if pattern == "position:*":
                return position_keys
            elif pattern == "order:*":
                return order_keys
            return []

        mock_state_manager.get_keys_by_pattern.side_effect = get_keys_side_effect

        await checkpoint_restoration._clear_current_state()

        # Should call batch_delete_state with all 4 keys
        assert mock_state_manager.batch_delete_state.call_count == 1
        call_args = mock_state_manager.batch_delete_state.call_args[0][0]
        assert len(call_args) == 4
        assert set(call_args) == set(position_keys + order_keys)

    @pytest.mark.asyncio
    async def test_clear_current_state_handles_errors(
        self, checkpoint_restoration, mock_state_manager, caplog
    ):
        """Test error handling in clear_current_state."""
        mock_state_manager.get_keys_by_pattern.side_effect = Exception("Redis error")

        await checkpoint_restoration._clear_current_state()

        # Should log error
        assert "Failed to clear current state" in caplog.text

    @pytest.mark.asyncio
    async def test_restore_state_from_snapshot_positions(
        self, checkpoint_restoration, sample_checkpoint, mock_state_manager
    ):
        """Test restoring positions from snapshot."""
        from bot_v2.state.state_manager import StateCategory

        result = await checkpoint_restoration._restore_state_from_snapshot(
            sample_checkpoint.state_snapshot
        )

        assert result is True

        # Check positions were restored via batch_set_state
        assert mock_state_manager.batch_set_state.call_count == 1
        items = mock_state_manager.batch_set_state.call_args[0][0]

        # Check that position:BTC is in items with HOT category
        position_key = "position:BTC"
        assert position_key in items
        value, category = items[position_key]
        assert category == StateCategory.HOT
        assert value["symbol"] == "BTC"

    @pytest.mark.asyncio
    async def test_restore_state_from_snapshot_orders(
        self, checkpoint_restoration, sample_checkpoint, mock_state_manager
    ):
        """Test restoring orders from snapshot."""
        from bot_v2.state.state_manager import StateCategory

        result = await checkpoint_restoration._restore_state_from_snapshot(
            sample_checkpoint.state_snapshot
        )

        assert result is True

        # Check orders were restored via batch_set_state
        assert mock_state_manager.batch_set_state.call_count == 1
        items = mock_state_manager.batch_set_state.call_args[0][0]

        # Check that order:1 is in items with HOT category
        order_key = "order:1"
        assert order_key in items
        value, category = items[order_key]
        assert category == StateCategory.HOT
        assert value["order_id"] == "1"

    @pytest.mark.asyncio
    async def test_restore_state_from_snapshot_portfolio(
        self, checkpoint_restoration, sample_checkpoint, mock_state_manager
    ):
        """Test restoring portfolio from snapshot."""
        from bot_v2.state.state_manager import StateCategory

        result = await checkpoint_restoration._restore_state_from_snapshot(
            sample_checkpoint.state_snapshot
        )

        assert result is True

        # Check portfolio was restored via batch_set_state
        assert mock_state_manager.batch_set_state.call_count == 1
        items = mock_state_manager.batch_set_state.call_args[0][0]

        # Check that portfolio_current is in items with HOT category
        assert "portfolio_current" in items
        value, category = items["portfolio_current"]
        assert category == StateCategory.HOT
        assert value["total_value"] == 100000

    @pytest.mark.asyncio
    async def test_restore_state_from_snapshot_ml_models(
        self, checkpoint_restoration, sample_checkpoint, mock_state_manager
    ):
        """Test restoring ML models from snapshot."""
        from bot_v2.state.state_manager import StateCategory

        result = await checkpoint_restoration._restore_state_from_snapshot(
            sample_checkpoint.state_snapshot
        )

        assert result is True

        # Check ML models were restored via batch_set_state to WARM tier
        assert mock_state_manager.batch_set_state.call_count == 1
        items = mock_state_manager.batch_set_state.call_args[0][0]

        # Check that ml_model:trend is in items with WARM category
        ml_key = "ml_model:trend"
        assert ml_key in items
        value, category = items[ml_key]
        assert category == StateCategory.WARM
        assert value["type"] == "LSTM"

    @pytest.mark.asyncio
    async def test_restore_state_from_snapshot_configuration(
        self, checkpoint_restoration, sample_checkpoint, mock_state_manager
    ):
        """Test restoring configuration from snapshot."""
        from bot_v2.state.state_manager import StateCategory

        result = await checkpoint_restoration._restore_state_from_snapshot(
            sample_checkpoint.state_snapshot
        )

        assert result is True

        # Check configuration was restored via batch_set_state to WARM tier
        assert mock_state_manager.batch_set_state.call_count == 1
        items = mock_state_manager.batch_set_state.call_args[0][0]

        # Check that config:risk is in items with WARM category
        config_key = "config:risk"
        assert config_key in items
        value, category = items[config_key]
        assert category == StateCategory.WARM
        assert value["max_position"] == 1000

    @pytest.mark.asyncio
    async def test_restore_state_from_snapshot_metrics(
        self, checkpoint_restoration, sample_checkpoint, mock_state_manager
    ):
        """Test restoring performance metrics from snapshot."""
        from bot_v2.state.state_manager import StateCategory

        result = await checkpoint_restoration._restore_state_from_snapshot(
            sample_checkpoint.state_snapshot
        )

        assert result is True

        # Check metrics were restored via batch_set_state to WARM tier
        assert mock_state_manager.batch_set_state.call_count == 1
        items = mock_state_manager.batch_set_state.call_args[0][0]

        # Check that performance_metrics is in items with WARM category
        assert "performance_metrics" in items
        value, category = items["performance_metrics"]
        assert category == StateCategory.WARM
        assert value["sharpe_ratio"] == 1.5

    @pytest.mark.asyncio
    async def test_restore_state_from_snapshot_empty(
        self, checkpoint_restoration, mock_state_manager
    ):
        """Test restoring from empty snapshot."""
        mock_state_manager.set_state.return_value = True

        result = await checkpoint_restoration._restore_state_from_snapshot({})

        # Should return False when nothing to restore
        assert result is False

    @pytest.mark.asyncio
    async def test_restore_state_from_snapshot_counts_restored(
        self, checkpoint_restoration, sample_checkpoint, mock_state_manager, caplog
    ):
        """Test that restored count is logged."""
        import logging

        mock_state_manager.set_state.return_value = True

        with caplog.at_level(logging.INFO):
            await checkpoint_restoration._restore_state_from_snapshot(
                sample_checkpoint.state_snapshot
            )

        # Should log number of restored entries
        assert "Restored" in caplog.text
        assert "state entries" in caplog.text

    @pytest.mark.asyncio
    async def test_restore_state_from_snapshot_handles_errors(
        self, checkpoint_restoration, sample_checkpoint, mock_state_manager, caplog
    ):
        """Test error handling during state restoration."""
        mock_state_manager.batch_set_state.side_effect = Exception("State error")

        result = await checkpoint_restoration._restore_state_from_snapshot(
            sample_checkpoint.state_snapshot
        )

        assert result is False
        assert "State restoration failed" in caplog.text

    @pytest.mark.asyncio
    async def test_pause_trading_operations(self, checkpoint_restoration, mock_state_manager):
        """Test pausing trading operations."""
        await checkpoint_restoration._pause_trading_operations()

        # Should set trading_paused flag to True
        mock_state_manager.set_state.assert_called_with("system:trading_paused", True)

    @pytest.mark.asyncio
    async def test_pause_trading_handles_errors(
        self, checkpoint_restoration, mock_state_manager, caplog
    ):
        """Test error handling when pausing trading."""
        mock_state_manager.set_state.side_effect = Exception("State error")

        await checkpoint_restoration._pause_trading_operations()

        # Should log error
        assert "Failed to pause trading operations" in caplog.text

    @pytest.mark.asyncio
    async def test_resume_trading_operations(self, checkpoint_restoration, mock_state_manager):
        """Test resuming trading operations."""
        await checkpoint_restoration._resume_trading_operations()

        # Should set trading_paused flag to False
        mock_state_manager.set_state.assert_called_with("system:trading_paused", False)

    @pytest.mark.asyncio
    async def test_resume_called_on_exception(
        self, checkpoint_restoration, sample_checkpoint, mock_storage, mock_state_manager
    ):
        """Test that resume is called even when restoration fails."""
        mock_storage.verify_checkpoint_integrity.side_effect = Exception("Integrity error")

        result = await checkpoint_restoration.restore_from_checkpoint(sample_checkpoint)

        assert result is False
        # Resume should still be called
        resume_calls = [
            call
            for call in mock_state_manager.set_state.call_args_list
            if len(call[0]) > 0 and call[0][0] == "system:trading_paused" and call[0][1] is False
        ]
        assert len(resume_calls) == 1

    @pytest.mark.asyncio
    async def test_restore_logs_progress(
        self,
        checkpoint_restoration,
        sample_checkpoint,
        mock_storage,
        mock_verification,
        mock_state_manager,
        caplog,
    ):
        """Test that restoration logs progress."""
        import logging

        mock_storage.verify_checkpoint_integrity.return_value = True
        mock_verification.verify_restoration.return_value = True
        mock_state_manager.get_keys_by_pattern.return_value = []
        mock_state_manager.set_state.return_value = True

        with caplog.at_level(logging.INFO):
            await checkpoint_restoration.restore_from_checkpoint(sample_checkpoint)

        # Should log start and success
        assert "Restoring from checkpoint" in caplog.text
        assert "Successfully restored from checkpoint" in caplog.text
