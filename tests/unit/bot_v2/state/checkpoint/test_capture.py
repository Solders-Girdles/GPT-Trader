"""Tests for state checkpoint capture module."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

from bot_v2.state.checkpoint.capture import StateCapture


class TestStateCapture:
    """Test suite for StateCapture class."""

    @pytest.fixture
    def mock_state_manager(self):
        """Create mock state manager."""
        manager = Mock()
        manager.get_keys_by_pattern = AsyncMock()
        manager.get_state = AsyncMock()
        return manager

    @pytest.fixture
    def state_capture(self, mock_state_manager):
        """Create StateCapture instance."""
        return StateCapture(mock_state_manager)

    @pytest.mark.asyncio
    async def test_initialization(self, mock_state_manager):
        """Test StateCapture initialization."""
        capture = StateCapture(mock_state_manager)
        assert capture.state_manager == mock_state_manager

    @pytest.mark.asyncio
    async def test_capture_system_state_structure(self, state_capture, mock_state_manager):
        """Test that captured state has expected structure."""
        # Setup mocks
        mock_state_manager.get_keys_by_pattern.return_value = []
        mock_state_manager.get_state.return_value = None

        result = await state_capture.capture_system_state()

        # Check structure
        assert "timestamp" in result
        assert "positions" in result
        assert "orders" in result
        assert "portfolio" in result
        assert "ml_models" in result
        assert "configuration" in result
        assert "performance_metrics" in result
        assert "market_data_cache" in result

        # Check types
        assert isinstance(result["positions"], dict)
        assert isinstance(result["orders"], dict)
        assert isinstance(result["portfolio"], dict)

    @pytest.mark.asyncio
    async def test_capture_positions(self, state_capture, mock_state_manager):
        """Test capturing position data."""
        # Setup mocks
        position_keys = ["position:BTC-USD", "position:ETH-USD"]
        position_data = {"symbol": "BTC-USD", "quantity": 1.5, "entry_price": 50000}

        mock_state_manager.get_keys_by_pattern.side_effect = lambda pattern: {
            "position:*": position_keys,
            "order:*": [],
            "ml_model:*": [],
            "config:*": [],
        }.get(pattern, [])

        mock_state_manager.get_state.return_value = position_data

        result = await state_capture.capture_system_state()

        # Should have captured positions
        assert len(result["positions"]) == 2
        assert "position:BTC-USD" in result["positions"]
        assert "position:ETH-USD" in result["positions"]

    @pytest.mark.asyncio
    async def test_capture_open_orders_only(self, state_capture, mock_state_manager):
        """Test that only non-filled orders are captured."""
        # Setup mocks
        order_keys = ["order:1", "order:2", "order:3"]
        orders = [
            {"order_id": "1", "status": "open"},
            {"order_id": "2", "status": "filled"},  # Should be excluded
            {"order_id": "3", "status": "pending"},
        ]

        mock_state_manager.get_keys_by_pattern.side_effect = lambda pattern: {
            "position:*": [],
            "order:*": order_keys,
            "ml_model:*": [],
            "config:*": [],
        }.get(pattern, [])

        # Return different order data for each call
        call_count = [0]

        async def get_state_side_effect(key):
            if key.startswith("order:"):
                idx = call_count[0]
                call_count[0] += 1
                return orders[idx] if idx < len(orders) else None
            return None

        mock_state_manager.get_state.side_effect = get_state_side_effect

        result = await state_capture.capture_system_state()

        # Should only have non-filled orders
        assert len(result["orders"]) == 2
        # Filled order should not be in results
        filled_in_results = any(v.get("status") == "filled" for v in result["orders"].values())
        assert not filled_in_results

    @pytest.mark.asyncio
    async def test_capture_portfolio(self, state_capture, mock_state_manager):
        """Test capturing portfolio data."""
        portfolio_data = {"total_value": 100000, "cash": 50000, "positions_value": 50000}

        mock_state_manager.get_keys_by_pattern.return_value = []

        async def get_state_side_effect(key):
            if key == "portfolio_current":
                return portfolio_data
            return None

        mock_state_manager.get_state.side_effect = get_state_side_effect

        result = await state_capture.capture_system_state()

        assert result["portfolio"] == portfolio_data

    @pytest.mark.asyncio
    async def test_capture_ml_models(self, state_capture, mock_state_manager):
        """Test capturing ML model states."""
        ml_keys = ["ml_model:trend", "ml_model:volatility"]
        ml_data = {"model_type": "LSTM", "accuracy": 0.85}

        mock_state_manager.get_keys_by_pattern.side_effect = lambda pattern: {
            "position:*": [],
            "order:*": [],
            "ml_model:*": ml_keys,
            "config:*": [],
        }.get(pattern, [])

        mock_state_manager.get_state.return_value = ml_data

        result = await state_capture.capture_system_state()

        assert len(result["ml_models"]) == 2
        assert "ml_model:trend" in result["ml_models"]

    @pytest.mark.asyncio
    async def test_capture_configuration(self, state_capture, mock_state_manager):
        """Test capturing configuration data."""
        config_keys = ["config:risk", "config:trading"]
        config_data = {"max_position_size": 1000}

        mock_state_manager.get_keys_by_pattern.side_effect = lambda pattern: {
            "position:*": [],
            "order:*": [],
            "ml_model:*": [],
            "config:*": config_keys,
        }.get(pattern, [])

        mock_state_manager.get_state.return_value = config_data

        result = await state_capture.capture_system_state()

        assert len(result["configuration"]) == 2

    @pytest.mark.asyncio
    async def test_capture_performance_metrics(self, state_capture, mock_state_manager):
        """Test capturing performance metrics."""
        metrics_data = {"total_return": 15.5, "sharpe_ratio": 1.8, "max_drawdown": -5.2}

        mock_state_manager.get_keys_by_pattern.return_value = []

        async def get_state_side_effect(key):
            if key == "performance_metrics":
                return metrics_data
            return None

        mock_state_manager.get_state.side_effect = get_state_side_effect

        result = await state_capture.capture_system_state()

        assert result["performance_metrics"] == metrics_data

    @pytest.mark.asyncio
    async def test_timestamp_format(self, state_capture, mock_state_manager):
        """Test that timestamp is in ISO format."""
        mock_state_manager.get_keys_by_pattern.return_value = []

        result = await state_capture.capture_system_state()

        # Should be able to parse timestamp
        timestamp_str = result["timestamp"]
        parsed = datetime.fromisoformat(timestamp_str)
        assert isinstance(parsed, datetime)

    @pytest.mark.asyncio
    async def test_handles_none_values(self, state_capture, mock_state_manager):
        """Test handling of None values from state manager."""
        mock_state_manager.get_keys_by_pattern.return_value = ["position:BTC"]
        mock_state_manager.get_state.return_value = None

        result = await state_capture.capture_system_state()

        # Should not include None values
        assert len(result["positions"]) == 0

    @pytest.mark.asyncio
    async def test_error_handling(self, state_capture, mock_state_manager, caplog):
        """Test error handling during capture."""
        mock_state_manager.get_keys_by_pattern.side_effect = Exception("Redis error")

        result = await state_capture.capture_system_state()

        # Should return empty dict on error
        assert result == {}

        # Should log error
        assert "Failed to capture system state" in caplog.text

    @pytest.mark.asyncio
    async def test_debug_logging(self, state_capture, mock_state_manager, caplog):
        """Test debug logging output."""
        import logging

        position_keys = ["position:BTC"]
        order_keys = ["order:1", "order:2"]

        mock_state_manager.get_keys_by_pattern.side_effect = lambda pattern: {
            "position:*": position_keys,
            "order:*": order_keys,
            "ml_model:*": [],
            "config:*": [],
        }.get(pattern, [])

        mock_state_manager.get_state.return_value = {"data": "test"}

        with caplog.at_level(logging.DEBUG):
            await state_capture.capture_system_state()

        # Should log number of positions and orders
        assert "Captured state" in caplog.text
        assert "1 positions" in caplog.text
        assert "2 orders" in caplog.text

    @pytest.mark.asyncio
    async def test_empty_state_capture(self, state_capture, mock_state_manager):
        """Test capturing when no state exists."""
        mock_state_manager.get_keys_by_pattern.return_value = []
        mock_state_manager.get_state.return_value = None

        result = await state_capture.capture_system_state()

        # Should have structure but empty collections
        assert len(result["positions"]) == 0
        assert len(result["orders"]) == 0
        assert result["portfolio"] == {}
        assert result["performance_metrics"] == {}
