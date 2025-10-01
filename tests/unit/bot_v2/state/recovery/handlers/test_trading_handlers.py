"""Tests for trading recovery handlers"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, Mock
from bot_v2.state.recovery.handlers.trading import TradingRecoveryHandlers
from bot_v2.state.recovery.models import (
    FailureEvent,
    FailureType,
    RecoveryMode,
    RecoveryOperation,
    RecoveryStatus,
)


@pytest.fixture
def mock_state_manager():
    """Mock state manager"""
    manager = Mock()
    manager.get_state = AsyncMock()
    manager.set_state = AsyncMock()
    manager.delete_state = AsyncMock()
    manager.get_keys_by_pattern = AsyncMock()
    return manager


@pytest.fixture
def trading_handlers(mock_state_manager):
    """Create TradingRecoveryHandlers instance"""
    return TradingRecoveryHandlers(mock_state_manager)


@pytest.fixture
def recovery_operation():
    """Create sample recovery operation"""
    failure_event = FailureEvent(
        failure_type=FailureType.TRADING_ENGINE_CRASH,
        timestamp=datetime.utcnow(),
        severity="critical",
        affected_components=["order_execution"],
        error_message="Trading engine crashed",
    )

    return RecoveryOperation(
        operation_id="REC_TEST_001",
        failure_event=failure_event,
        recovery_mode=RecoveryMode.AUTOMATIC,
        status=RecoveryStatus.IN_PROGRESS,
        started_at=datetime.utcnow(),
    )


class TestTradingRecoveryHandlers:
    """Test suite for TradingRecoveryHandlers"""

    @pytest.mark.asyncio
    async def test_recover_trading_engine_success(self, trading_handlers, recovery_operation):
        """Test successful trading engine recovery"""
        # Mock pending orders
        trading_handlers.state_manager.get_keys_by_pattern.side_effect = [
            ["order:1", "order:2", "order:3"],  # order keys
            ["position:BTC", "position:ETH"],  # position keys
        ]

        # Mock order data
        async def mock_get_state(key):
            if key.startswith("order:"):
                return {"status": "pending", "symbol": "BTC-USD"}
            elif key.startswith("position:"):
                return {"symbol": "BTC", "quantity": 1.5, "entry_price": 50000}
            elif key == "portfolio_current":
                return {"balance": 10000, "equity": 15000}
            return None

        trading_handlers.state_manager.get_state = AsyncMock(side_effect=mock_get_state)

        success = await trading_handlers.recover_trading_engine(recovery_operation)

        assert success is True
        assert len(recovery_operation.actions_taken) > 0
        assert "Cancelled 3 pending orders" in recovery_operation.actions_taken[1]
        assert "Restored portfolio state" in recovery_operation.actions_taken[2]
        assert "Trading engine recovery completed" in recovery_operation.actions_taken[3]

        # Verify trading engine status was updated
        trading_handlers.state_manager.set_state.assert_any_call(
            "system:trading_engine_status", "recovered"
        )

    @pytest.mark.asyncio
    async def test_recover_trading_engine_no_pending_orders(
        self, trading_handlers, recovery_operation
    ):
        """Test trading engine recovery with no pending orders"""
        trading_handlers.state_manager.get_keys_by_pattern.side_effect = [
            [],  # no order keys
            [],  # no position keys
        ]
        trading_handlers.state_manager.get_state.return_value = {
            "balance": 10000,
            "equity": 15000,
        }

        success = await trading_handlers.recover_trading_engine(recovery_operation)

        assert success is True
        assert "Cancelled 0 pending orders" in recovery_operation.actions_taken[1]

    @pytest.mark.asyncio
    async def test_recover_trading_engine_no_portfolio_data(
        self, trading_handlers, recovery_operation
    ):
        """Test trading engine recovery without portfolio data"""
        trading_handlers.state_manager.get_keys_by_pattern.return_value = []
        trading_handlers.state_manager.get_state.return_value = None

        success = await trading_handlers.recover_trading_engine(recovery_operation)

        assert success is True
        # Should still complete but without portfolio restoration

    @pytest.mark.asyncio
    async def test_recover_trading_engine_invalid_positions(
        self, trading_handlers, recovery_operation
    ):
        """Test trading engine recovery with invalid positions"""
        trading_handlers.state_manager.get_keys_by_pattern.side_effect = [
            [],  # no order keys
            ["position:BTC", "position:ETH"],  # position keys
        ]

        async def mock_get_state(key):
            if key == "position:BTC":
                return {"symbol": "BTC", "quantity": 1.5, "entry_price": 50000}  # Valid
            elif key == "position:ETH":
                return {"symbol": "ETH"}  # Invalid - missing required fields
            elif key == "portfolio_current":
                return {"balance": 10000}
            return None

        trading_handlers.state_manager.get_state = AsyncMock(side_effect=mock_get_state)

        success = await trading_handlers.recover_trading_engine(recovery_operation)

        assert success is True
        # Invalid position should be deleted
        trading_handlers.state_manager.delete_state.assert_called_once_with("position:ETH")

    @pytest.mark.asyncio
    async def test_recover_trading_engine_exception(self, trading_handlers, recovery_operation):
        """Test trading engine recovery with exception"""
        trading_handlers.state_manager.get_keys_by_pattern.side_effect = Exception("State error")

        success = await trading_handlers.recover_trading_engine(recovery_operation)

        assert success is False
        assert any("error" in action.lower() for action in recovery_operation.actions_taken)

    @pytest.mark.asyncio
    async def test_recover_ml_models_success(self, trading_handlers, recovery_operation):
        """Test successful ML model recovery"""
        trading_handlers.state_manager.get_keys_by_pattern.return_value = [
            "ml_model:momentum",
            "ml_model:reversal",
        ]

        async def mock_get_state(key):
            return {
                "current_version": "v2.0",
                "last_stable_version": "v1.5",
                "accuracy": 0.75,
            }

        trading_handlers.state_manager.get_state = AsyncMock(side_effect=mock_get_state)

        success = await trading_handlers.recover_ml_models(recovery_operation)

        assert success is True
        assert "Recovered 2 ML models" in recovery_operation.actions_taken[1]

        # Verify models were reset to stable version
        assert trading_handlers.state_manager.set_state.call_count >= 2

    @pytest.mark.asyncio
    async def test_recover_ml_models_no_stable_version(self, trading_handlers, recovery_operation):
        """Test ML model recovery without stable versions"""
        trading_handlers.state_manager.get_keys_by_pattern.return_value = ["ml_model:momentum"]

        async def mock_get_state(key):
            return {"current_version": "v2.0", "accuracy": 0.75}  # No last_stable_version

        trading_handlers.state_manager.get_state = AsyncMock(side_effect=mock_get_state)

        success = await trading_handlers.recover_ml_models(recovery_operation)

        assert success is True
        assert "Recovered 0 ML models" in recovery_operation.actions_taken[1]

    @pytest.mark.asyncio
    async def test_recover_ml_models_no_models(self, trading_handlers, recovery_operation):
        """Test ML model recovery with no models"""
        trading_handlers.state_manager.get_keys_by_pattern.return_value = []

        success = await trading_handlers.recover_ml_models(recovery_operation)

        assert success is True
        assert "using baseline strategies" in recovery_operation.actions_taken[1].lower()
        trading_handlers.state_manager.set_state.assert_called_once_with(
            "system:ml_models_available", False
        )

    @pytest.mark.asyncio
    async def test_recover_ml_models_with_exception(self, trading_handlers, recovery_operation):
        """Test ML model recovery with exception"""
        trading_handlers.state_manager.get_keys_by_pattern.side_effect = Exception("State error")

        success = await trading_handlers.recover_ml_models(recovery_operation)

        assert success is False

    def test_validate_position_valid(self, trading_handlers):
        """Test position validation with valid data"""
        position = {"symbol": "BTC", "quantity": 1.5, "entry_price": 50000}

        result = trading_handlers._validate_position(position)

        assert result is True

    def test_validate_position_missing_symbol(self, trading_handlers):
        """Test position validation with missing symbol"""
        position = {"quantity": 1.5, "entry_price": 50000}

        result = trading_handlers._validate_position(position)

        assert result is False

    def test_validate_position_missing_quantity(self, trading_handlers):
        """Test position validation with missing quantity"""
        position = {"symbol": "BTC", "entry_price": 50000}

        result = trading_handlers._validate_position(position)

        assert result is False

    def test_validate_position_missing_entry_price(self, trading_handlers):
        """Test position validation with missing entry price"""
        position = {"symbol": "BTC", "quantity": 1.5}

        result = trading_handlers._validate_position(position)

        assert result is False

    def test_validate_position_empty(self, trading_handlers):
        """Test position validation with empty dict"""
        position = {}

        result = trading_handlers._validate_position(position)

        assert result is False

    def test_validate_position_with_extra_fields(self, trading_handlers):
        """Test position validation with extra fields"""
        position = {
            "symbol": "BTC",
            "quantity": 1.5,
            "entry_price": 50000,
            "unrealized_pnl": 1000,
            "leverage": 2,
        }

        result = trading_handlers._validate_position(position)

        assert result is True

    @pytest.mark.asyncio
    async def test_recover_trading_engine_cancels_only_pending(
        self, trading_handlers, recovery_operation
    ):
        """Test that only pending orders are cancelled"""
        trading_handlers.state_manager.get_keys_by_pattern.side_effect = [
            ["order:1", "order:2", "order:3"],
            [],
        ]

        # Mix of pending and filled orders
        async def mock_get_state(key):
            if key == "order:1":
                return {"status": "pending"}
            elif key == "order:2":
                return {"status": "filled"}
            elif key == "order:3":
                return {"status": "pending"}
            elif key == "portfolio_current":
                return {"balance": 10000}
            return None

        trading_handlers.state_manager.get_state = AsyncMock(side_effect=mock_get_state)

        success = await trading_handlers.recover_trading_engine(recovery_operation)

        assert success is True
        # Only 2 pending orders should be cancelled (order:1 and order:3)
        assert "Cancelled 2 pending orders" in recovery_operation.actions_taken[1]
