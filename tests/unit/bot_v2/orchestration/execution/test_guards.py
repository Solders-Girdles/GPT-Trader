"""Tests for execution guards"""

import pytest
import time
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch
from bot_v2.features.brokerages.core.interfaces import Balance, Position
from bot_v2.features.live_trade.guard_errors import (
    RiskGuardActionError,
    RiskGuardDataCorrupt,
    RiskGuardDataUnavailable,
)
from bot_v2.orchestration.execution.guards import GuardManager, RuntimeGuardState


@pytest.fixture
def mock_broker():
    """Mock broker"""
    broker = Mock()
    broker.list_balances = Mock()
    broker.list_positions = Mock()
    return broker


@pytest.fixture
def mock_risk_manager():
    """Mock risk manager"""
    manager = Mock()
    manager.config = Mock()
    manager.track_daily_pnl = Mock(return_value=False)
    manager.check_liquidation_buffer = Mock()
    manager.check_mark_staleness = Mock(return_value=False)
    manager.append_risk_metrics = Mock()
    manager.check_correlation_risk = Mock()
    manager.check_volatility_circuit_breaker = Mock(return_value=Mock(triggered=False))
    manager.last_mark_update = {}
    return manager


@pytest.fixture
def equity_calculator():
    """Mock equity calculator"""
    return Mock(return_value=(Decimal("10000"), Decimal("0"), Decimal("0")))


@pytest.fixture
def cancel_orders_callback():
    """Mock cancel orders callback"""
    return Mock(return_value=0)


@pytest.fixture
def invalidate_cache_callback():
    """Mock cache invalidation callback"""
    return Mock()


@pytest.fixture
def guard_manager(
    mock_broker,
    mock_risk_manager,
    equity_calculator,
    cancel_orders_callback,
    invalidate_cache_callback,
):
    """Create GuardManager instance"""
    return GuardManager(
        broker=mock_broker,
        risk_manager=mock_risk_manager,
        equity_calculator=equity_calculator,
        cancel_orders_callback=cancel_orders_callback,
        invalidate_cache_callback=invalidate_cache_callback,
    )


class TestGuardManager:
    """Test suite for GuardManager"""

    def test_initialization(self, guard_manager):
        """Test guard manager initialization"""
        assert guard_manager.broker is not None
        assert guard_manager.risk_manager is not None
        assert guard_manager._runtime_guard_state is None
        assert guard_manager._runtime_guard_dirty is True

    def test_invalidate_cache(self, guard_manager):
        """Test cache invalidation"""
        guard_manager._runtime_guard_state = Mock()
        guard_manager._runtime_guard_dirty = False

        guard_manager.invalidate_cache()

        assert guard_manager._runtime_guard_state is None
        assert guard_manager._runtime_guard_dirty is True

    def test_should_run_full_guard_when_dirty(self, guard_manager):
        """Test full guard run when cache is dirty"""
        guard_manager._runtime_guard_dirty = True

        assert guard_manager.should_run_full_guard(time.time()) is True

    def test_should_run_full_guard_when_no_state(self, guard_manager):
        """Test full guard run when no cached state"""
        guard_manager._runtime_guard_dirty = False
        guard_manager._runtime_guard_state = None

        assert guard_manager.should_run_full_guard(time.time()) is True

    def test_should_run_full_guard_when_interval_exceeded(self, guard_manager):
        """Test full guard run when interval exceeded"""
        guard_manager._runtime_guard_dirty = False
        guard_manager._runtime_guard_state = Mock()
        guard_manager._runtime_guard_last_full_ts = time.time() - 70

        assert guard_manager.should_run_full_guard(time.time()) is True

    def test_should_not_run_full_guard_when_cached(self, guard_manager):
        """Test no full guard run when cache is fresh"""
        guard_manager._runtime_guard_dirty = False
        guard_manager._runtime_guard_state = Mock()
        guard_manager._runtime_guard_last_full_ts = time.time()

        assert guard_manager.should_run_full_guard(time.time()) is False

    def test_collect_runtime_guard_state(self, guard_manager, mock_broker):
        """Test guard state collection"""
        balance = Mock(spec=Balance)
        balance.asset = "USD"
        balance.available = Decimal("10000")
        mock_broker.list_balances.return_value = [balance]
        mock_broker.list_positions.return_value = []

        state = guard_manager.collect_runtime_guard_state()

        assert isinstance(state, RuntimeGuardState)
        assert state.equity == Decimal("10000")
        assert len(state.balances) == 1
        assert len(state.positions) == 0

    def test_collect_runtime_guard_state_with_positions(
        self, guard_manager, mock_broker
    ):
        """Test state collection with positions"""
        balance = Mock(spec=Balance)
        balance.asset = "USD"
        balance.available = Decimal("10000")

        position = Mock(spec=Position)
        position.symbol = "BTC-USD"
        position.quantity = Decimal("0.5")
        position.entry_price = Decimal("48000")
        position.mark_price = Decimal("50000")
        position.side = "long"

        mock_broker.list_balances.return_value = [balance]
        mock_broker.list_positions.return_value = [position]

        state = guard_manager.collect_runtime_guard_state()

        assert "BTC-USD" in state.positions_pnl
        assert "unrealized_pnl" in state.positions_pnl["BTC-USD"]
        assert "BTC-USD" in state.positions_dict

    def test_run_guard_step_success(self, guard_manager):
        """Test successful guard step execution"""
        func = Mock()

        guard_manager.run_guard_step("test_guard", func)

        func.assert_called_once()

    def test_run_guard_step_with_recoverable_error(self, guard_manager):
        """Test guard step with recoverable error"""
        from bot_v2.features.live_trade.guard_errors import RiskGuardTelemetryError

        error = RiskGuardTelemetryError(
            guard="test", message="Telemetry failed", details={}
        )
        func = Mock(side_effect=error)

        # Should not raise for recoverable errors
        guard_manager.run_guard_step("test_guard", func)

    def test_run_guard_step_with_fatal_error(self, guard_manager):
        """Test guard step with fatal error"""
        from bot_v2.features.live_trade.guard_errors import RiskGuardDataCorrupt

        error = RiskGuardDataCorrupt(
            guard="test", message="Data corrupt", details={}
        )
        error.recoverable = False
        func = Mock(side_effect=error)

        with pytest.raises(RiskGuardDataCorrupt):
            guard_manager.run_guard_step("test_guard", func)

    def test_guard_daily_loss_not_triggered(self, guard_manager, mock_risk_manager):
        """Test daily loss guard when not triggered"""
        state = RuntimeGuardState(
            timestamp=time.time(),
            balances=[],
            equity=Decimal("10000"),
            positions=[],
            positions_pnl={},
            positions_dict={},
        )

        mock_risk_manager.track_daily_pnl.return_value = False

        guard_manager.guard_daily_loss(state)

        # Should not cancel orders
        guard_manager._cancel_all_orders.assert_not_called()

    def test_guard_daily_loss_triggered(
        self, guard_manager, mock_risk_manager, cancel_orders_callback
    ):
        """Test daily loss guard when triggered"""
        state = RuntimeGuardState(
            timestamp=time.time(),
            balances=[],
            equity=Decimal("8000"),
            positions=[],
            positions_pnl={},
            positions_dict={},
        )

        mock_risk_manager.track_daily_pnl.return_value = True

        guard_manager.guard_daily_loss(state)

        # Should cancel orders and invalidate cache
        cancel_orders_callback.assert_called_once()
        guard_manager._invalidate_cache.assert_called_once()

    def test_guard_daily_loss_cancel_fails(self, guard_manager, mock_risk_manager):
        """Test daily loss guard when cancel fails"""
        state = RuntimeGuardState(
            timestamp=time.time(),
            balances=[],
            equity=Decimal("8000"),
            positions=[],
            positions_pnl={},
            positions_dict={},
        )

        mock_risk_manager.track_daily_pnl.return_value = True
        guard_manager._cancel_all_orders.side_effect = Exception("Cancel failed")

        with pytest.raises(RiskGuardActionError):
            guard_manager.guard_daily_loss(state)

    def test_guard_liquidation_buffers(self, guard_manager, mock_broker):
        """Test liquidation buffer guard"""
        position = Mock()
        position.symbol = "BTC-USD"
        position.quantity = Decimal("0.5")
        position.mark_price = Decimal("50000")

        state = RuntimeGuardState(
            timestamp=time.time(),
            balances=[],
            equity=Decimal("10000"),
            positions=[position],
            positions_pnl={},
            positions_dict={},
        )

        guard_manager.guard_liquidation_buffers(state, incremental=False)

        guard_manager.risk_manager.check_liquidation_buffer.assert_called_once()

    def test_guard_liquidation_buffers_corrupt_data(self, guard_manager):
        """Test liquidation buffer guard with corrupt data"""
        position = Mock()
        position.symbol = "BTC-USD"
        # Missing required attributes to trigger error

        state = RuntimeGuardState(
            timestamp=time.time(),
            balances=[],
            equity=Decimal("10000"),
            positions=[position],
            positions_pnl={},
            positions_dict={},
        )

        with pytest.raises(RiskGuardDataCorrupt):
            guard_manager.guard_liquidation_buffers(state, incremental=False)

    def test_guard_mark_staleness(self, guard_manager, mock_broker, mock_risk_manager):
        """Test mark staleness guard"""
        mock_broker._mark_cache = Mock()
        mock_broker._mark_cache.get_mark = Mock(return_value=Decimal("50000"))
        mock_risk_manager.last_mark_update = {"BTC-USD": time.time()}

        state = RuntimeGuardState(
            timestamp=time.time(),
            balances=[],
            equity=Decimal("10000"),
            positions=[],
            positions_pnl={},
            positions_dict={},
        )

        guard_manager.guard_mark_staleness(state)

        # Should check staleness
        mock_risk_manager.check_mark_staleness.assert_not_called()

    def test_guard_mark_staleness_no_mark_cache(self, guard_manager):
        """Test mark staleness guard without mark cache"""
        state = RuntimeGuardState(
            timestamp=time.time(),
            balances=[],
            equity=Decimal("10000"),
            positions=[],
            positions_pnl={},
            positions_dict={},
        )

        # Should return early without error
        guard_manager.guard_mark_staleness(state)

    def test_guard_risk_metrics(self, guard_manager, mock_risk_manager):
        """Test risk metrics guard"""
        state = RuntimeGuardState(
            timestamp=time.time(),
            balances=[],
            equity=Decimal("10000"),
            positions=[],
            positions_pnl={},
            positions_dict={"BTC-USD": {"quantity": Decimal("0.5")}},
        )

        guard_manager.guard_risk_metrics(state)

        mock_risk_manager.append_risk_metrics.assert_called_once_with(
            state.equity, state.positions_dict
        )

    def test_guard_correlation(self, guard_manager, mock_risk_manager):
        """Test correlation risk guard"""
        state = RuntimeGuardState(
            timestamp=time.time(),
            balances=[],
            equity=Decimal("10000"),
            positions=[],
            positions_pnl={},
            positions_dict={
                "BTC-USD": {"quantity": Decimal("0.5")},
                "ETH-USD": {"quantity": Decimal("5.0")},
            },
        )

        guard_manager.guard_correlation(state)

        mock_risk_manager.check_correlation_risk.assert_called_once()

    def test_run_runtime_guards_full(self, guard_manager, mock_broker):
        """Test full runtime guards execution"""
        balance = Mock(spec=Balance)
        balance.asset = "USD"
        balance.available = Decimal("10000")
        mock_broker.list_balances.return_value = [balance]
        mock_broker.list_positions.return_value = []

        state = guard_manager.run_runtime_guards(force_full=True)

        assert isinstance(state, RuntimeGuardState)
        assert guard_manager._runtime_guard_state is not None
        assert guard_manager._runtime_guard_dirty is False

    def test_run_runtime_guards_incremental(self, guard_manager):
        """Test incremental runtime guards execution"""
        # Set up cached state
        cached_state = RuntimeGuardState(
            timestamp=time.time(),
            balances=[],
            equity=Decimal("10000"),
            positions=[],
            positions_pnl={},
            positions_dict={},
        )
        guard_manager._runtime_guard_state = cached_state
        guard_manager._runtime_guard_dirty = False
        guard_manager._runtime_guard_last_full_ts = time.time()

        state = guard_manager.run_runtime_guards(force_full=False)

        # Should reuse cached state
        assert state == cached_state

    @patch('bot_v2.orchestration.execution.guards.get_logger')
    def test_log_guard_telemetry(self, mock_get_logger, guard_manager):
        """Test P&L telemetry logging"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        state = RuntimeGuardState(
            timestamp=time.time(),
            balances=[],
            equity=Decimal("10000"),
            positions=[],
            positions_pnl={
                "BTC-USD": {
                    "realized_pnl": Decimal("100"),
                    "unrealized_pnl": Decimal("200"),
                }
            },
            positions_dict={},
        )

        guard_manager.log_guard_telemetry(state)

        mock_logger.log_pnl.assert_called_once()

    def test_guard_volatility(self, guard_manager, mock_broker, mock_risk_manager):
        """Test volatility circuit breaker guard"""
        mock_risk_manager.config.volatility_window_periods = 20
        mock_risk_manager.last_mark_update = {"BTC-USD": time.time()}

        mock_candle = Mock()
        mock_candle.close = 50000.0
        mock_broker.get_candles = Mock(return_value=[mock_candle] * 20)

        state = RuntimeGuardState(
            timestamp=time.time(),
            balances=[],
            equity=Decimal("10000"),
            positions=[],
            positions_pnl={},
            positions_dict={},
        )

        guard_manager.guard_volatility(state)

        mock_risk_manager.check_volatility_circuit_breaker.assert_called()