"""Tests for orchestration/execution/guards.py - GuardManager runtime safety checks."""

import time
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.features.live_trade.guard_errors import (
    RiskGuardActionError,
    RiskGuardComputationError,
    RiskGuardDataCorrupt,
    RiskGuardDataUnavailable,
    RiskGuardTelemetryError,
)
from gpt_trader.orchestration.execution.guards import GuardManager, RuntimeGuardState

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_broker():
    broker = MagicMock()
    broker.list_balances.return_value = []
    broker.list_positions.return_value = []
    broker.cancel_order.return_value = True
    return broker


@pytest.fixture
def mock_risk_manager():
    rm = MagicMock()
    rm.track_daily_pnl.return_value = False
    rm.last_mark_update = {}
    rm.config = MagicMock()
    rm.config.volatility_window_periods = 20
    return rm


@pytest.fixture
def mock_equity_calculator():
    return MagicMock(return_value=(Decimal("1000"), [], Decimal("1000")))


@pytest.fixture
def guard_manager(mock_broker, mock_risk_manager, mock_equity_calculator):
    open_orders = ["order1", "order2"]
    invalidate_cache = MagicMock()
    return GuardManager(
        broker=mock_broker,
        risk_manager=mock_risk_manager,
        equity_calculator=mock_equity_calculator,
        open_orders=open_orders,
        invalidate_cache_callback=invalidate_cache,
    )


@pytest.fixture
def mock_position():
    """Create a mock position object."""
    pos = MagicMock()
    pos.symbol = "BTC-PERP"
    pos.entry_price = "50000"
    pos.mark_price = "51000"
    pos.quantity = "0.1"
    pos.side = "long"
    return pos


@pytest.fixture
def sample_guard_state(mock_position):
    """Create a sample RuntimeGuardState for testing."""
    return RuntimeGuardState(
        timestamp=time.time(),
        balances=[],
        equity=Decimal("10000"),
        positions=[mock_position],
        positions_pnl={
            "BTC-PERP": {"realized_pnl": Decimal("0"), "unrealized_pnl": Decimal("100")}
        },
        positions_dict={
            "BTC-PERP": {
                "quantity": Decimal("0.1"),
                "mark": Decimal("51000"),
                "entry": Decimal("50000"),
            }
        },
        guard_events=[],
    )


# =============================================================================
# Tests for invalidate_cache
# =============================================================================


def test_invalidate_cache_clears_state(guard_manager):
    """Test that invalidate_cache clears cached state."""
    guard_manager._runtime_guard_state = MagicMock()
    guard_manager._runtime_guard_dirty = False

    guard_manager.invalidate_cache()

    assert guard_manager._runtime_guard_state is None
    assert guard_manager._runtime_guard_dirty is True
    guard_manager._invalidate_cache_callback.assert_called_once()


def test_invalidate_cache_handles_no_callback(
    mock_broker, mock_risk_manager, mock_equity_calculator
):
    """Test invalidate_cache works when callback is None."""
    gm = GuardManager(
        broker=mock_broker,
        risk_manager=mock_risk_manager,
        equity_calculator=mock_equity_calculator,
        open_orders=[],
        invalidate_cache_callback=None,
    )
    gm._runtime_guard_state = MagicMock()

    # Should not raise
    gm.invalidate_cache()
    assert gm._runtime_guard_state is None


# =============================================================================
# Tests for should_run_full_guard
# =============================================================================


def test_should_run_full_guard_when_dirty(guard_manager):
    """Test returns True when dirty flag is set."""
    guard_manager._runtime_guard_dirty = True
    assert guard_manager.should_run_full_guard(time.time()) is True


def test_should_run_full_guard_when_no_state(guard_manager):
    """Test returns True when no cached state exists."""
    guard_manager._runtime_guard_dirty = False
    guard_manager._runtime_guard_state = None
    assert guard_manager.should_run_full_guard(time.time()) is True


def test_should_run_full_guard_when_interval_elapsed(guard_manager):
    """Test returns True when full interval has elapsed."""
    guard_manager._runtime_guard_dirty = False
    guard_manager._runtime_guard_state = MagicMock()
    guard_manager._runtime_guard_last_full_ts = time.time() - 120  # 2 minutes ago
    guard_manager._runtime_guard_full_interval = 60  # 1 minute interval

    assert guard_manager.should_run_full_guard(time.time()) is True


def test_should_run_full_guard_returns_false_within_interval(guard_manager):
    """Test returns False when within interval and not dirty."""
    guard_manager._runtime_guard_dirty = False
    guard_manager._runtime_guard_state = MagicMock()
    guard_manager._runtime_guard_last_full_ts = time.time() - 30  # 30 seconds ago
    guard_manager._runtime_guard_full_interval = 60  # 1 minute interval

    assert guard_manager.should_run_full_guard(time.time()) is False


# =============================================================================
# Tests for collect_runtime_guard_state
# =============================================================================


def test_collect_runtime_guard_state_basic(guard_manager, mock_broker):
    """Test basic state collection."""
    mock_balance = MagicMock()
    mock_balance.available = Decimal("5000")
    mock_broker.list_balances.return_value = [mock_balance]
    mock_broker.list_positions.return_value = []

    state = guard_manager.collect_runtime_guard_state()

    assert isinstance(state, RuntimeGuardState)
    assert state.equity == Decimal("1000")  # From mock equity calculator
    assert state.balances == [mock_balance]
    assert state.positions == []
    assert state.positions_pnl == {}
    assert state.positions_dict == {}


def test_collect_runtime_guard_state_with_positions(guard_manager, mock_broker, mock_position):
    """Test state collection with positions."""
    mock_broker.list_balances.return_value = []
    mock_broker.list_positions.return_value = [mock_position]

    state = guard_manager.collect_runtime_guard_state()

    assert len(state.positions) == 1
    assert "BTC-PERP" in state.positions_pnl
    assert "BTC-PERP" in state.positions_dict


def test_collect_runtime_guard_state_uses_broker_pnl(guard_manager, mock_broker, mock_position):
    """Test state collection uses broker's get_position_pnl when available."""
    mock_broker.list_balances.return_value = []
    mock_broker.list_positions.return_value = [mock_position]
    mock_broker.get_position_pnl.return_value = {
        "realized_pnl": "500",
        "unrealized_pnl": "200",
    }

    state = guard_manager.collect_runtime_guard_state()

    assert state.positions_pnl["BTC-PERP"]["realized_pnl"] == Decimal("500")
    assert state.positions_pnl["BTC-PERP"]["unrealized_pnl"] == Decimal("200")


def test_collect_runtime_guard_state_fallback_equity(guard_manager, mock_broker):
    """Test equity fallback when calculator returns zero."""
    mock_balance = MagicMock()
    mock_balance.available = Decimal("5000")
    mock_broker.list_balances.return_value = [mock_balance]
    guard_manager._calculate_equity = MagicMock(return_value=(Decimal("0"), [], Decimal("0")))

    state = guard_manager.collect_runtime_guard_state()

    assert state.equity == Decimal("5000")  # Fallback sum of available balances


def test_collect_runtime_guard_state_handles_position_errors(guard_manager, mock_broker):
    """Test state collection handles malformed position data gracefully."""
    bad_position = MagicMock()
    bad_position.symbol = "BAD-PERP"
    # These will cause exceptions when converted
    bad_position.entry_price = "invalid"
    bad_position.mark_price = "invalid"
    del bad_position.quantity

    mock_broker.list_balances.return_value = []
    mock_broker.list_positions.return_value = [bad_position]

    state = guard_manager.collect_runtime_guard_state()

    # Should still create state, with zero/empty values for bad position
    assert isinstance(state, RuntimeGuardState)


# =============================================================================
# Tests for run_guard_step
# =============================================================================


def test_run_guard_step_success(guard_manager):
    """Test successful guard step execution."""
    func = MagicMock()

    with patch("gpt_trader.orchestration.execution.guards.record_guard_success") as mock_success:
        guard_manager.run_guard_step("test_guard", func)

    func.assert_called_once()
    mock_success.assert_called_with("test_guard")


def test_run_guard_step_recoverable_error(guard_manager):
    """Test guard step with recoverable error continues."""
    error = RiskGuardTelemetryError(
        guard_name="test_guard",
        message="Recoverable error",
        details={},
    )
    func = MagicMock(side_effect=error)

    with patch("gpt_trader.orchestration.execution.guards.record_guard_failure") as mock_failure:
        # Should not raise because error is recoverable
        guard_manager.run_guard_step("test_guard", func)

    mock_failure.assert_called_once()


def test_run_guard_step_unrecoverable_error(guard_manager):
    """Test guard step with unrecoverable error raises."""
    error = RiskGuardActionError(
        guard_name="test_guard",
        message="Fatal error",
        details={},
    )
    func = MagicMock(side_effect=error)

    with patch("gpt_trader.orchestration.execution.guards.record_guard_failure"):
        with pytest.raises(RiskGuardActionError):
            guard_manager.run_guard_step("test_guard", func)


def test_run_guard_step_unexpected_error(guard_manager):
    """Test guard step wraps unexpected errors."""
    func = MagicMock(side_effect=ValueError("Unexpected"))

    with patch("gpt_trader.orchestration.execution.guards.record_guard_failure") as mock_failure:
        with pytest.raises(RiskGuardComputationError):
            guard_manager.run_guard_step("test_guard", func)

    # Should have recorded the wrapped error
    assert mock_failure.called


# =============================================================================
# Tests for log_guard_telemetry
# =============================================================================


def test_log_guard_telemetry_success(guard_manager, sample_guard_state):
    """Test successful telemetry logging."""
    with patch("gpt_trader.orchestration.execution.guards._get_plog") as mock_get_plog:
        mock_plog = MagicMock()
        mock_get_plog.return_value = mock_plog

        guard_manager.log_guard_telemetry(sample_guard_state)

        mock_plog.log_pnl.assert_called_once()


def test_log_guard_telemetry_failure_raises(guard_manager, sample_guard_state):
    """Test telemetry failure raises RiskGuardTelemetryError."""
    with patch("gpt_trader.orchestration.execution.guards._get_plog") as mock_get_plog:
        mock_plog = MagicMock()
        mock_plog.log_pnl.side_effect = Exception("Telemetry failed")
        mock_get_plog.return_value = mock_plog

        with pytest.raises(RiskGuardTelemetryError) as exc_info:
            guard_manager.log_guard_telemetry(sample_guard_state)

        assert "BTC-PERP" in str(exc_info.value.details)


# =============================================================================
# Tests for guard_daily_loss
# =============================================================================


def test_guard_daily_loss_not_triggered(guard_manager, sample_guard_state, mock_risk_manager):
    """Test daily loss guard when not triggered."""
    mock_risk_manager.track_daily_pnl.return_value = False

    guard_manager.guard_daily_loss(sample_guard_state)

    mock_risk_manager.track_daily_pnl.assert_called_once()


def test_guard_daily_loss_triggered_cancels_orders(
    guard_manager, sample_guard_state, mock_risk_manager
):
    """Test daily loss guard cancels orders when triggered."""
    mock_risk_manager.track_daily_pnl.return_value = True

    with patch.object(guard_manager, "cancel_all_orders") as mock_cancel:
        guard_manager.guard_daily_loss(sample_guard_state)

    mock_cancel.assert_called_once()
    guard_manager._invalidate_cache_callback.assert_called()


def test_guard_daily_loss_cancel_failure(guard_manager, sample_guard_state, mock_risk_manager):
    """Test daily loss guard raises on cancel failure."""
    mock_risk_manager.track_daily_pnl.return_value = True

    with patch.object(guard_manager, "cancel_all_orders", side_effect=Exception("Cancel failed")):
        with pytest.raises(RiskGuardActionError):
            guard_manager.guard_daily_loss(sample_guard_state)


# =============================================================================
# Tests for guard_liquidation_buffers
# =============================================================================


def test_guard_liquidation_buffers_basic(guard_manager, sample_guard_state, mock_risk_manager):
    """Test basic liquidation buffer check."""
    guard_manager.guard_liquidation_buffers(sample_guard_state, incremental=True)

    mock_risk_manager.check_liquidation_buffer.assert_called_once()


def test_guard_liquidation_buffers_full_with_risk_info(
    guard_manager, sample_guard_state, mock_broker, mock_risk_manager
):
    """Test full liquidation buffer check fetches risk info."""
    mock_broker.get_position_risk.return_value = {"liquidation_price": "45000"}

    guard_manager.guard_liquidation_buffers(sample_guard_state, incremental=False)

    mock_broker.get_position_risk.assert_called_once()
    mock_risk_manager.check_liquidation_buffer.assert_called_once()


def test_guard_liquidation_buffers_corrupt_data(guard_manager, mock_risk_manager):
    """Test liquidation buffer raises on corrupt position data."""
    bad_position = MagicMock()
    bad_position.symbol = "BAD"
    bad_position.mark_price = "invalid"
    del bad_position.quantity

    state = RuntimeGuardState(
        timestamp=time.time(),
        balances=[],
        equity=Decimal("10000"),
        positions=[bad_position],
        positions_pnl={},
        positions_dict={},
        guard_events=[],
    )

    with pytest.raises(RiskGuardDataCorrupt):
        guard_manager.guard_liquidation_buffers(state, incremental=True)


def test_guard_liquidation_buffers_risk_fetch_failure(
    guard_manager, sample_guard_state, mock_broker
):
    """Test liquidation buffer raises on risk fetch failure."""
    mock_broker.get_position_risk.side_effect = Exception("API error")

    with pytest.raises(RiskGuardDataUnavailable):
        guard_manager.guard_liquidation_buffers(sample_guard_state, incremental=False)


# =============================================================================
# Tests for guard_mark_staleness
# =============================================================================


def test_guard_mark_staleness_no_cache(guard_manager, sample_guard_state, mock_broker):
    """Test mark staleness guard skips when no mark cache."""
    del mock_broker._mark_cache

    # Should not raise
    guard_manager.guard_mark_staleness(sample_guard_state)


def test_guard_mark_staleness_with_cache(
    guard_manager, sample_guard_state, mock_broker, mock_risk_manager
):
    """Test mark staleness guard checks cached marks."""
    mock_broker._mark_cache = MagicMock()
    mock_broker._mark_cache.get_mark.return_value = None
    mock_risk_manager.last_mark_update = {"BTC-PERP": time.time()}

    guard_manager.guard_mark_staleness(sample_guard_state)

    mock_risk_manager.check_mark_staleness.assert_called_with("BTC-PERP")


def test_guard_mark_staleness_fetch_failure(
    guard_manager, sample_guard_state, mock_broker, mock_risk_manager
):
    """Test mark staleness raises on fetch failures."""
    mock_broker._mark_cache = MagicMock()
    mock_broker._mark_cache.get_mark.side_effect = Exception("Cache error")
    mock_risk_manager.last_mark_update = {"BTC-PERP": time.time()}

    with pytest.raises(RiskGuardDataUnavailable):
        guard_manager.guard_mark_staleness(sample_guard_state)


# =============================================================================
# Tests for guard_risk_metrics
# =============================================================================


def test_guard_risk_metrics_success(guard_manager, sample_guard_state, mock_risk_manager):
    """Test successful risk metrics append."""
    guard_manager.guard_risk_metrics(sample_guard_state)

    mock_risk_manager.append_risk_metrics.assert_called_once()


def test_guard_risk_metrics_failure(guard_manager, sample_guard_state, mock_risk_manager):
    """Test risk metrics raises on failure."""
    mock_risk_manager.append_risk_metrics.side_effect = Exception("Metrics error")

    with pytest.raises(RiskGuardTelemetryError):
        guard_manager.guard_risk_metrics(sample_guard_state)


def test_guard_risk_metrics_propagates_guard_error(
    guard_manager, sample_guard_state, mock_risk_manager
):
    """Test risk metrics propagates RiskGuardError."""
    error = RiskGuardComputationError(guard_name="risk_metrics", message="Test", details={})
    mock_risk_manager.append_risk_metrics.side_effect = error

    with pytest.raises(RiskGuardComputationError):
        guard_manager.guard_risk_metrics(sample_guard_state)


# =============================================================================
# Tests for guard_volatility
# =============================================================================


def test_guard_volatility_skips_short_window(guard_manager, sample_guard_state, mock_risk_manager):
    """Test volatility guard skips with short window."""
    mock_risk_manager.config.volatility_window_periods = 3  # Too short

    guard_manager.guard_volatility(sample_guard_state)

    # Should not call get_candles
    guard_manager.broker.get_candles.assert_not_called()


def test_guard_volatility_skips_no_symbols(guard_manager, mock_risk_manager):
    """Test volatility guard skips with no symbols."""
    state = RuntimeGuardState(
        timestamp=time.time(),
        balances=[],
        equity=Decimal("10000"),
        positions=[],
        positions_pnl={},
        positions_dict={},
        guard_events=[],
    )
    mock_risk_manager.last_mark_update = {}

    guard_manager.guard_volatility(state)

    guard_manager.broker.get_candles.assert_not_called()


def test_guard_volatility_checks_symbols(
    guard_manager, sample_guard_state, mock_broker, mock_risk_manager
):
    """Test volatility guard checks all relevant symbols."""
    mock_risk_manager.last_mark_update = {"BTC-PERP": time.time()}
    mock_risk_manager.config.volatility_window_periods = 20

    mock_candle = MagicMock()
    mock_candle.close = Decimal("50000")
    mock_broker.get_candles.return_value = [mock_candle] * 20

    mock_risk_manager.check_volatility_circuit_breaker.return_value = MagicMock(triggered=False)

    guard_manager.guard_volatility(sample_guard_state)

    mock_broker.get_candles.assert_called()
    mock_risk_manager.check_volatility_circuit_breaker.assert_called()


def test_guard_volatility_records_triggered_events(
    guard_manager, sample_guard_state, mock_broker, mock_risk_manager
):
    """Test volatility guard records triggered events."""
    mock_risk_manager.last_mark_update = {"BTC-PERP": time.time()}
    mock_risk_manager.config.volatility_window_periods = 20

    mock_candle = MagicMock()
    mock_candle.close = Decimal("50000")
    mock_broker.get_candles.return_value = [mock_candle] * 20

    outcome = MagicMock()
    outcome.triggered = True
    outcome.to_payload.return_value = {"type": "volatility_breach"}
    mock_risk_manager.check_volatility_circuit_breaker.return_value = outcome

    guard_manager.guard_volatility(sample_guard_state)

    assert len(sample_guard_state.guard_events) == 1
    assert sample_guard_state.guard_events[0]["type"] == "volatility_breach"


def test_guard_volatility_fetch_failure(
    guard_manager, sample_guard_state, mock_broker, mock_risk_manager
):
    """Test volatility raises on candle fetch failures."""
    mock_risk_manager.last_mark_update = {"BTC-PERP": time.time()}
    mock_risk_manager.config.volatility_window_periods = 20
    mock_broker.get_candles.side_effect = Exception("API error")

    with pytest.raises(RiskGuardDataUnavailable):
        guard_manager.guard_volatility(sample_guard_state)


# =============================================================================
# Tests for run_guards_for_state
# =============================================================================


def test_run_guards_for_state_calls_all_guards(guard_manager, sample_guard_state):
    """Test run_guards_for_state calls all guard steps."""
    with patch.object(guard_manager, "run_guard_step") as mock_step:
        guard_manager.run_guards_for_state(sample_guard_state, incremental=False)

    # Should call all 6 guards
    assert mock_step.call_count == 6
    guard_names = [call[0][0] for call in mock_step.call_args_list]
    assert "pnl_telemetry" in guard_names
    assert "daily_loss" in guard_names
    assert "liquidation_buffer" in guard_names
    assert "mark_staleness" in guard_names
    assert "risk_metrics" in guard_names
    assert "volatility_circuit_breaker" in guard_names


# =============================================================================
# Tests for run_runtime_guards
# =============================================================================


def test_run_runtime_guards_first_run(guard_manager):
    """Test first runtime guards run is full."""
    with (
        patch.object(guard_manager, "collect_runtime_guard_state") as mock_collect,
        patch.object(guard_manager, "run_guards_for_state") as mock_run_guards,
    ):
        mock_state = MagicMock()
        mock_collect.return_value = mock_state

        state = guard_manager.run_runtime_guards()

        assert state == mock_state
        mock_collect.assert_called_once()
        mock_run_guards.assert_called_with(mock_state, False)  # incremental=False


def test_run_runtime_guards_incremental(guard_manager):
    """Test incremental runtime guards reuses cached state."""
    mock_state = MagicMock()
    guard_manager._runtime_guard_state = mock_state
    guard_manager._runtime_guard_dirty = False
    guard_manager._runtime_guard_last_full_ts = time.time()

    with (
        patch.object(guard_manager, "collect_runtime_guard_state") as mock_collect,
        patch.object(guard_manager, "run_guards_for_state") as mock_run_guards,
    ):
        state = guard_manager.run_runtime_guards()

        assert state == mock_state
        mock_collect.assert_not_called()
        mock_run_guards.assert_called_with(mock_state, True)  # incremental=True


def test_run_runtime_guards_force_full(guard_manager):
    """Test force_full overrides incremental."""
    mock_state = MagicMock()
    guard_manager._runtime_guard_state = mock_state
    guard_manager._runtime_guard_dirty = False
    guard_manager._runtime_guard_last_full_ts = time.time()

    with (
        patch.object(guard_manager, "collect_runtime_guard_state") as mock_collect,
        patch.object(guard_manager, "run_guards_for_state") as mock_run_guards,
    ):
        new_state = MagicMock()
        mock_collect.return_value = new_state

        state = guard_manager.run_runtime_guards(force_full=True)

        assert state == new_state
        mock_collect.assert_called_once()
        mock_run_guards.assert_called_with(new_state, False)  # incremental=False


# =============================================================================
# Tests for cancel_all_orders
# =============================================================================


def test_cancel_all_orders_success(guard_manager, mock_broker):
    """Test successful cancellation of all orders."""
    mock_broker.cancel_order.return_value = True

    cancelled_count = guard_manager.cancel_all_orders()

    assert cancelled_count == 2
    assert mock_broker.cancel_order.call_count == 2
    assert len(guard_manager.open_orders) == 0
    guard_manager._invalidate_cache_callback.assert_called()


def test_cancel_all_orders_partial_failure(guard_manager, mock_broker):
    """Test partial failure during order cancellation."""
    mock_broker.cancel_order.side_effect = [Exception("Failed"), True]

    cancelled_count = guard_manager.cancel_all_orders()

    assert cancelled_count == 1
    assert "order1" in guard_manager.open_orders
    assert "order2" not in guard_manager.open_orders


def test_cancel_all_orders_none_cancelled(guard_manager, mock_broker):
    """Test when no orders are cancelled."""
    mock_broker.cancel_order.return_value = False

    cancelled_count = guard_manager.cancel_all_orders()

    assert cancelled_count == 0
    guard_manager._invalidate_cache_callback.assert_not_called()


def test_cancel_all_orders_empty_list(mock_broker, mock_risk_manager, mock_equity_calculator):
    """Test cancellation with empty order list."""
    gm = GuardManager(
        broker=mock_broker,
        risk_manager=mock_risk_manager,
        equity_calculator=mock_equity_calculator,
        open_orders=[],
        invalidate_cache_callback=MagicMock(),
    )

    cancelled_count = gm.cancel_all_orders()

    assert cancelled_count == 0
    mock_broker.cancel_order.assert_not_called()


# =============================================================================
# Tests for safe_run_runtime_guards
# =============================================================================


def test_safe_run_runtime_guards_success(guard_manager):
    """Test successful safe run."""
    with patch.object(guard_manager, "run_runtime_guards") as mock_run:
        guard_manager.safe_run_runtime_guards()
        mock_run.assert_called_once_with(force_full=False)


def test_safe_run_runtime_guards_force_full(guard_manager):
    """Test safe run with force_full."""
    with patch.object(guard_manager, "run_runtime_guards") as mock_run:
        guard_manager.safe_run_runtime_guards(force_full=True)
        mock_run.assert_called_once_with(force_full=True)


def test_safe_run_runtime_guards_recoverable_error(guard_manager, mock_risk_manager):
    """Test safe run handles recoverable error."""
    error = RiskGuardTelemetryError(guard_name="test", message="Recoverable", details={})

    with patch.object(guard_manager, "run_runtime_guards", side_effect=error):
        guard_manager.safe_run_runtime_guards()

    # Should NOT set reduce-only mode for recoverable errors
    mock_risk_manager.set_reduce_only_mode.assert_not_called()


def test_safe_run_runtime_guards_unrecoverable_error(guard_manager, mock_risk_manager):
    """Test safe run handles unrecoverable error."""
    error = RiskGuardActionError(guard_name="test", message="Fatal", details={})

    with patch.object(guard_manager, "run_runtime_guards", side_effect=error):
        guard_manager.safe_run_runtime_guards()

    mock_risk_manager.set_reduce_only_mode.assert_called_with(True, reason="guard_failure")
    guard_manager._invalidate_cache_callback.assert_called()


def test_safe_run_runtime_guards_unexpected_error(guard_manager):
    """Test safe run handles unexpected exceptions."""
    with patch.object(guard_manager, "run_runtime_guards", side_effect=ValueError("Unexpected")):
        # Should not raise
        guard_manager.safe_run_runtime_guards()


def test_safe_run_runtime_guards_reduce_only_failure(guard_manager, mock_risk_manager):
    """Test safe run handles reduce-only mode failure."""
    error = RiskGuardActionError(guard_name="test", message="Fatal", details={})
    mock_risk_manager.set_reduce_only_mode.side_effect = Exception("Failed")

    with patch.object(guard_manager, "run_runtime_guards", side_effect=error):
        # Should not raise even if set_reduce_only_mode fails
        guard_manager.safe_run_runtime_guards()

    guard_manager._invalidate_cache_callback.assert_called()
