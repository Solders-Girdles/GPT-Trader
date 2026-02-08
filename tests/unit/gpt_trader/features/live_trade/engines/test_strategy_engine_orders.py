"""Tests for TradingEngine order flow, guards, and quantity calculations."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from strategy_engine_chaos_helpers import make_position

import gpt_trader.security.validate as security_validate_module
from gpt_trader.core import Balance, OrderSide, OrderType, Product
from gpt_trader.features.live_trade.execution.decision_trace import OrderDecisionTrace
from gpt_trader.features.live_trade.execution.submission_result import OrderSubmissionStatus
from gpt_trader.features.live_trade.strategies.perps_baseline import Action, Decision


async def _place_order(engine, action: Action = Action.BUY):
    return await engine._validate_and_place_order(
        symbol="BTC-USD",
        decision=Decision(action, "test"),
        price=Decimal("50000"),
        equity=Decimal("10000"),
    )


def _mock_security_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_validator = MagicMock()
    mock_validator.validate_order_request.return_value.is_valid = True
    monkeypatch.setattr(security_validate_module, "get_validator", lambda: mock_validator)


def _setup_pre_trade_validation_block(engine) -> None:
    from gpt_trader.features.live_trade.risk.manager import ValidationError

    engine._order_validator.run_pre_trade_validation.side_effect = ValidationError(
        "Leverage exceeds limit"
    )


def _setup_mark_staleness_block(engine) -> None:
    engine.context.risk_manager.check_mark_staleness.return_value = True
    engine.context.risk_manager.config.mark_staleness_allow_reduce_only = False


@pytest.fixture
def reset_metrics():
    from gpt_trader.monitoring.metrics_collector import reset_all

    reset_all()
    yield
    reset_all()


def test_finalize_decision_trace_records_blocked_metric(engine, reset_metrics) -> None:
    from gpt_trader.monitoring.metrics_collector import get_metrics_collector

    trace = OrderDecisionTrace(
        symbol="BTC-USD",
        side="BUY",
        price=Decimal("50000"),
        equity=Decimal("10000"),
        quantity=Decimal("0.1"),
        reduce_only=False,
        reason="test",
    )

    result = engine._finalize_decision_trace(
        trace,
        status=OrderSubmissionStatus.BLOCKED,
        reason="guard_block",
    )

    assert result.status is OrderSubmissionStatus.BLOCKED
    collector = get_metrics_collector()
    assert collector.counters["gpt_trader_trades_blocked_total"] == 1


@pytest.mark.asyncio
async def test_order_placed_with_dynamic_quantity(engine, monkeypatch: pytest.MonkeyPatch):
    """Test full flow from decision to order placement with calculated size."""
    engine.strategy.decide.return_value = Decision(Action.BUY, "test")
    engine.strategy.config.position_fraction = Decimal("0.1")
    engine.context.broker.get_ticker.return_value = {"price": "50000"}
    engine.context.broker.list_balances.return_value = [
        Balance(asset="USD", total=Decimal("10000"), available=Decimal("10000"))
    ]
    engine.context.broker.list_positions.return_value = []

    mock_validator = MagicMock()
    mock_validator.validate_order_request.return_value.is_valid = True
    monkeypatch.setattr(security_validate_module, "get_validator", lambda: mock_validator)

    await engine._cycle()

    engine._order_submitter.submit_order_with_result.assert_called_once()
    call_kwargs = engine._order_submitter.submit_order_with_result.call_args[1]
    assert call_kwargs["symbol"] == "BTC-USD"
    assert call_kwargs["side"] == OrderSide.BUY
    assert call_kwargs["order_type"] == OrderType.MARKET
    assert call_kwargs["order_quantity"] == Decimal("0.02")


@pytest.mark.asyncio
async def test_mark_staleness_seeded_from_rest_fetch(engine):
    """Test that REST price fetch seeds mark staleness timestamp."""
    engine.context.risk_manager.last_mark_update = {}

    assert "BTC-USD" not in engine.context.risk_manager.last_mark_update

    engine.strategy.decide.return_value = Decision(Action.HOLD, "test")
    engine.context.broker.get_ticker.return_value = {"price": "50000"}
    engine.context.broker.list_balances.return_value = [
        Balance(asset="USD", total=Decimal("10000"), available=Decimal("10000"))
    ]
    engine.context.broker.list_positions.return_value = []

    await engine._cycle()

    assert "BTC-USD" in engine.context.risk_manager.last_mark_update
    assert engine.context.risk_manager.last_mark_update["BTC-USD"] > 0


@pytest.mark.asyncio
async def test_exchange_rules_blocks_small_order(engine, monkeypatch: pytest.MonkeyPatch):
    """Test that exchange rules guard blocks orders below min size."""
    from gpt_trader.core import MarketType
    from gpt_trader.features.live_trade.risk.manager import ValidationError

    engine.strategy.decide.return_value = Decision(Action.BUY, "test")
    engine.strategy.config.position_fraction = Decimal("0.001")
    engine.context.broker.get_ticker.return_value = {"price": "50000"}
    engine.context.broker.list_balances.return_value = [
        Balance(asset="USD", total=Decimal("100"), available=Decimal("100"))
    ]
    engine.context.broker.list_positions.return_value = []

    engine._order_validator = MagicMock()
    engine._order_validator.validate_exchange_rules.side_effect = ValidationError(
        "Order size 0.00002 below minimum 0.0001"
    )

    engine._state_collector = MagicMock()
    engine._state_collector.require_product.return_value = Product(
        symbol="BTC-USD",
        base_asset="BTC",
        quote_asset="USD",
        market_type=MarketType.SPOT,
        min_size=Decimal("0.0001"),
        step_size=Decimal("0.00001"),
        min_notional=Decimal("1"),
        price_increment=Decimal("0.01"),
        leverage_max=None,
    )

    engine._order_submitter = MagicMock()

    mock_validator = MagicMock()
    mock_validator.validate_order_request.return_value.is_valid = True
    monkeypatch.setattr(security_validate_module, "get_validator", lambda: mock_validator)

    await engine._cycle()

    engine.context.broker.place_order.assert_not_called()
    engine._order_submitter.record_rejection.assert_called_once()
    events = [
        event
        for event in engine._event_store.list_events()
        if event.get("type") == "trade_gate_blocked"
    ]
    assert events
    payload = events[-1].get("data", {})
    assert payload.get("gate") == "exchange_rules"


@pytest.mark.asyncio
async def test_slippage_guard_blocks_order(engine, monkeypatch: pytest.MonkeyPatch):
    """Test that slippage guard blocks orders with excessive expected slippage."""
    from gpt_trader.core import MarketType
    from gpt_trader.features.live_trade.risk.manager import ValidationError

    engine.strategy.decide.return_value = Decision(Action.BUY, "test")
    engine.strategy.config.position_fraction = Decimal("0.1")
    engine.context.broker.get_ticker.return_value = {"price": "50000"}
    engine.context.broker.list_balances.return_value = [
        Balance(asset="USD", total=Decimal("10000"), available=Decimal("10000"))
    ]
    engine.context.broker.list_positions.return_value = []

    engine._order_validator = MagicMock()
    engine._order_validator.validate_exchange_rules.return_value = (
        Decimal("0.02"),
        None,
    )
    engine._order_validator.enforce_slippage_guard.side_effect = ValidationError(
        "Expected slippage 150 bps exceeds guard 50"
    )

    engine._state_collector = MagicMock()
    engine._state_collector.require_product.return_value = Product(
        symbol="BTC-USD",
        base_asset="BTC",
        quote_asset="USD",
        market_type=MarketType.SPOT,
        min_size=Decimal("0.0001"),
        step_size=Decimal("0.00001"),
        min_notional=Decimal("1"),
        price_increment=Decimal("0.01"),
        leverage_max=None,
    )

    engine._order_submitter = MagicMock()

    mock_validator = MagicMock()
    mock_validator.validate_order_request.return_value.is_valid = True
    monkeypatch.setattr(security_validate_module, "get_validator", lambda: mock_validator)

    await engine._cycle()

    engine.context.broker.place_order.assert_not_called()
    engine._order_submitter.record_rejection.assert_called_once()


@pytest.mark.asyncio
async def test_stale_mark_pauses_symbol(engine) -> None:
    engine.context.risk_manager.check_mark_staleness.return_value = True
    await _place_order(engine)
    assert engine._degradation.is_paused(symbol="BTC-USD")
    assert "mark_staleness" in (engine._degradation.get_pause_reason("BTC-USD") or "")
    assert any(e.get("type") == "stale_mark_detected" for e in engine._event_store.list_events())
    events = [
        event
        for event in engine._event_store.list_events()
        if event.get("type") == "trade_gate_blocked"
    ]
    assert events
    payload = events[-1].get("data", {})
    assert payload.get("gate") == "mark_staleness"


@pytest.mark.asyncio
async def test_stale_mark_allows_reduce_only_when_configured(engine) -> None:
    engine.context.risk_manager.check_mark_staleness.return_value = True
    engine.context.risk_manager.config.mark_staleness_allow_reduce_only = True
    engine._current_positions = {"BTC-USD": make_position()}
    await _place_order(engine, Action.SELL)
    engine._order_submitter.submit_order_with_result.assert_called()


@pytest.mark.asyncio
async def test_close_signal_submits_reduce_only_exit_for_long_position(
    engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_security_validation(monkeypatch)
    engine.context.risk_manager.is_reduce_only_mode.return_value = False
    engine._order_validator.validate_exchange_rules.side_effect = lambda **kwargs: (
        kwargs["order_quantity"],
        None,
    )
    engine._order_validator.finalize_reduce_only_flag.side_effect = lambda reduce_only, _symbol: (
        reduce_only
    )
    engine._current_positions = {"BTC-USD": make_position(qty="0.75", side="long")}

    await engine._handle_decision(
        symbol="BTC-USD",
        decision=Decision(Action.CLOSE, "exit_long"),
        price=Decimal("50000"),
        equity=Decimal("10000"),
        position_state={
            "quantity": Decimal("0.75"),
            "entry_price": Decimal("40000"),
            "side": "long",
        },
    )

    engine._order_submitter.submit_order_with_result.assert_called_once()
    call_kwargs = engine._order_submitter.submit_order_with_result.call_args[1]
    assert call_kwargs["side"] == OrderSide.SELL
    assert call_kwargs["order_quantity"] == Decimal("0.75")
    assert call_kwargs["reduce_only"] is True


@pytest.mark.asyncio
async def test_close_signal_submits_reduce_only_exit_for_short_position(
    engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_security_validation(monkeypatch)
    engine.context.risk_manager.is_reduce_only_mode.return_value = False
    engine._order_validator.validate_exchange_rules.side_effect = lambda **kwargs: (
        kwargs["order_quantity"],
        None,
    )
    engine._order_validator.finalize_reduce_only_flag.side_effect = lambda reduce_only, _symbol: (
        reduce_only
    )
    engine._current_positions = {"BTC-USD": make_position(qty="0.5", side="short")}

    await engine._handle_decision(
        symbol="BTC-USD",
        decision=Decision(Action.CLOSE, "exit_short"),
        price=Decimal("50000"),
        equity=Decimal("10000"),
        position_state={
            "quantity": Decimal("0.5"),
            "entry_price": Decimal("40000"),
            "side": "short",
        },
    )

    engine._order_submitter.submit_order_with_result.assert_called_once()
    call_kwargs = engine._order_submitter.submit_order_with_result.call_args[1]
    assert call_kwargs["side"] == OrderSide.BUY
    assert call_kwargs["order_quantity"] == Decimal("0.5")
    assert call_kwargs["reduce_only"] is True


@pytest.mark.asyncio
async def test_order_blocked_when_risk_manager_unavailable(
    engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_security_validation(monkeypatch)
    engine.context.risk_manager = None

    result = await engine._validate_and_place_order(
        symbol="BTC-USD",
        decision=Decision(Action.BUY, "test"),
        price=Decimal("50000"),
        equity=Decimal("10000"),
    )

    assert result.status == OrderSubmissionStatus.BLOCKED
    assert result.reason == "risk_manager_unavailable"
    engine._order_submitter.submit_order_with_result.assert_not_called()


def test_resolve_close_order_legacy_signed_quantity_fallback(engine) -> None:
    close_for_short = engine._resolve_close_order({"quantity": Decimal("-0.75")})
    close_for_long = engine._resolve_close_order({"quantity": Decimal("0.75")})

    assert close_for_short == (OrderSide.BUY, Decimal("0.75"))
    assert close_for_long == (OrderSide.SELL, Decimal("0.75"))


@pytest.mark.asyncio
async def test_slippage_failures_pause_symbol_after_threshold(engine) -> None:
    from gpt_trader.features.live_trade.risk.manager import ValidationError

    engine._order_validator.enforce_slippage_guard.side_effect = ValidationError(
        "Slippage too high"
    )
    for _ in range(3):
        await _place_order(engine)
    assert engine._degradation.is_paused(symbol="BTC-USD")


@pytest.mark.asyncio
async def test_preview_disabled_after_threshold_failures(engine) -> None:
    from gpt_trader.features.live_trade.execution.validation import get_failure_tracker

    tracker = get_failure_tracker()
    for _ in range(3):
        tracker.record_failure("order_preview")
    engine._order_validator.enable_order_preview = True
    result = await _place_order(engine)
    assert result.status in (OrderSubmissionStatus.SUCCESS, OrderSubmissionStatus.BLOCKED)
    assert engine._order_validator.enable_order_preview is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "setup_guard, expected_gate, expected_blocked_stage",
    [
        (_setup_pre_trade_validation_block, "pre_trade_validation", "pre_trade_validation"),
        (_setup_mark_staleness_block, "mark_staleness", None),
    ],
)
async def test_guard_block_records_blocked_reason(
    engine,
    monkeypatch: pytest.MonkeyPatch,
    setup_guard,
    expected_gate: str,
    expected_blocked_stage: str | None,
) -> None:
    """Guard blocks should emit telemetry with the blocked reason tag."""
    _mock_security_validation(monkeypatch)
    setup_guard(engine)

    result = await engine._validate_and_place_order(
        symbol="BTC-USD",
        decision=Decision(Action.BUY, "test"),
        price=Decimal("50000"),
        equity=Decimal("10000"),
    )

    assert result.status == OrderSubmissionStatus.BLOCKED
    engine._order_submitter.submit_order_with_result.assert_not_called()

    record_call = engine._order_submitter.record_rejection.call_args
    assert record_call is not None
    assert record_call.args[4] == expected_gate

    events = [
        event
        for event in engine._event_store.list_events()
        if event.get("type") == "trade_gate_blocked"
    ]
    assert events
    payload = events[-1].get("data", {})
    assert payload.get("gate") == expected_gate
    if expected_blocked_stage is not None:
        params = payload.get("params", {})
        assert params.get("blocked_stage") == expected_blocked_stage


@pytest.mark.asyncio
async def test_mark_staleness_allowed_emits_allowed_telemetry(
    engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reduce-only stale mark path should record an allowed telemetry label."""
    _mock_security_validation(monkeypatch)
    engine.context.risk_manager.check_mark_staleness.return_value = True
    engine.context.risk_manager.config.mark_staleness_allow_reduce_only = True
    engine._order_validator.finalize_reduce_only_flag.return_value = True
    engine._current_positions = {"BTC-USD": make_position()}

    result = await engine._validate_and_place_order(
        symbol="BTC-USD",
        decision=Decision(Action.SELL, "test"),
        price=Decimal("50000"),
        equity=Decimal("10000"),
    )

    assert result.status == OrderSubmissionStatus.SUCCESS
    engine._order_submitter.submit_order_with_result.assert_called_once()
    assert result.decision_trace is not None
    assert result.decision_trace.outcomes["mark_staleness"]["status"] == "allowed"
