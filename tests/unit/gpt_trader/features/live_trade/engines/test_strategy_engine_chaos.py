"""Chaos tests for TradingEngine graceful degradation."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from tests.support.chaos import (
    ChaosBroker,
    api_outage_scenario,
    broker_read_failures_scenario,
    ws_disconnect_scenario,
    ws_gap_scenario,
    ws_reconnect_scenario,
    ws_stale_heartbeat_scenario,
    ws_stale_messages_scenario,
)

from gpt_trader.app.config import BotConfig, BotRiskConfig
from gpt_trader.app.container import (
    ApplicationContainer,
    clear_application_container,
    set_application_container,
)
from gpt_trader.core import Balance, Position
from gpt_trader.features.live_trade.engines.base import CoordinatorContext
from gpt_trader.features.live_trade.engines.strategy import TradingEngine
from gpt_trader.features.live_trade.guard_errors import GuardError
from gpt_trader.features.live_trade.strategies.perps_baseline import Action, Decision


def _make_position(symbol: str = "BTC-USD", qty: str = "1.0", side: str = "long") -> Position:
    return Position(
        symbol=symbol,
        quantity=Decimal(qty),
        entry_price=Decimal("40000"),
        mark_price=Decimal("50000"),
        unrealized_pnl=Decimal("10000"),
        realized_pnl=Decimal("0"),
        side=side,
    )


@pytest.fixture
def mock_broker():
    broker = MagicMock()
    broker.get_ticker.return_value = {"price": "50000"}
    broker.list_balances.return_value = [
        Balance(asset="USD", total=Decimal("10000"), available=Decimal("10000"))
    ]
    broker.list_positions.return_value = [_make_position(qty="0.5")]
    broker.get_resilience_status.return_value = None
    broker.get_market_snapshot.return_value = {"spread_bps": 10, "depth_l1": 10000}
    broker.place_order.return_value = "order-123"
    return broker


@pytest.fixture
def mock_risk_config():
    c = MagicMock()
    c.broker_outage_max_failures, c.broker_outage_cooldown_seconds = 3, 120
    c.mark_staleness_cooldown_seconds, c.mark_staleness_allow_reduce_only = 60, True
    c.slippage_failure_pause_after, c.slippage_pause_seconds = 3, 60
    c.validation_failure_cooldown_seconds, c.preview_failure_disable_after = 180, 3
    c.api_health_cooldown_seconds, c.api_error_rate_threshold, c.api_rate_limit_usage_threshold = (
        300,
        0.2,
        0.9,
    )
    # WS health config
    c.ws_health_interval_seconds = 1  # Fast for tests
    c.ws_message_stale_seconds = 15
    c.ws_heartbeat_stale_seconds = 30
    c.ws_reconnect_pause_seconds = 10
    return c


@pytest.fixture
def context(mock_broker, mock_risk_config):
    risk = BotRiskConfig(position_fraction=Decimal("0.1"))
    config = BotConfig(symbols=["BTC-USD"], interval=1, risk=risk)
    rm = MagicMock()
    rm._start_of_day_equity, rm.check_mark_staleness.return_value = Decimal("10000.0"), False
    rm.is_reduce_only_mode.return_value, rm.config = False, mock_risk_config
    return CoordinatorContext(config=config, broker=mock_broker, risk_manager=rm)


@pytest.fixture
def application_container(context):
    """Set up application container for TradingEngine chaos tests."""
    container = ApplicationContainer(context.config)
    set_application_container(container)
    yield container
    clear_application_container()


@pytest.fixture
def mock_security_validator():
    v, r = MagicMock(), MagicMock()
    r.is_valid, r.errors = True, []
    v.validate_order_request.return_value = r
    with patch("gpt_trader.security.security_validator.get_validator", return_value=v):
        yield v


@pytest.fixture
def engine(context, mock_security_validator, application_container):
    strategy = MagicMock()
    strategy.decide.return_value, strategy.config.position_fraction = Decision(
        Action.HOLD, "test"
    ), Decimal("0.1")
    with patch(
        "gpt_trader.features.live_trade.engines.strategy.create_strategy", return_value=strategy
    ):
        eng = TradingEngine(context)
        eng._state_collector = MagicMock()
        eng._state_collector.require_product.return_value = MagicMock()
        eng._state_collector.resolve_effective_price.return_value = Decimal("50000")
        eng._state_collector.build_positions_dict.return_value = {}
        eng._order_validator = MagicMock()
        eng._order_validator.validate_exchange_rules.return_value = (Decimal("0.02"), None)
        for attr in [
            "enforce_slippage_guard",
            "ensure_mark_is_fresh",
            "run_pre_trade_validation",
            "maybe_preview_order",
        ]:
            setattr(eng._order_validator, attr, MagicMock(return_value=None))
        eng._order_validator.finalize_reduce_only_flag.return_value = False
        eng._order_validator.enable_order_preview = True
        eng._order_submitter = MagicMock()
        yield eng


class TestBrokerOutageDegradation:
    @pytest.mark.asyncio
    async def test_broker_failures_trigger_pause_after_threshold(
        self, engine, mock_broker, mock_risk_config
    ) -> None:
        engine.context.broker = ChaosBroker(mock_broker, broker_read_failures_scenario(times=3))
        await engine._fetch_total_equity({})
        assert engine._degradation._broker_failures == 1
        await engine._fetch_total_equity({})
        assert engine._degradation._broker_failures == 2
        await engine._fetch_total_equity({})
        assert engine._degradation.is_paused() and "broker_outage" in (
            engine._degradation.get_pause_reason() or ""
        )

    @pytest.mark.asyncio
    async def test_successful_broker_call_resets_counter(self, engine, mock_broker) -> None:
        engine._degradation._broker_failures = 2
        await engine._fetch_total_equity({})
        assert engine._degradation._broker_failures == 0


class TestGuardFailureDegradation:
    @pytest.mark.asyncio
    async def test_guard_failure_triggers_pause_and_reduce_only(self, engine, mock_broker) -> None:
        guard_manager = MagicMock()
        guard_manager.run_runtime_guards.side_effect = GuardError(
            guard_name="api_health", message="degraded"
        )
        guard_manager.cancel_all_orders.return_value = 2
        engine._guard_manager, engine.running = guard_manager, True

        async def stop(_):
            engine.running = False
            raise asyncio.CancelledError()

        with patch.object(asyncio, "sleep", stop), pytest.raises(asyncio.CancelledError):
            await engine._runtime_guard_sweep()
        assert engine._degradation.is_paused()
        engine.context.risk_manager.set_reduce_only_mode.assert_called_with(
            True, reason="guard_failure:api_health"
        )

    @pytest.mark.asyncio
    async def test_api_outage_scenario_triggers_degradation(
        self, engine, mock_broker, mock_risk_config
    ) -> None:
        from gpt_trader.features.live_trade.execution.guard_manager import GuardManager

        chaos_broker = ChaosBroker(
            mock_broker, api_outage_scenario(error_rate=0.3, open_breakers=["orders"])
        )
        engine._guard_manager = GuardManager(
            broker=chaos_broker,
            risk_manager=engine.context.risk_manager,
            equity_calculator=lambda b: (Decimal("10000"), b, Decimal("0")),
            open_orders=[],
            invalidate_cache_callback=lambda: None,
        )
        engine.running = True

        async def stop(_):
            engine.running = False
            raise asyncio.CancelledError()

        with patch.object(asyncio, "sleep", stop), pytest.raises(asyncio.CancelledError):
            await engine._runtime_guard_sweep()
        assert engine._degradation.is_paused()


class TestMarkStalenessDegradation:
    async def _place_order(self, engine, action=Action.BUY):
        await engine._validate_and_place_order(
            symbol="BTC-USD",
            decision=Decision(action, "test"),
            price=Decimal("50000"),
            equity=Decimal("10000"),
        )

    @pytest.mark.asyncio
    async def test_stale_mark_pauses_symbol(self, engine) -> None:
        engine.context.risk_manager.check_mark_staleness.return_value = True
        await self._place_order(engine)
        assert engine._degradation.is_paused(symbol="BTC-USD")
        assert "mark_staleness" in (engine._degradation.get_pause_reason("BTC-USD") or "")

    @pytest.mark.asyncio
    async def test_stale_mark_allows_reduce_only_when_configured(self, engine) -> None:
        engine.context.risk_manager.check_mark_staleness.return_value = True
        engine.context.risk_manager.config.mark_staleness_allow_reduce_only = True
        engine._current_positions = {"BTC-USD": _make_position()}
        await self._place_order(engine, Action.SELL)
        engine.context.broker.place_order.assert_called()


class TestSlippageFailureDegradation:
    @pytest.mark.asyncio
    async def test_slippage_failures_pause_symbol_after_threshold(self, engine) -> None:
        from gpt_trader.features.live_trade.risk.manager import ValidationError

        engine._order_validator.enforce_slippage_guard.side_effect = ValidationError(
            "Slippage too high"
        )
        for _ in range(3):
            await engine._validate_and_place_order(
                symbol="BTC-USD",
                decision=Decision(Action.BUY, "test"),
                price=Decimal("50000"),
                equity=Decimal("10000"),
            )
        assert engine._degradation.is_paused(symbol="BTC-USD")


class TestPreviewDisableDegradation:
    @pytest.mark.asyncio
    async def test_preview_disabled_after_threshold_failures(self, engine) -> None:
        from gpt_trader.features.live_trade.execution.validation import get_failure_tracker

        tracker = get_failure_tracker()
        for _ in range(3):
            tracker.record_failure("order_preview")
        engine._order_validator.enable_order_preview = True
        await engine._validate_and_place_order(
            symbol="BTC-USD",
            decision=Decision(Action.BUY, "test"),
            price=Decimal("50000"),
            equity=Decimal("10000"),
        )
        assert engine._order_validator.enable_order_preview is False


class TestPausedOrderRejection:
    @pytest.mark.asyncio
    async def test_order_rejected_when_globally_paused(self, engine) -> None:
        engine._degradation.pause_all(seconds=60, reason="test_pause")
        await engine._validate_and_place_order(
            symbol="BTC-USD",
            decision=Decision(Action.BUY, "test"),
            price=Decimal("50000"),
            equity=Decimal("10000"),
        )
        engine.context.broker.place_order.assert_not_called()
        engine._order_submitter.record_rejection.assert_called()

    @pytest.mark.asyncio
    async def test_order_rejected_when_symbol_paused(self, engine) -> None:
        engine._degradation.pause_symbol("BTC-USD", seconds=60, reason="test_pause")
        await engine._validate_and_place_order(
            symbol="BTC-USD",
            decision=Decision(Action.BUY, "test"),
            price=Decimal("50000"),
            equity=Decimal("10000"),
        )
        engine.context.broker.place_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_reduce_only_allowed_through_pause(self, engine) -> None:
        engine._degradation.pause_all(seconds=60, reason="test", allow_reduce_only=True)
        engine._current_positions = {"BTC-USD": _make_position()}
        await engine._validate_and_place_order(
            symbol="BTC-USD",
            decision=Decision(Action.SELL, "test"),
            price=Decimal("50000"),
            equity=Decimal("10000"),
        )
        engine.context.broker.place_order.assert_called()


class TestWSHealthDegradation:
    """Tests for WebSocket health-triggered degradation."""

    @pytest.mark.asyncio
    async def test_ws_stale_messages_triggers_pause_and_reduce_only(
        self, engine, mock_broker, mock_risk_config
    ) -> None:
        """Stale WS messages should trigger pause + reduce-only mode."""
        # Set up chaos broker that returns stale message health
        engine.context.broker = ChaosBroker(
            mock_broker,
            ws_stale_messages_scenario(stale_age_seconds=30.0),
        )
        engine.running = True
        iterations = 0

        async def stop_after_one(_):
            nonlocal iterations
            iterations += 1
            if iterations >= 1:
                engine.running = False
                raise asyncio.CancelledError()

        with patch.object(asyncio, "sleep", stop_after_one), pytest.raises(asyncio.CancelledError):
            await engine._monitor_ws_health()

        # Verify degradation was triggered
        assert engine._degradation.is_paused()
        assert "ws_message_stale" in (engine._degradation.get_pause_reason() or "")
        engine.context.risk_manager.set_reduce_only_mode.assert_called_with(
            True, reason="ws_message_stale"
        )

    @pytest.mark.asyncio
    async def test_ws_stale_heartbeat_triggers_pause(
        self, engine, mock_broker, mock_risk_config
    ) -> None:
        """Stale WS heartbeat should trigger pause + reduce-only mode."""
        engine.context.broker = ChaosBroker(
            mock_broker,
            ws_stale_heartbeat_scenario(stale_age_seconds=60.0),
        )
        engine.running = True
        iterations = 0

        async def stop_after_one(_):
            nonlocal iterations
            iterations += 1
            if iterations >= 1:
                engine.running = False
                raise asyncio.CancelledError()

        with patch.object(asyncio, "sleep", stop_after_one), pytest.raises(asyncio.CancelledError):
            await engine._monitor_ws_health()

        # Verify degradation was triggered
        assert engine._degradation.is_paused()
        assert "ws_heartbeat_stale" in (engine._degradation.get_pause_reason() or "")

    @pytest.mark.asyncio
    async def test_ws_reconnect_triggers_pause_for_sync(
        self, engine, mock_broker, mock_risk_config
    ) -> None:
        """WS reconnect should trigger a brief pause for state synchronization."""
        engine.context.broker = ChaosBroker(
            mock_broker,
            ws_reconnect_scenario(reconnect_count=1),
        )
        engine.running = True
        iterations = 0

        async def stop_after_one(_):
            nonlocal iterations
            iterations += 1
            if iterations >= 1:
                engine.running = False
                raise asyncio.CancelledError()

        with patch.object(asyncio, "sleep", stop_after_one), pytest.raises(asyncio.CancelledError):
            await engine._monitor_ws_health()

        # Verify pause was triggered for sync
        assert engine._degradation.is_paused()
        assert "ws_reconnect" in (engine._degradation.get_pause_reason() or "")
        # Verify reconnect tracking was reset
        assert engine._ws_reconnect_attempts == 0

    @pytest.mark.asyncio
    async def test_ws_disconnect_with_stale_timestamps_triggers_degradation(
        self, engine, mock_broker, mock_risk_config
    ) -> None:
        """WS disconnect (with stale timestamps) should trigger degradation."""
        engine.context.broker = ChaosBroker(
            mock_broker,
            ws_disconnect_scenario(),
        )
        engine.running = True
        iterations = 0

        async def stop_after_one(_):
            nonlocal iterations
            iterations += 1
            if iterations >= 1:
                engine.running = False
                raise asyncio.CancelledError()

        with patch.object(asyncio, "sleep", stop_after_one), pytest.raises(asyncio.CancelledError):
            await engine._monitor_ws_health()

        # Verify degradation was triggered (due to stale timestamps)
        assert engine._degradation.is_paused()

    @pytest.mark.asyncio
    async def test_ws_gap_count_tracked_in_status(
        self, engine, mock_broker, mock_risk_config
    ) -> None:
        """WS gaps should be tracked and reported in status."""
        engine.context.broker = ChaosBroker(
            mock_broker,
            ws_gap_scenario(gap_count=5),
        )
        engine.running = True
        engine._cycle_count = 60  # Trigger the gap logging condition

        # Mock the status reporter's update_ws_health method
        engine._status_reporter.update_ws_health = MagicMock()

        iterations = 0

        async def stop_after_one(_):
            nonlocal iterations
            iterations += 1
            if iterations >= 1:
                engine.running = False
                raise asyncio.CancelledError()

        with patch.object(asyncio, "sleep", stop_after_one), pytest.raises(asyncio.CancelledError):
            await engine._monitor_ws_health()

        # Verify status reporter was updated with gap count
        engine._status_reporter.update_ws_health.assert_called()
        call_args = engine._status_reporter.update_ws_health.call_args[0][0]
        assert call_args.get("gap_count") == 5

    @pytest.mark.asyncio
    async def test_no_degradation_when_broker_lacks_ws_health(
        self, engine, mock_broker, mock_risk_config
    ) -> None:
        """No degradation when broker doesn't support get_ws_health."""
        # Remove get_ws_health method
        del mock_broker.get_ws_health
        engine.running = True
        iterations = 0

        async def stop_after_one(_):
            nonlocal iterations
            iterations += 1
            if iterations >= 1:
                engine.running = False
                raise asyncio.CancelledError()

        with patch.object(asyncio, "sleep", stop_after_one), pytest.raises(asyncio.CancelledError):
            await engine._monitor_ws_health()

        # No degradation should have been triggered
        assert not engine._degradation.is_paused()

    @pytest.mark.asyncio
    async def test_ws_health_exception_handled_gracefully(
        self, engine, mock_broker, mock_risk_config
    ) -> None:
        """Exceptions in get_ws_health should be handled gracefully."""
        mock_broker.get_ws_health.side_effect = Exception("WS health error")
        engine.running = True
        iterations = 0

        async def stop_after_one(_):
            nonlocal iterations
            iterations += 1
            if iterations >= 1:
                engine.running = False
                raise asyncio.CancelledError()

        # Should not raise, exception should be caught
        with patch.object(asyncio, "sleep", stop_after_one), pytest.raises(asyncio.CancelledError):
            await engine._monitor_ws_health()

        # No degradation from the exception itself
        assert not engine._degradation.is_paused()
