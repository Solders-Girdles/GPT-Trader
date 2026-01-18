"""Chaos tests for TradingEngine guard failures and kill switch behavior."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from tests.support.chaos import ChaosBroker, api_outage_scenario

from gpt_trader.features.live_trade.execution.submission_result import OrderSubmissionStatus
from gpt_trader.features.live_trade.guard_errors import GuardError
from gpt_trader.features.live_trade.strategies.perps_baseline import Action, Decision

pytest_plugins = ["strategy_engine_chaos_fixtures"]


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
        events = engine._event_store.list_events()
        assert any(e.get("type") == "guard_triggered" for e in events)

    @pytest.mark.asyncio
    async def test_api_health_guard_emits_api_error_event(self, engine) -> None:
        from gpt_trader.features.live_trade.guard_errors import RiskLimitExceeded

        err = RiskLimitExceeded(
            guard_name="api_health",
            message="API health degraded",
            details={"error_rate": 0.5},
        )
        await engine._handle_guard_failure(err)
        events = engine._event_store.list_events()
        assert any(
            e.get("type") == "guard_triggered" and e.get("data", {}).get("guard") == "api_health"
            for e in events
        )
        assert any(e.get("type") == "api_error" for e in events)


@pytest.mark.asyncio
async def test_kill_switch_blocks_submission(engine) -> None:
    engine.context.risk_manager.config.kill_switch_enabled = True

    engine._order_submitter.submit_order = MagicMock(return_value="order-123")

    decision = Decision(Action.BUY, "kill switch test")
    result = await engine._validate_and_place_order(
        symbol="BTC-USD",
        decision=decision,
        price=Decimal("50000"),
        equity=Decimal("10000"),
        quantity_override=Decimal("1"),
        reduce_only_requested=False,
    )

    assert result.status == OrderSubmissionStatus.BLOCKED
    assert result.reason == "kill_switch"
    assert result.decision_trace is not None
    assert result.decision_trace.decision_id
    assert result.decision_trace.outcomes["kill_switch"]["status"] == "blocked"
    engine._order_submitter.submit_order.assert_not_called()
    engine._order_submitter.record_rejection.assert_called_with(
        "BTC-USD",
        "BUY",
        Decimal("0"),
        Decimal("50000"),
        "kill_switch",
        client_order_id=result.decision_trace.decision_id,
    )
