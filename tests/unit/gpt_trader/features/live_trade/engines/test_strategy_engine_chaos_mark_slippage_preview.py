"""Chaos tests for TradingEngine symbol-level degradation cases."""

from __future__ import annotations

from decimal import Decimal

import pytest
from strategy_engine_chaos_fixtures import make_position

from gpt_trader.features.live_trade.execution.submission_result import OrderSubmissionStatus
from gpt_trader.features.live_trade.strategies.perps_baseline import Action, Decision

pytest_plugins = ["strategy_engine_chaos_fixtures"]


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
        events = engine._event_store.list_events()
        assert any(e.get("type") == "stale_mark_detected" for e in events)

    @pytest.mark.asyncio
    async def test_stale_mark_allows_reduce_only_when_configured(self, engine) -> None:
        engine.context.risk_manager.check_mark_staleness.return_value = True
        engine.context.risk_manager.config.mark_staleness_allow_reduce_only = True
        engine._current_positions = {"BTC-USD": make_position()}
        await self._place_order(engine, Action.SELL)
        engine._order_submitter.submit_order.assert_called()


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
        result = await engine._validate_and_place_order(
            symbol="BTC-USD",
            decision=Decision(Action.BUY, "test"),
            price=Decimal("50000"),
            equity=Decimal("10000"),
        )
        assert result.status in (OrderSubmissionStatus.SUCCESS, OrderSubmissionStatus.BLOCKED)
        assert engine._order_validator.enable_order_preview is False
