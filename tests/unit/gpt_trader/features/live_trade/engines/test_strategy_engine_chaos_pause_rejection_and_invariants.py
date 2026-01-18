"""Chaos tests for TradingEngine pause behavior and guard invariants."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from strategy_engine_chaos_fixtures import make_position

from gpt_trader.features.live_trade.execution.submission_result import OrderSubmissionStatus
from gpt_trader.features.live_trade.strategies.perps_baseline import Action, Decision

pytest_plugins = ["strategy_engine_chaos_fixtures"]


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
        engine._current_positions = {"BTC-USD": make_position()}
        await engine._validate_and_place_order(
            symbol="BTC-USD",
            decision=Decision(Action.SELL, "test"),
            price=Decimal("50000"),
            equity=Decimal("10000"),
        )
        engine._order_submitter.submit_order.assert_called()


class TestGuardOutcomeInvariants:
    @pytest.mark.asyncio
    async def test_degradation_blocks_before_validation(self, engine) -> None:
        engine._degradation.pause_all(seconds=60, reason="test_pause")
        engine._order_validator.run_pre_trade_validation = MagicMock()
        engine._order_validator.validate_exchange_rules = MagicMock()

        result = await engine._validate_and_place_order(
            symbol="BTC-USD",
            decision=Decision(Action.BUY, "test"),
            price=Decimal("50000"),
            equity=Decimal("10000"),
        )

        assert result.status == OrderSubmissionStatus.BLOCKED
        engine._order_validator.run_pre_trade_validation.assert_not_called()
        engine._order_validator.validate_exchange_rules.assert_not_called()
        engine._order_submitter.submit_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_risk_validation_blocked_vs_failed(self, engine) -> None:
        from gpt_trader.features.live_trade.risk.manager import ValidationError

        engine._order_validator.run_pre_trade_validation.side_effect = ValidationError("risk")

        result = await engine._validate_and_place_order(
            symbol="BTC-USD",
            decision=Decision(Action.BUY, "test"),
            price=Decimal("50000"),
            equity=Decimal("10000"),
        )

        assert result.status == OrderSubmissionStatus.BLOCKED
        assert result.reason is not None
        engine._order_submitter.submit_order.assert_not_called()

        engine._order_validator.run_pre_trade_validation.side_effect = Exception("boom")

        result = await engine._validate_and_place_order(
            symbol="BTC-USD",
            decision=Decision(Action.BUY, "test"),
            price=Decimal("50000"),
            equity=Decimal("10000"),
        )

        assert result.status == OrderSubmissionStatus.FAILED
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_reduce_only_requested_allows_stale_mark(self, engine) -> None:
        engine.context.risk_manager.check_mark_staleness.return_value = True
        engine.context.risk_manager.config.mark_staleness_allow_reduce_only = True
        engine._order_validator.finalize_reduce_only_flag.return_value = True
        engine._order_submitter.submit_order.return_value = "order-1"
        engine._current_positions = {"BTC-USD": make_position()}

        result = await engine._validate_and_place_order(
            symbol="BTC-USD",
            decision=Decision(Action.SELL, "test"),
            price=Decimal("50000"),
            equity=Decimal("10000"),
            reduce_only_requested=True,
        )

        assert result.status == OrderSubmissionStatus.SUCCESS
        call_kwargs = engine._order_submitter.submit_order.call_args.kwargs
        assert call_kwargs["reduce_only"] is True
