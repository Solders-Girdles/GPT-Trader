"""
Baseline execute_decision behaviours (skips, quantity handling).
"""

from decimal import Decimal
from unittest.mock import Mock

import pytest

from bot_v2.features.live_trade.strategies.perps_baseline import Action, Decision


class TestExecutionCoordinatorExecuteDecisionBasic:
    """Cover early-return and quantity handling branches."""

    @pytest.mark.asyncio
    async def test_skips_in_dry_run(self, execution_coordinator, execution_context, fake_product):
        new_config = execution_context.config.with_overrides(dry_run=True)
        execution_context = execution_context.with_updates(config=new_config)
        decision = Decision(action=Action.BUY, reason="test")

        await execution_coordinator.execute_decision(
            symbol="BTC-PERP",
            decision=decision,
            mark=Decimal("50000"),
            product=fake_product,
            position_state=None,
        )

    @pytest.mark.asyncio
    async def test_handles_missing_runtime_state(
        self, execution_coordinator, execution_context, fake_product
    ):
        execution_context = execution_context.with_updates(runtime_state=None)
        decision = Decision(action=Action.BUY, reason="test")

        await execution_coordinator.execute_decision(
            symbol="BTC-PERP",
            decision=decision,
            mark=Decimal("50000"),
            product=fake_product,
            position_state=None,
        )

    @pytest.mark.asyncio
    async def test_handles_invalid_mark(
        self, execution_coordinator, execution_context, fake_product
    ):
        decision = Decision(action=Action.BUY, reason="test")

        await execution_coordinator.execute_decision(
            symbol="BTC-PERP",
            decision=decision,
            mark=Decimal("0"),
            product=fake_product,
            position_state=None,
        )

    @pytest.mark.asyncio
    async def test_handles_close_without_position(
        self, execution_coordinator, execution_context, fake_product
    ):
        decision = Decision(action=Action.CLOSE, reason="test")

        await execution_coordinator.execute_decision(
            symbol="BTC-PERP",
            decision=decision,
            mark=Decimal("50000"),
            product=fake_product,
            position_state=None,
        )

    @pytest.mark.asyncio
    async def test_calculates_quantity_from_notional(
        self, execution_coordinator, execution_context, fake_product
    ):
        execution_context.runtime_state.exec_engine = Mock()
        execution_context.runtime_state.exec_engine.place_order = Mock(return_value=Mock())
        decision = Decision(action=Action.BUY, target_notional=Decimal("1000"), reason="test")

        await execution_coordinator.execute_decision(
            symbol="BTC-PERP",
            decision=decision,
            mark=Decimal("50000"),
            product=fake_product,
            position_state=None,
        )

        expected_quantity = Decimal("1000") / Decimal("50000")
        execution_context.runtime_state.exec_engine.place_order.assert_called_once()
        call_args = execution_context.runtime_state.exec_engine.place_order.call_args
        assert call_args[1]["quantity"] == expected_quantity

    @pytest.mark.asyncio
    async def test_uses_explicit_quantity(
        self, execution_coordinator, execution_context, fake_product
    ):
        execution_context.runtime_state.exec_engine = Mock()
        execution_context.runtime_state.exec_engine.place_order = Mock(return_value=Mock())
        decision = Decision(action=Action.BUY, quantity=Decimal("0.1"), reason="test")

        await execution_coordinator.execute_decision(
            symbol="BTC-PERP",
            decision=decision,
            mark=Decimal("50000"),
            product=fake_product,
            position_state=None,
        )

        execution_context.runtime_state.exec_engine.place_order.assert_called_once()
        call_args = execution_context.runtime_state.exec_engine.place_order.call_args
        assert call_args[1]["quantity"] == Decimal("0.1")

    @pytest.mark.asyncio
    async def test_skips_when_no_quantity_available(
        self, execution_coordinator, execution_context, fake_product
    ):
        decision = Decision(action=Action.BUY, reason="test")

        await execution_coordinator.execute_decision(
            symbol="BTC-PERP",
            decision=decision,
            mark=Decimal("50000"),
            product=fake_product,
            position_state=None,
        )

    @pytest.mark.asyncio
    async def test_handles_missing_execution_engine(
        self, execution_coordinator, execution_context, fake_product
    ):
        decision = Decision(action=Action.BUY, quantity=Decimal("0.1"), reason="test")

        await execution_coordinator.execute_decision(
            symbol="BTC-PERP",
            decision=decision,
            mark=Decimal("50000"),
            product=fake_product,
            position_state=None,
        )
