"""Tests for TradingEngine daily tracking helpers."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

from gpt_trader.core import Balance


def test_reset_daily_tracking_recomputes_equity(engine):
    """reset_daily_tracking recomputes equity and invalidates guard cache."""
    engine.context.broker.list_balances.return_value = [
        Balance(asset="USD", total=Decimal("1000"), available=Decimal("1000"))
    ]
    engine._state_collector.calculate_equity_from_balances.return_value = (
        Decimal("1000"),
        [],
        Decimal("1000"),
    )
    engine.context.risk_manager.reset_daily_tracking = MagicMock()
    engine._guard_manager = MagicMock()

    engine.reset_daily_tracking()

    engine.context.broker.list_balances.assert_called_once()
    engine._state_collector.calculate_equity_from_balances.assert_called_once()
    engine.context.risk_manager.reset_daily_tracking.assert_called_once()
    engine._guard_manager.invalidate_cache.assert_called_once()
