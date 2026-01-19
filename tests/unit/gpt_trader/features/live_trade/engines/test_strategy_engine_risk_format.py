"""Tests for TradingEngine risk-format helpers."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.core import Balance, Position
from gpt_trader.features.live_trade.strategies.perps_baseline import Action, Decision


def test_positions_to_risk_format(engine):
    """Test conversion of positions to risk manager dict format."""
    positions = {
        "BTC-USD": Position(
            symbol="BTC-USD",
            quantity=Decimal("0.5"),
            entry_price=Decimal("40000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        )
    }

    risk_format = engine._positions_to_risk_format(positions)
    assert "BTC-USD" in risk_format
    assert risk_format["BTC-USD"]["quantity"] == Decimal("0.5")
    assert risk_format["BTC-USD"]["mark"] == Decimal("50000")
    assert not isinstance(risk_format["BTC-USD"], Position)


@pytest.mark.asyncio
async def test_risk_manager_receives_dict_format(engine):
    """Test that risk manager receives correctly formatted dicts."""
    mock_risk_manager = MagicMock()
    mock_risk_manager._start_of_day_equity = Decimal("1000.0")
    mock_risk_manager.check_mark_staleness.return_value = False
    mock_risk_manager.track_daily_pnl.return_value = False
    mock_risk_manager.is_reduce_only_mode.return_value = False
    mock_risk_manager.check_order.return_value = True
    engine.context.risk_manager = mock_risk_manager

    with patch("gpt_trader.security.security_validator.get_validator") as mock_get_validator:
        mock_validator = MagicMock()
        mock_validator.validate_order_request.return_value.is_valid = True
        mock_get_validator.return_value = mock_validator

        engine.strategy.decide.return_value = Decision(Action.BUY, "test")
        engine.strategy.config.position_fraction = Decimal("0.1")

        engine.context.broker.list_positions.return_value = [
            Position(
                symbol="BTC-USD",
                quantity=Decimal("1.0"),
                entry_price=Decimal("40000"),
                mark_price=Decimal("50000"),
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                side="long",
            )
        ]
        engine.context.broker.list_balances.return_value = [
            Balance(asset="USD", total=Decimal("10000"), available=Decimal("10000"))
        ]
        engine._state_collector.build_positions_dict.side_effect = (
            lambda positions: engine._positions_to_risk_format(
                {pos.symbol: pos for pos in positions}
            )
        )

        await engine._cycle()

    engine._order_validator.run_pre_trade_validation.assert_called_once()
    call_args = engine._order_validator.run_pre_trade_validation.call_args
    current_positions = call_args.kwargs["current_positions"]

    assert "BTC-USD" in current_positions
    assert isinstance(current_positions["BTC-USD"], dict)
    assert current_positions["BTC-USD"]["quantity"] == Decimal("1.0")
