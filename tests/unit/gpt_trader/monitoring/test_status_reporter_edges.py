"""Edge-case unit tests for StatusReporter update methods."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

import gpt_trader.monitoring.status_reporter as status_reporter_module
from gpt_trader.monitoring.status_reporter import StatusReporter


def test_update_positions_coerces_invalid_and_skips_non_dict() -> None:
    reporter = StatusReporter()

    reporter.update_positions(
        {
            "BTC-PERP": {
                "quantity": "bad",
                "mark_price": "oops",
                "entry_price": None,
                "unrealized_pnl": "1.23",
                "realized_pnl": "not-a-number",
                "side": "LONG",
            },
            "ETH-PERP": "not-a-dict",
        }
    )

    assert "ETH-PERP" not in reporter._positions
    position = reporter._positions["BTC-PERP"]
    assert position["quantity"] == Decimal("0")
    assert position["mark_price"] == Decimal("0")
    assert position["entry_price"] == Decimal("0")
    assert position["realized_pnl"] == Decimal("0")
    assert position["unrealized_pnl"] == Decimal("1.23")
    assert position["side"] == "LONG"


def test_update_orders_invalid_fields_use_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(status_reporter_module.time, "time", lambda: 123.45)

    reporter = StatusReporter()

    reporter.update_orders(
        [
            {
                "order_id": "order-1",
                "product_id": "BTC-USD",
                "side": "BUY",
                "status": "OPEN",
                "size": "bad",
                "price": "bad",
                "filled_size": "bad",
                "average_filled_price": "bad",
                "created_time": "bad",
                "order_configuration": {},
            }
        ]
    )

    assert len(reporter._status.orders) == 1
    order = reporter._status.orders[0]
    assert order.quantity == Decimal("0")
    assert order.price is None
    assert order.filled_quantity == Decimal("0")
    assert order.avg_fill_price is None
    assert order.creation_time == 123.45
    assert order.order_type == "MARKET"
    assert order.time_in_force == "GTC"


def test_update_account_skips_zero_or_invalid_balances() -> None:
    reporter = StatusReporter()

    reporter.update_account(
        balances=[
            {"currency": "BTC", "balance": "bad", "available": "bad", "hold": "bad"},
            {"currency": "ETH", "balance": "0", "available": "0", "hold": "0"},
            {"currency": "USD", "balance": "1.5", "available": "bad", "hold": "bad"},
        ],
        summary={},
    )

    balances = reporter._status.account.balances
    assert len(balances) == 1
    assert balances[0].asset == "USD"
    assert balances[0].total == Decimal("1.5")
    assert balances[0].available == Decimal("0")
    assert balances[0].hold == Decimal("0")


def test_update_strategy_generates_decision_id_and_handles_invalid_timestamp() -> None:
    reporter = StatusReporter()

    reporter.update_strategy(
        active_strategies=["baseline"],
        decisions=[
            {"symbol": "BTC-USD", "action": "BUY", "timestamp": 171.25},
            {"symbol": "ETH-USD", "action": "HOLD", "timestamp": "bad"},
        ],
    )

    decisions = reporter._status.strategy.last_decisions
    assert len(decisions) == 2

    first = decisions[0]
    assert first.decision_id == f"{int(171.25 * 1000)}_BTC-USD"
    assert first.timestamp == 171.25

    second = decisions[1]
    assert second.timestamp == 0.0
    assert second.decision_id == ""


def test_update_equity_swallows_record_gauge_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_record_gauge = MagicMock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(status_reporter_module, "record_gauge", mock_record_gauge)

    reporter = StatusReporter()

    reporter.update_equity(Decimal("123.45"))

    assert reporter._equity == Decimal("123.45")
