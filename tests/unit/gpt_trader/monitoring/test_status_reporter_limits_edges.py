"""Edge-case tests for StatusReporter list limits and balance parsing."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import patch

from gpt_trader.monitoring.status_reporter import StatusReporter


class _BalanceStub:
    def __init__(self, asset: str, total, available, hold):
        self.asset = asset
        self.total = total
        self.available = available
        self.hold = hold


@patch("gpt_trader.monitoring.status_reporter.time.time", return_value=123.0)
def test_add_trade_caps_recent_trades(mock_time) -> None:
    reporter = StatusReporter()

    for i in range(52):
        reporter.add_trade(
            {
                "trade_id": f"trade-{i}",
                "symbol": "BTC-USD",
                "side": "BUY",
                "quantity": "1",
                "price": "100",
                "timestamp": 100 + i,
                "order_id": f"order-{i}",
                "fee": "0.1",
            }
        )

    assert len(reporter._status.trades) == 50
    trade_ids = [trade.trade_id for trade in reporter._status.trades]
    assert "trade-0" not in trade_ids
    assert "trade-1" not in trade_ids
    assert trade_ids[0] == "trade-51"


def test_update_strategy_caps_decisions() -> None:
    reporter = StatusReporter()

    decisions = [{"symbol": "BTC-USD", "action": "HOLD", "timestamp": float(i)} for i in range(55)]
    reporter.update_strategy(active_strategies=["baseline"], decisions=decisions)

    assert len(reporter._status.strategy.last_decisions) == 50
    timestamps = [decision.timestamp for decision in reporter._status.strategy.last_decisions]
    assert timestamps[0] == 5.0
    assert timestamps[-1] == 54.0


def test_update_account_accepts_object_balances_and_coerces_invalid() -> None:
    reporter = StatusReporter()

    reporter.update_account(
        balances=[
            _BalanceStub("USD", "bad", "bad", "bad"),
            _BalanceStub("BTC", "1.25", "oops", None),
        ],
        summary={},
    )

    balances = reporter._status.account.balances
    assert len(balances) == 1
    assert balances[0].asset == "BTC"
    assert balances[0].total == Decimal("1.25")
    assert balances[0].available == Decimal("0")
    assert balances[0].hold == Decimal("0")
