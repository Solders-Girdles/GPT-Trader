"""Edge-case unit tests for StatusReporter updates and health."""

from __future__ import annotations

import tempfile
import time
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gpt_trader.monitoring.metrics_collector import reset_all
from gpt_trader.monitoring.status_reporter import HEARTBEAT_LAG_METRIC, StatusReporter


class _BalanceStub:
    def __init__(self, asset: str, total, available, hold):
        self.asset = asset
        self.total = total
        self.available = available
        self.hold = hold


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


def test_update_orders_invalid_fields_use_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    import gpt_trader.monitoring.status_reporter as status_reporter

    class _TestClock:
        def time(self) -> float:
            return 123.45

    monkeypatch.setattr(status_reporter, "get_clock", lambda: _TestClock())
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


def test_update_equity_swallows_record_gauge_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    import gpt_trader.monitoring.status_reporter as status_reporter

    mock_record_gauge = MagicMock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(status_reporter, "record_gauge", mock_record_gauge)
    reporter = StatusReporter()

    reporter.update_equity(Decimal("123.45"))

    mock_record_gauge.assert_called()
    assert reporter._equity == Decimal("123.45")


def test_add_trade_caps_recent_trades(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "time", lambda: 123.0)
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


def test_update_ws_health_coerces_invalid_timestamps_and_clamps_future_age(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gpt_trader.monitoring.status_reporter as status_reporter
    from gpt_trader.monitoring.metrics_collector import get_metrics_collector

    class _Clock:
        def time(self) -> float:
            return 1000.0

    reset_all()
    monkeypatch.setattr(status_reporter, "get_clock", lambda: _Clock())
    mock_record_histogram = MagicMock()
    monkeypatch.setattr(status_reporter, "record_histogram", mock_record_histogram)
    reporter = StatusReporter()

    reporter.update_ws_health(
        {
            "connected": True,
            "last_message_ts": True,
            "last_heartbeat_ts": "invalid",
            "last_close_ts": 1010.0,
            "last_error_ts": 1020.0,
            "gap_count": "3",
            "reconnect_count": "2",
        }
    )

    websocket = reporter._status.websocket
    assert websocket.last_message_ts is None
    assert websocket.last_heartbeat_ts is None
    assert websocket.last_close_ts == 1010.0
    assert websocket.last_error_ts == 1020.0
    assert websocket.gap_count == 3
    assert websocket.reconnect_count == 2

    collector = get_metrics_collector()
    assert collector.gauges["gpt_trader_ws_last_message_age_seconds"] == 0.0
    assert collector.gauges["gpt_trader_ws_last_heartbeat_age_seconds"] == 0.0
    assert collector.gauges["gpt_trader_ws_last_close_age_seconds"] == 0.0
    assert collector.gauges["gpt_trader_ws_last_error_age_seconds"] == 0.0
    mock_record_histogram.assert_not_called()


def test_update_ws_health_records_heartbeat_lag_histogram(monkeypatch: pytest.MonkeyPatch) -> None:
    import gpt_trader.monitoring.status_reporter as status_reporter

    class _Clock:
        def time(self) -> float:
            return 2000.0

    monkeypatch.setattr(status_reporter, "get_clock", lambda: _Clock())
    mock_record_histogram = MagicMock()
    monkeypatch.setattr(status_reporter, "record_histogram", mock_record_histogram)

    reporter = StatusReporter()
    heartbeat_ts = 1900.0
    reporter.update_ws_health({"last_heartbeat_ts": heartbeat_ts})

    mock_record_histogram.assert_called_once_with(
        HEARTBEAT_LAG_METRIC,
        100.0,
        buckets=status_reporter.HEARTBEAT_LAG_BUCKETS,
    )


def test_update_risk_normalizes_guard_payloads() -> None:
    reporter = StatusReporter()

    reporter.update_risk(
        max_leverage=3.0,
        daily_loss_limit=2.5,
        current_daily_loss=1.1,
        reduce_only=True,
        reduce_reason="guard_trip",
        guards=[
            {
                "name": "max_slippage",
                "severity": "HIGH",
                "last_triggered": "bad-timestamp",
                "triggered_count": "7",
                "description": "Price impact",
            },
            "manual_guard",
        ],
    )

    guards = reporter._status.risk.guards
    assert len(guards) == 2
    assert guards[0].name == "max_slippage"
    assert guards[0].severity == "HIGH"
    assert guards[0].last_triggered == 0.0
    assert guards[0].triggered_count == 7
    assert guards[1].name == "manual_guard"

    reporter.update_risk(
        max_leverage=3.0,
        daily_loss_limit=2.5,
        current_daily_loss=1.1,
        reduce_only=False,
        reduce_reason="",
        active_guards=["cooldown_guard"],
    )
    assert [guard.name for guard in reporter._status.risk.guards] == ["cooldown_guard"]


class TestStatusReporterHealth:
    """Tests for StatusReporter health assessment."""

    @pytest.mark.asyncio
    async def test_healthy_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = Path(tmpdir) / "status.json"
            reporter = StatusReporter(status_file=str(status_file))

            reset_all()
            await reporter.start()
            reporter.update_price("BTC-USD", Decimal("50000"))
            try:
                status = reporter.get_status()
                assert status.healthy is True
                assert status.health_issues == []
            finally:
                await reporter.stop()

    def test_unhealthy_recent_error(self) -> None:
        reporter = StatusReporter()
        reporter._running = True
        reporter._start_time = time.time()

        reporter.record_error("Something went wrong")
        reporter.update_price("BTC-USD", Decimal("50000"))

        status = reporter.get_status()
        assert status.healthy is False
        assert any("Recent error" in issue for issue in status.health_issues)

    def test_unhealthy_stale_prices(self) -> None:
        reporter = StatusReporter()
        reporter._running = True
        reporter._start_time = time.time()

        reporter._last_prices["BTC-USD"] = Decimal("50000")
        reporter._last_price_update = time.time() - 300

        status = reporter.get_status()
        assert status.healthy is False
        assert any("Stale prices" in issue for issue in status.health_issues)
