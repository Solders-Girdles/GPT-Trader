"""Tests for daily report PnL analytics functions."""

from __future__ import annotations

from gpt_trader.monitoring.daily_report.analytics import calculate_pnl_metrics


class TestCalculatePnlMetrics:
    """Tests for calculate_pnl_metrics function."""

    def test_returns_none_equity_when_missing(self) -> None:
        result = calculate_pnl_metrics([], {})
        assert result["equity"] is None

    def test_extracts_equity_from_account(self) -> None:
        result = calculate_pnl_metrics(
            [],
            {"account": {"equity": 10000.0}},
        )
        assert result["equity"] == 10000.0

    def test_accumulates_realized_pnl_from_events(self) -> None:
        events = [
            {"type": "pnl_update", "realized_pnl": 100.0},
            {"type": "pnl_update", "realized_pnl": 50.0},
        ]
        result = calculate_pnl_metrics(events, {})
        assert result["realized_pnl"] == 150.0

    def test_uses_last_unrealized_pnl(self) -> None:
        events = [
            {"type": "pnl_update", "unrealized_pnl": 100.0},
            {"type": "pnl_update", "unrealized_pnl": 200.0},
        ]
        result = calculate_pnl_metrics(events, {})
        assert result["unrealized_pnl"] == 200.0

    def test_accumulates_funding_payments(self) -> None:
        events = [
            {"type": "funding_payment", "amount": 10.0},
            {"type": "funding_payment", "amount": -5.0},
        ]
        result = calculate_pnl_metrics(events, {})
        assert result["funding_pnl"] == 5.0

    def test_accumulates_fees_from_fills(self) -> None:
        events = [
            {"type": "fill", "fee": 1.5},
            {"type": "fill", "fee": 2.5},
        ]
        result = calculate_pnl_metrics(events, {})
        assert result["fees_paid"] == 4.0

    def test_calculates_total_pnl(self) -> None:
        events = [
            {"type": "pnl_update", "realized_pnl": 100.0, "unrealized_pnl": 50.0},
        ]
        result = calculate_pnl_metrics(events, {})
        assert result["total_pnl"] == 150.0

    def test_calculates_equity_change(self) -> None:
        events = [
            {"type": "pnl_update", "realized_pnl": 100.0, "unrealized_pnl": 0},
        ]
        result = calculate_pnl_metrics(events, {"account": {"equity": 1100.0}})
        assert result["equity_change"] == 100.0

    def test_calculates_equity_change_pct(self) -> None:
        events = [
            {"type": "pnl_update", "realized_pnl": 100.0, "unrealized_pnl": 0},
        ]
        result = calculate_pnl_metrics(events, {"account": {"equity": 1100.0}})
        assert result["equity_change_pct"] == 10.0

    def test_equity_change_pct_none_when_equity_missing(self) -> None:
        events = [
            {"type": "pnl_update", "realized_pnl": 100.0, "unrealized_pnl": 0},
        ]
        result = calculate_pnl_metrics(events, {})
        assert result["equity_change_pct"] is None

    def test_equity_change_pct_zero_when_no_prev_equity(self) -> None:
        events = [
            {"type": "pnl_update", "realized_pnl": 100.0, "unrealized_pnl": 0},
        ]
        result = calculate_pnl_metrics(events, {"account": {"equity": 100.0}})
        assert result["equity_change_pct"] == 0

    def test_equity_none_when_not_parseable(self) -> None:
        result = calculate_pnl_metrics([], {"account": {"equity": "nope"}})
        assert result["equity"] is None
